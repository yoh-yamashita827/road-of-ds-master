import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import japanize_matplotlib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

class FactoryDataExplorer:
    """工場データの包括的な探索分析クラス"""
    
    def __init__(self, data_path):
        """
        データ読み込みと初期化
        
        Args:
            data_path (str): CSVファイルのパス
        """
        self.data = pd.read_csv(data_path)
        self.original_data = self.data.copy()
        self.findings = []  # 発見事項を記録
        
        # 日時列の自動検出・変換
        self._detect_datetime_columns()
        
        print(f"データ読み込み完了: {self.data.shape[0]}行 x {self.data.shape[1]}列")
        
    def _detect_datetime_columns(self):
        """日時列の自動検出と変換"""
        datetime_candidates = []
        
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    # 日時変換を試行
                    pd.to_datetime(self.data[col].head(100))
                    datetime_candidates.append(col)
                except:
                    continue
        
        if datetime_candidates:
            print(f"日時列候補: {datetime_candidates}")
            for col in datetime_candidates:
                try:
                    self.data[col] = pd.to_datetime(self.data[col])
                    print(f"  {col} を日時型に変換しました")
                except:
                    continue
    
    def basic_info(self):
        """基本的なデータ情報の表示"""
        print("=" * 60)
        print("【基本データ情報】")
        print("=" * 60)
        
        # データ形状
        print(f"データ形状: {self.data.shape}")
        print(f"メモリ使用量: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print()
        
        # 列情報
        print("【列情報】")
        info_df = pd.DataFrame({
            '列名': self.data.columns,
            'データ型': self.data.dtypes,
            '欠損値数': self.data.isnull().sum(),
            '欠損率(%)': (self.data.isnull().sum() / len(self.data) * 100).round(2),
            'ユニーク数': [self.data[col].nunique() for col in self.data.columns],
            'サンプル値': [str(self.data[col].iloc[0])[:20] for col in self.data.columns]
        })
        print(info_df.to_string(index=False))
        
        # データ型別サマリー
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
        
        print(f"\n数値列: {len(numeric_cols)}個")
        print(f"カテゴリ列: {len(categorical_cols)}個") 
        print(f"日時列: {len(datetime_cols)}個")
        
        return info_df
    
    def statistical_summary(self):
        """統計的サマリーの詳細分析"""
        print("\n" + "=" * 60)
        print("【統計的サマリー】")
        print("=" * 60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("数値列が見つかりません")
            return
        
        # 基本統計量
        stats_df = self.data[numeric_cols].describe()
        print("基本統計量:")
        print(stats_df.round(3))
        
        # 追加統計量
        additional_stats = pd.DataFrame(index=numeric_cols)
        additional_stats['歪度'] = self.data[numeric_cols].skew()
        additional_stats['尖度'] = self.data[numeric_cols].kurtosis()
        additional_stats['変動係数'] = self.data[numeric_cols].std() / self.data[numeric_cols].mean()
        additional_stats['ゼロ値数'] = (self.data[numeric_cols] == 0).sum()
        additional_stats['負値数'] = (self.data[numeric_cols] < 0).sum()
        
        print("\n追加統計量:")
        print(additional_stats.round(3))
        
        # 特徴的な発見の記録
        for col in numeric_cols:
            skew_val = self.data[col].skew()
            if abs(skew_val) > 2:
                self.findings.append(f"{col}: 強い歪み (歪度={skew_val:.3f})")
            
            cv = self.data[col].std() / self.data[col].mean()
            if cv > 1:
                self.findings.append(f"{col}: 高い変動性 (変動係数={cv:.3f})")
        
        return stats_df, additional_stats
    
    def missing_value_analysis(self):
        """欠損値の詳細分析"""
        print("\n" + "=" * 60)
        print("【欠損値分析】")
        print("=" * 60)
        
        missing_summary = pd.DataFrame({
            '列名': self.data.columns,
            '欠損値数': self.data.isnull().sum(),
            '欠損率(%)': (self.data.isnull().sum() / len(self.data) * 100).round(2)
        })
        missing_summary = missing_summary[missing_summary['欠損値数'] > 0].sort_values('欠損率(%)', ascending=False)
        
        if len(missing_summary) == 0:
            print("欠損値は見つかりませんでした")
            return
        
        print("欠損値サマリー:")
        print(missing_summary.to_string(index=False))
        
        # 欠損パターンの分析
        if len(missing_summary) > 1:
            print("\n欠損パターン分析:")
            missing_patterns = self.data[missing_summary['列名']].isnull()
            pattern_counts = missing_patterns.value_counts()
            print(pattern_counts.head(10))
        
        # 欠損値の可視化
        if len(missing_summary) > 0:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            missing_summary.plot(x='列名', y='欠損率(%)', kind='bar', ax=plt.gca())
            plt.title('列別欠損率')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 欠損値のヒートマップ
            plt.subplot(1, 2, 2)
            missing_data = self.data[missing_summary['列名']].isnull()
            if len(missing_data.columns) > 1:
                sns.heatmap(missing_data.corr(), annot=True, cmap='YlOrRd')
                plt.title('欠損値間の相関')
            
            plt.tight_layout()
            plt.show()
        
        return missing_summary
    
    def time_series_analysis(self):
        """時系列データの分析"""
        print("\n" + "=" * 60)
        print("【時系列分析】")
        print("=" * 60)
        
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
        
        if len(datetime_cols) == 0:
            print("日時列が見つかりません")
            return
        
        for time_col in datetime_cols:
            print(f"\n--- {time_col} の分析 ---")
            
            # 基本情報
            print(f"期間: {self.data[time_col].min()} ～ {self.data[time_col].max()}")
            print(f"データ数: {len(self.data)}件")
            
            # 時間間隔の分析
            time_diffs = self.data[time_col].diff().dropna()
            print(f"平均間隔: {time_diffs.mean()}")
            print(f"最小間隔: {time_diffs.min()}")
            print(f"最大間隔: {time_diffs.max()}")
            
            # 重複タイムスタンプの確認
            duplicates = self.data[time_col].duplicated().sum()
            if duplicates > 0:
                print(f"重複タイムスタンプ: {duplicates}件")
                self.findings.append(f"{time_col}: 重複タイムスタンプが{duplicates}件存在")
            
            # 時間間隔の不規則性チェック
            if time_diffs.std() / time_diffs.mean() > 0.1:
                self.findings.append(f"{time_col}: 不規則な時間間隔")
            
            # 時系列の可視化
            self._plot_time_series_patterns(time_col)
    
    def _plot_time_series_patterns(self, time_col):
        """時系列パターンの可視化"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return
        
        # 代表的な数値列を選択（最初の4列まで）
        plot_cols = numeric_cols[:4]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, col in enumerate(plot_cols):
            if i < len(axes):
                self.data.plot(x=time_col, y=col, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col} の時系列変化')
                axes[i].tick_params(axis='x', rotation=45)
        
        # 余った軸を非表示
        for i in range(len(plot_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def outlier_detection(self):
        """外れ値の検出と分析"""
        print("\n" + "=" * 60)
        print("【外れ値検出】")
        print("=" * 60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("数値列が見つかりません")
            return
        
        outlier_summary = pd.DataFrame(index=numeric_cols)
        
        for col in numeric_cols:
            data_col = self.data[col].dropna()
            
            # IQR法
            Q1 = data_col.quantile(0.25)
            Q3 = data_col.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((data_col < lower_bound) | (data_col > upper_bound)).sum()
            
            # Z-score法
            z_scores = np.abs(stats.zscore(data_col))
            z_outliers = (z_scores > 3).sum()
            
            outlier_summary.loc[col, 'IQR外れ値数'] = iqr_outliers
            outlier_summary.loc[col, 'IQR外れ値率(%)'] = (iqr_outliers / len(data_col) * 100).round(2)
            outlier_summary.loc[col, 'Z-score外れ値数'] = z_outliers
            outlier_summary.loc[col, 'Z-score外れ値率(%)'] = (z_outliers / len(data_col) * 100).round(2)
            
            # 発見事項の記録
            if iqr_outliers / len(data_col) > 0.05:  # 5%以上
                self.findings.append(f"{col}: 多数の外れ値 (IQR法で{iqr_outliers}件, {iqr_outliers/len(data_col)*100:.1f}%)")
        
        print("外れ値サマリー:")
        print(outlier_summary)
        
        # 外れ値の可視化
        self._plot_outliers(numeric_cols[:6])  # 最初の6列まで
        
        return outlier_summary
    
    def _plot_outliers(self, columns):
        """外れ値の可視化"""
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                self.data.boxplot(column=col, ax=axes[i])
                axes[i].set_title(f'{col} のボックスプロット')
        
        # 余った軸を非表示
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def correlation_analysis(self):
        """相関分析"""
        print("\n" + "=" * 60)
        print("【相関分析】")
        print("=" * 60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("相関分析に十分な数値列がありません")
            return
        
        # 相関行列の計算
        corr_matrix = self.data[numeric_cols].corr()
        
        # 高い相関を持つペアの抽出
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # 絶対値0.7以上
                    high_corr_pairs.append({
                        '変数1': corr_matrix.columns[i],
                        '変数2': corr_matrix.columns[j],
                        '相関係数': corr_val
                    })
        
        if high_corr_pairs:
            print("高い相関を持つ変数ペア (|r| > 0.7):")
            high_corr_df = pd.DataFrame(high_corr_pairs)
            print(high_corr_df.sort_values('相関係数', key=abs, ascending=False))
            
            # 発見事項の記録
            for pair in high_corr_pairs:
                self.findings.append(f"高相関: {pair['変数1']} - {pair['変数2']} (r={pair['相関係数']:.3f})")
        
        # 相関ヒートマップ
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('変数間の相関ヒートマップ')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    def distribution_analysis(self):
        """分布の分析"""
        print("\n" + "=" * 60)
        print("【分布分析】")
        print("=" * 60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("数値列が見つかりません")
            return
        
        # 分布の可視化
        n_cols = min(len(numeric_cols), 9)  # 最大9個まで
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = [axes]
        axes = np.array(axes).flatten()
        
        for i, col in enumerate(numeric_cols[:n_cols]):
            data_col = self.data[col].dropna()
            
            # ヒストグラム + 密度推定
            axes[i].hist(data_col, bins=50, alpha=0.7, density=True)
            data_col.plot.density(ax=axes[i], color='red', linewidth=2)
            axes[i].set_title(f'{col} の分布')
            axes[i].set_ylabel('密度')
            
            # 正規性の検定
            if len(data_col) > 3:
                _, p_value = stats.shapiro(data_col.sample(min(5000, len(data_col))))
                if p_value < 0.05:
                    self.findings.append(f"{col}: 非正規分布 (Shapiro-Wilk p={p_value:.3e})")
        
        # 余った軸を非表示
        for i in range(n_cols, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def categorical_analysis(self):
        """カテゴリ変数の分析"""
        print("\n" + "=" * 60)
        print("【カテゴリ変数分析】")
        print("=" * 60)
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
        
        # 日時列をカテゴリ列から除外
        categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
        
        if len(categorical_cols) == 0:
            print("カテゴリ列が見つかりません")
            return
        
        for col in categorical_cols:
            print(f"\n--- {col} の分析 ---")
            
            value_counts = self.data[col].value_counts()
            print(f"ユニーク値数: {len(value_counts)}")
            print(f"最頻値: {value_counts.index[0]} ({value_counts.iloc[0]}件)")
            
            # 分布の偏り
            if len(value_counts) > 1:
                concentration = value_counts.iloc[0] / len(self.data)
                if concentration > 0.9:
                    self.findings.append(f"{col}: 高度に偏った分布 (最頻値が{concentration*100:.1f}%)")
            
            # 上位10件まで表示
            print("値の分布:")
            print(value_counts.head(10))
            
            # 可視化（カテゴリ数が適度な場合）
            if 2 <= len(value_counts) <= 20:
                plt.figure(figsize=(10, 6))
                value_counts.head(10).plot(kind='bar')
                plt.title(f'{col} の分布')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.show()
    
    def data_quality_check(self):
        """データ品質の総合チェック"""
        print("\n" + "=" * 60)
        print("【データ品質チェック】")
        print("=" * 60)
        
        quality_issues = []
        
        # 1. 重複行の確認
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"重複行: {duplicates}件")
        
        # 2. 完全に同じ値を持つ列の確認
        constant_cols = []
        for col in self.data.columns:
            if self.data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            quality_issues.append(f"定数列: {constant_cols}")
        
        # 3. 異常に高い/低い値の確認
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data_col = self.data[col].dropna()
            if len(data_col) > 0:
                # 極端に大きな値
                if data_col.max() > data_col.median() * 1000:
                    quality_issues.append(f"{col}: 極端に大きな値 (最大値={data_col.max():.2e})")
                
                # 負の値（物理的におかしい場合）
                if data_col.min() < 0 and 'temp' in col.lower():  # 温度の例
                    quality_issues.append(f"{col}: 負の温度値")
        
        # 4. 時系列の連続性チェック
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
        for time_col in datetime_cols:
            if len(self.data) > 1:
                time_diffs = self.data[time_col].diff().dropna()
                if time_diffs.min() <= pd.Timedelta(0):
                    quality_issues.append(f"{time_col}: 時系列の逆転または停止")
        
        # 結果の表示
        if quality_issues:
            print("発見されたデータ品質の問題:")
            for issue in quality_issues:
                print(f"  ⚠ {issue}")
                self.findings.append(f"品質問題: {issue}")
        else:
            print("重大なデータ品質問題は見つかりませんでした")
        
        return quality_issues
    
    def pattern_detection(self):
        """パターンの検出"""
        print("\n" + "=" * 60)
        print("【パターン検出】")
        print("=" * 60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("パターン検出に十分なデータがありません")
            return
        
        # PCA分析
        data_scaled = StandardScaler().fit_transform(self.data[numeric_cols].fillna(0))
        pca = PCA()
        pca_result = pca.fit_transform(data_scaled)
        
        # 累積寄与率
        cumsum_ratio = np.cumsum(pca.explained_variance_ratio_)
        n_components_80 = np.argmax(cumsum_ratio >= 0.8) + 1
        n_components_95 = np.argmax(cumsum_ratio >= 0.95) + 1
        
        print(f"主成分分析結果:")
        print(f"  80%の分散を説明するのに必要な主成分数: {n_components_80}")
        print(f"  95%の分散を説明するのに必要な主成分数: {n_components_95}")
        
        # 寄与率の可視化
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.bar(range(1, min(11, len(pca.explained_variance_ratio_)+1)), 
                pca.explained_variance_ratio_[:10])
        plt.title('主成分の寄与率')
        plt.xlabel('主成分')
        plt.ylabel('寄与率')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1, min(11, len(cumsum_ratio)+1)), cumsum_ratio[:10], 'o-')
        plt.axhline(y=0.8, color='r', linestyle='--', label='80%')
        plt.axhline(y=0.95, color='g', linestyle='--', label='95%')
        plt.title('累積寄与率')
        plt.xlabel('主成分数')
        plt.ylabel('累積寄与率')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # クラスタリング分析
        if len(self.data) > 10:
            optimal_k = min(8, len(self.data) // 10)
            if optimal_k >= 2:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                clusters = kmeans.fit_predict(data_scaled)
                
                print(f"\nクラスタリング結果 (k={optimal_k}):")
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                print(cluster_counts)
                
                if optimal_k <= 10:  # クラスタ数が適度な場合のみ記録
                    self.findings.append(f"データは{optimal_k}個のクラスタに分類可能")
        
        return pca, cumsum_ratio
    
    def anomaly_scoring(self):
        """異常スコアの計算"""
        print("\n" + "=" * 60)
        print("【異常度スコアリング】")
        print("=" * 60)
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            print("数値列が見つかりません")
            return
        
        # マルティバリエート異常検出
        from sklearn.ensemble import IsolationForest
        
        # 欠損値を平均で埋める
        data_filled = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
        
        # Isolation Forestによる異常検出
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomaly_scores = iso_forest.fit_predict(data_filled)
        anomaly_score_values = iso_forest.score_samples(data_filled)
        
        # 異常データのサマリー
        n_anomalies = (anomaly_scores == -1).sum()
        print(f"検出された異常データ: {n_anomalies}件 ({n_anomalies/len(self.data)*100:.2f}%)")
        
        # 異常スコアの分布
        plt.figure(figsize=(10, 6))
        plt.hist(anomaly_score_values, bins=50, alpha=0.7)
        plt.axvline(x=np.percentile(anomaly_score_values, 10), color='r', 
                   linestyle='--', label='下位10%閾値')
        plt.title('異常度スコアの分布')
        plt.xlabel('異常度スコア（低いほど異常）')
        plt.ylabel('頻度')
        plt.legend()
        plt.show()
        
        # 最も異常なデータポイントを表示
        anomaly_indices = np.argsort(anomaly_score_values)[:5]
        print("\n最も異常なデータポイント（上位5件）:")
        for i, idx in enumerate(anomaly_indices):
            print(f"{i+1}. インデックス {idx}: スコア = {anomaly_score_values[idx]:.3f}")
        
        if n_anomalies > len(self.data) * 0.05:  # 5%以上が異常
            self.findings.append(f"多数の異常データが検出 ({n_anomalies}件, {n_anomalies/len(self.data)*100:.1f}%)")
        
        return anomaly_scores, anomaly_score_values
    
    def generate_insights_report(self):
        """発見事項のレポート生成"""
        print("\n" + "=" * 80)
        print("【総合レポート - 発見事項とヒアリング推奨項目】")
        print("=" * 80)
        
        if not self.findings:
            print("特筆すべき発見事項はありませんでした")
            return
        
        print("■ データ分析で発見された特徴的な点:")
        for i, finding in enumerate(self.findings, 1):
            print(f"{i:2d}. {finding}")
        
        print("\n■ ヒアリング推奨項目:")
        
        # カテゴリ別の推奨質問を生成
        questions = []
        
        # 欠損値関連
        missing_findings = [f for f in self.findings if '欠損' in f]
        if missing_findings:
            questions.append("欠損値の発生原因と処理方針について")
            questions.append("欠損期間中の設備の状況について")
        
        # 外れ値関連
        outlier_findings = [f for f in self.findings if '外れ値' in f]
        if outlier_findings:
            questions.append("外れ値が発生する典型的な原因について")
            questions.append("設備異常やメンテナンス作業との関連について")
        
        # 相関関連
        corr_findings = [f for f in self.findings if '相関' in f]
        if corr_findings:
            questions.append("高い相関を示す変数間の因果関係について")
            questions.append("工程制御のロジックについて")
        
        # 時系列関連
        time_findings = [f for f in self.findings if '時間' in f or '時系列' in f]
        if time_findings:
            questions.append("サンプリング頻度の変更履歴について")
            questions.append("データ収集システムの仕様について")
        
        # 分布関連
        dist_findings = [f for f in self.findings if '分布' in f or '歪み' in f]
        if dist_findings:
            questions.append("変数の正常値範囲について")
            questions.append("運転条件や設備の制約について")
        
        # 品質関連
        quality_findings = [f for f in self.findings if '品質' in f]
        if quality_findings:
            questions.append("データ収集・記録プロセスについて")
            questions.append("計測器の校正履歴について")
        
        # 一般的な推奨質問
        questions.extend([
            "データの前処理や補正の方法について",
            "目的変数（水分値）に影響する主要因子について",
            "30分の遅れ時間の妥当性について",
            "予測精度の要求レベルについて",
            "運用時の更新頻度や許容できる計算時間について"
        ])
        
        for i, question in enumerate(questions, 1):
            print(f"{i:2d}. {question}")
        
        print(f"\n発見事項総数: {len(self.findings)}件")
        print(f"推奨ヒアリング項目: {len(questions)}件")
    
    def run_comprehensive_analysis(self):
        """包括的な分析の実行"""
        print("🔍 工場データの包括的探索分析を開始します...")
        
        # 順次分析を実行
        self.basic_info()
        self.statistical_summary()
        self.missing_value_analysis()
        self.time_series_analysis()
        self.outlier_detection()
        self.correlation_analysis()
        self.distribution_analysis()
        self.categorical_analysis()
        self.data_quality_check()
        self.pattern_detection()
        self.anomaly_scoring()
        
        # 最終レポート
        self.generate_insights_report()
        
        print("\n✅ 包括的分析が完了しました")
        return self.findings

# 使用例
if __name__ == "__main__":
    # データファイルのパスを指定
    data_path = "factory_data.csv"  # 実際のファイルパスに変更してください
    
    try:
        # 分析器の初期化
        explorer = FactoryDataExplorer(data_path)
        
        # 包括的分析の実行
        findings = explorer.run_comprehensive_analysis()
        
        print(f"\n📊 分析完了: {len(findings)}個の発見事項が記録されました")
        
    except FileNotFoundError:
        print(f"エラー: ファイル '{data_path}' が見つかりません")
        print("正しいファイルパスを指定してください")
    except Exception as e:
        print(f"エラーが発生しました: {e}") 