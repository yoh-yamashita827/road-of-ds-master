import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import japanize_matplotlib

warnings.filterwarnings('ignore')

class AdvancedFactoryDataInsights:
    """工場データの高度な洞察分析クラス"""
    
    def __init__(self, data, datetime_col=None):
        """
        初期化
        
        Args:
            data (pd.DataFrame): 分析対象データ
            datetime_col (str): 日時列名（自動検出も可能）
        """
        self.data = data.copy()
        self.datetime_col = datetime_col
        self.insights = []
        
        if self.datetime_col is None:
            self._auto_detect_datetime_col()
        
        # 数値列とカテゴリ列の識別
        self.numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        
        print(f"高度分析準備完了: {len(self.numeric_cols)}個の数値列, {len(self.categorical_cols)}個のカテゴリ列")
    
    def _auto_detect_datetime_col(self):
        """日時列の自動検出"""
        datetime_cols = self.data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            self.datetime_col = datetime_cols[0]
            print(f"日時列として {self.datetime_col} を使用します")
        else:
            print("日時列が見つかりません。一部の時系列分析はスキップされます。")
    
    def operational_pattern_analysis(self):
        """稼働パターンの分析"""
        print("\n" + "=" * 60)
        print("【稼働パターン分析】")
        print("=" * 60)
        
        if self.datetime_col is None:
            print("日時列がないため、稼働パターン分析をスキップします")
            return
        
        # 時間別の特徴
        df_time = self.data.copy()
        df_time['hour'] = df_time[self.datetime_col].dt.hour
        df_time['day_of_week'] = df_time[self.datetime_col].dt.dayofweek
        df_time['day_of_month'] = df_time[self.datetime_col].dt.day
        df_time['month'] = df_time[self.datetime_col].dt.month
        
        # 代表的な数値列を選択
        target_cols = self.numeric_cols[:3] if len(self.numeric_cols) >= 3 else self.numeric_cols
        
        if len(target_cols) == 0:
            print("分析対象の数値列がありません")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 時間別パターン
        hourly_stats = df_time.groupby('hour')[target_cols].mean()
        hourly_stats.plot(ax=axes[0,0], marker='o')
        axes[0,0].set_title('時間別平均値パターン')
        axes[0,0].set_xlabel('時間')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 曜日別パターン
        daily_stats = df_time.groupby('day_of_week')[target_cols].mean()
        daily_stats.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('曜日別平均値パターン')
        axes[0,1].set_xlabel('曜日 (0=月曜)')
        axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 月別パターン
        monthly_stats = df_time.groupby('month')[target_cols].mean()
        monthly_stats.plot(ax=axes[1,0], marker='o')
        axes[1,0].set_title('月別平均値パターン')
        axes[1,0].set_xlabel('月')
        axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # データ密度（稼働率）
        hourly_counts = df_time.groupby('hour').size()
        hourly_counts.plot(kind='bar', ax=axes[1,1], color='orange')
        axes[1,1].set_title('時間別データ数（稼働密度）')
        axes[1,1].set_xlabel('時間')
        axes[1,1].set_ylabel('データ数')
        
        plt.tight_layout()
        plt.show()
        
        # 稼働パターンの特徴を記録
        peak_hour = hourly_counts.idxmax()
        min_hour = hourly_counts.idxmin()
        self.insights.append(f"ピーク稼働時間: {peak_hour}時")
        self.insights.append(f"最低稼働時間: {min_hour}時")
        
        # 週末の稼働状況
        weekend_ratio = df_time[df_time['day_of_week'].isin([5, 6])].shape[0] / len(df_time)
        if weekend_ratio < 0.1:
            self.insights.append("週末の稼働は限定的（平日中心の運転）")
        
        return df_time
    
    def periodicity_detection(self):
        """周期性の検出"""
        print("\n" + "=" * 60)
        print("【周期性検出】")
        print("=" * 60)
        
        if len(self.numeric_cols) == 0:
            print("数値列がありません")
            return
        
        # 代表的な列を選択
        target_cols = self.numeric_cols[:4]
        
        fig, axes = plt.subplots(len(target_cols), 2, figsize=(15, 4*len(target_cols)))
        if len(target_cols) == 1:
            axes = axes.reshape(1, -1)
        
        for i, col in enumerate(target_cols):
            data_col = self.data[col].dropna()
            
            if len(data_col) < 50:  # データが少なすぎる場合はスキップ
                continue
            
            # 自己相関
            autocorr = [data_col.autocorr(lag=lag) for lag in range(1, min(100, len(data_col)//2))]
            axes[i,0].plot(autocorr)
            axes[i,0].set_title(f'{col} - 自己相関')
            axes[i,0].set_xlabel('ラグ')
            axes[i,0].set_ylabel('自己相関係数')
            axes[i,0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            
            # 主要な周期の検出
            significant_lags = [lag+1 for lag, corr in enumerate(autocorr) if abs(corr) > 0.3]
            if significant_lags:
                self.insights.append(f"{col}: 主要周期候補 = {significant_lags[:3]}ポイント")
            
            # FFT による周波数分析
            if len(data_col) >= 100:
                # データを正規化
                normalized_data = (data_col - data_col.mean()) / data_col.std()
                
                # FFT計算
                fft_vals = fft(normalized_data.values)
                freqs = fftfreq(len(normalized_data))
                
                # パワースペクトル
                power = np.abs(fft_vals)**2
                
                # 正の周波数のみプロット
                pos_mask = freqs > 0
                axes[i,1].plot(freqs[pos_mask], power[pos_mask])
                axes[i,1].set_title(f'{col} - パワースペクトル')
                axes[i,1].set_xlabel('周波数')
                axes[i,1].set_ylabel('パワー')
                
                # 主要な周波数成分を特定
                top_freq_indices = np.argsort(power[pos_mask])[-3:][::-1]
                top_freqs = freqs[pos_mask][top_freq_indices]
                top_periods = [1/f if f > 0 else np.inf for f in top_freqs]
                
                significant_periods = [p for p in top_periods if 2 <= p <= len(data_col)/4]
                if significant_periods:
                    self.insights.append(f"{col}: FFT主要周期 = {[round(p, 1) for p in significant_periods[:2]]}")
        
        plt.tight_layout()
        plt.show()
    
    def seasonal_decomposition_analysis(self):
        """季節分解分析"""
        print("\n" + "=" * 60)
        print("【季節分解分析】")
        print("=" * 60)
        
        if self.datetime_col is None or len(self.numeric_cols) == 0:
            print("日時列または数値列がないため、季節分解分析をスキップします")
            return
        
        # 時系列データの準備
        ts_data = self.data.set_index(self.datetime_col)
        
        # 代表的な列を選択
        target_cols = self.numeric_cols[:2]
        
        for col in target_cols:
            try:
                # データの準備（欠損値補間）
                series = ts_data[col].fillna(method='ffill').fillna(method='bfill')
                
                # 十分なデータ点があるかチェック
                if len(series) < 20:
                    print(f"{col}: データ点が不足しています")
                    continue
                
                # 適切な周期を推定
                data_span_days = (series.index.max() - series.index.min()).days
                if data_span_days >= 14:
                    period = min(7, len(series) // 4)  # 週周期または適応的
                else:
                    period = max(2, len(series) // 10)
                
                # 季節分解
                if len(series) >= 2 * period:
                    decomp = seasonal_decompose(series, model='additive', period=period, extrapolate_trend='freq')
                    
                    # 可視化
                    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
                    
                    decomp.observed.plot(ax=axes[0], title=f'{col} - 観測値')
                    decomp.trend.plot(ax=axes[1], title='トレンド成分')
                    decomp.seasonal.plot(ax=axes[2], title='季節成分')
                    decomp.resid.plot(ax=axes[3], title='残差成分')
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # 成分の重要性評価
                    trend_var = decomp.trend.var()
                    seasonal_var = decomp.seasonal.var()
                    resid_var = decomp.resid.var()
                    
                    total_var = trend_var + seasonal_var + resid_var
                    
                    if seasonal_var / total_var > 0.1:
                        self.insights.append(f"{col}: 明確な季節性 (寄与率{seasonal_var/total_var:.1%})")
                    
                    if trend_var / total_var > 0.3:
                        self.insights.append(f"{col}: 強いトレンド (寄与率{trend_var/total_var:.1%})")
                        
            except Exception as e:
                print(f"{col}の季節分解でエラー: {e}")
    
    def change_point_detection(self):
        """変化点検出"""
        print("\n" + "=" * 60)
        print("【変化点検出】")
        print("=" * 60)
        
        if len(self.numeric_cols) == 0:
            print("数値列がありません")
            return
        
        target_cols = self.numeric_cols[:3]
        
        for col in target_cols:
            data_col = self.data[col].dropna()
            
            if len(data_col) < 20:
                continue
            
            # 移動平均と移動標準偏差
            window = min(10, len(data_col) // 5)
            rolling_mean = data_col.rolling(window=window).mean()
            rolling_std = data_col.rolling(window=window).std()
            
            # 統計的変化点の検出（簡単な手法）
            mean_changes = rolling_mean.diff().abs()
            std_changes = rolling_std.diff().abs()
            
            # 閾値を超える変化点
            mean_threshold = mean_changes.quantile(0.95)
            std_threshold = std_changes.quantile(0.95)
            
            mean_change_points = mean_changes[mean_changes > mean_threshold].index
            std_change_points = std_changes[std_changes > std_threshold].index
            
            # 可視化
            fig, axes = plt.subplots(3, 1, figsize=(15, 10))
            
            # 元データと移動平均
            axes[0].plot(data_col.index, data_col.values, alpha=0.7, label='原データ')
            axes[0].plot(rolling_mean.index, rolling_mean.values, color='red', linewidth=2, label='移動平均')
            
            # 変化点をマーク
            for cp in mean_change_points:
                axes[0].axvline(x=cp, color='red', linestyle='--', alpha=0.7)
            
            axes[0].set_title(f'{col} - データと移動平均')
            axes[0].legend()
            
            # 平均の変化量
            axes[1].plot(mean_changes.index, mean_changes.values)
            axes[1].axhline(y=mean_threshold, color='red', linestyle='--', label=f'閾値({mean_threshold:.3f})')
            axes[1].set_title('平均値の変化量')
            axes[1].legend()
            
            # 標準偏差の変化量
            axes[2].plot(std_changes.index, std_changes.values)
            axes[2].axhline(y=std_threshold, color='red', linestyle='--', label=f'閾値({std_threshold:.3f})')
            axes[2].set_title('標準偏差の変化量')
            axes[2].legend()
            
            plt.tight_layout()
            plt.show()
            
            # 発見事項の記録
            if len(mean_change_points) > 0:
                self.insights.append(f"{col}: {len(mean_change_points)}個の平均値変化点を検出")
            
            if len(std_change_points) > 0:
                self.insights.append(f"{col}: {len(std_change_points)}個の分散変化点を検出")
    
    def cross_correlation_analysis(self):
        """クロス相関分析"""
        print("\n" + "=" * 60)
        print("【クロス相関分析】")
        print("=" * 60)
        
        if len(self.numeric_cols) < 2:
            print("クロス相関分析には最低2つの数値列が必要です")
            return
        
        # 主要な変数ペアを選択
        target_cols = self.numeric_cols[:4]
        
        cross_corr_results = {}
        
        for i, col1 in enumerate(target_cols):
            for j, col2 in enumerate(target_cols[i+1:], i+1):
                
                # 共通のインデックスでデータを取得
                common_data = self.data[[col1, col2]].dropna()
                
                if len(common_data) < 20:
                    continue
                
                series1 = common_data[col1]
                series2 = common_data[col2]
                
                # クロス相関の計算
                max_lag = min(20, len(series1) // 4)
                lags = range(-max_lag, max_lag + 1)
                cross_corrs = []
                
                for lag in lags:
                    if lag == 0:
                        corr = series1.corr(series2)
                    elif lag > 0:
                        corr = series1[:-lag].corr(series2[lag:])
                    else:
                        corr = series1[-lag:].corr(series2[:lag])
                    cross_corrs.append(corr)
                
                # 最大相関とそのラグを特定
                max_corr_idx = np.nanargmax(np.abs(cross_corrs))
                max_corr = cross_corrs[max_corr_idx]
                max_lag = lags[max_corr_idx]
                
                cross_corr_results[f"{col1}-{col2}"] = {
                    'lags': lags,
                    'correlations': cross_corrs,
                    'max_corr': max_corr,
                    'max_lag': max_lag
                }
                
                # 有意な相関の記録
                if abs(max_corr) > 0.5:
                    if max_lag == 0:
                        self.insights.append(f"{col1}-{col2}: 同時相関 r={max_corr:.3f}")
                    else:
                        self.insights.append(f"{col1}-{col2}: ラグ{max_lag}で最大相関 r={max_corr:.3f}")
        
        # 可視化
        if cross_corr_results:
            n_pairs = len(cross_corr_results)
            n_cols = min(3, n_pairs)
            n_rows = (n_pairs + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
            if n_pairs == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for idx, (pair_name, result) in enumerate(cross_corr_results.items()):
                if idx < len(axes):
                    axes[idx].plot(result['lags'], result['correlations'], 'o-')
                    axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.5)
                    axes[idx].axvline(x=0, color='k', linestyle='--', alpha=0.5)
                    axes[idx].set_title(f'{pair_name}\n最大: r={result["max_corr"]:.3f} (ラグ{result["max_lag"]})')
                    axes[idx].set_xlabel('ラグ')
                    axes[idx].set_ylabel('相関係数')
            
            # 余った軸を非表示
            for idx in range(len(cross_corr_results), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.show()
        
        return cross_corr_results
    
    def process_state_analysis(self):
        """プロセス状態分析"""
        print("\n" + "=" * 60)
        print("【プロセス状態分析】")
        print("=" * 60)
        
        if len(self.numeric_cols) < 2:
            print("プロセス状態分析には複数の数値列が必要です")
            return
        
        # データの標準化
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.data[self.numeric_cols].fillna(0))
        scaled_df = pd.DataFrame(scaled_data, columns=self.numeric_cols, index=self.data.index)
        
        # 主成分分析
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(3, len(self.numeric_cols)))
        pca_result = pca.fit_transform(scaled_data)
        
        # プロセス状態の可視化
        if pca.n_components >= 2:
            fig = plt.figure(figsize=(15, 5))
            
            # PC1 vs PC2
            plt.subplot(1, 3, 1)
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6, s=20)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
            plt.title('プロセス状態空間 (PC1 vs PC2)')
            
            # 時系列での主成分変化
            if pca.n_components >= 1:
                plt.subplot(1, 3, 2)
                plt.plot(pca_result[:, 0], alpha=0.7)
                plt.title('PC1の時系列変化')
                plt.xlabel('時間')
                plt.ylabel('PC1')
                
            if pca.n_components >= 2:
                plt.subplot(1, 3, 3)
                plt.plot(pca_result[:, 1], alpha=0.7)
                plt.title('PC2の時系列変化')
                plt.xlabel('時間')
                plt.ylabel('PC2')
            
            plt.tight_layout()
            plt.show()
        
        # 異常な運転状態の検出
        from sklearn.ensemble import IsolationForest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_scores = iso_forest.fit_predict(scaled_data)
        
        n_anomalies = (anomaly_scores == -1).sum()
        anomaly_rate = n_anomalies / len(self.data) * 100
        
        print(f"異常運転状態: {n_anomalies}件 ({anomaly_rate:.2f}%)")
        
        if anomaly_rate > 2:
            self.insights.append(f"多数の異常運転状態を検出 ({n_anomalies}件, {anomaly_rate:.1f}%)")
        
        # プロセス安定性の評価
        pc1_stability = np.std(pca_result[:, 0])
        if pc1_stability > 2:
            self.insights.append("プロセスの主要成分に高い変動性")
        
        return pca_result, anomaly_scores
    
    def interactive_exploration_guide(self):
        """対話的探索のガイド"""
        print("\n" + "=" * 60)
        print("【対話的探索ガイド】")
        print("=" * 60)
        
        print("以下の関数を個別に実行することで、詳細な分析を行えます：\n")
        
        analysis_functions = {
            "1. 稼働パターン分析": "explorer.operational_pattern_analysis()",
            "2. 周期性検出": "explorer.periodicity_detection()",
            "3. 季節分解分析": "explorer.seasonal_decomposition_analysis()",
            "4. 変化点検出": "explorer.change_point_detection()",
            "5. クロス相関分析": "explorer.cross_correlation_analysis()",
            "6. プロセス状態分析": "explorer.process_state_analysis()",
            "7. カスタム変数分析": "explorer.custom_variable_analysis('列名')",
            "8. 特定期間分析": "explorer.time_period_analysis('開始日', '終了日')"
        }
        
        for desc, func in analysis_functions.items():
            print(f"{desc}:")
            print(f"   {func}")
            print()
        
        print("また、以下の属性で結果を確認できます：")
        print("   explorer.insights  # 発見事項リスト")
        print("   explorer.data      # 元データ")
        print("   explorer.numeric_cols  # 数値列リスト")
    
    def custom_variable_analysis(self, column_name):
        """特定変数のカスタム分析"""
        if column_name not in self.data.columns:
            print(f"列 '{column_name}' が見つかりません")
            return
        
        print(f"\n=== {column_name} の詳細分析 ===")
        
        data_col = self.data[column_name].dropna()
        
        # 基本統計
        print(f"基本統計:")
        print(f"  平均: {data_col.mean():.3f}")
        print(f"  中央値: {data_col.median():.3f}")
        print(f"  標準偏差: {data_col.std():.3f}")
        print(f"  範囲: {data_col.min():.3f} ～ {data_col.max():.3f}")
        
        # 分布の特徴
        skewness = stats.skew(data_col)
        kurtosis = stats.kurtosis(data_col)
        print(f"\n分布特性:")
        print(f"  歪度: {skewness:.3f}")
        print(f"  尖度: {kurtosis:.3f}")
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # ヒストグラム
        axes[0,0].hist(data_col, bins=50, alpha=0.7, density=True)
        axes[0,0].set_title(f'{column_name} - 分布')
        
        # ボックスプロット
        axes[0,1].boxplot(data_col)
        axes[0,1].set_title(f'{column_name} - ボックスプロット')
        
        # Q-Qプロット
        stats.probplot(data_col, dist="norm", plot=axes[1,0])
        axes[1,0].set_title('正規Q-Qプロット')
        
        # 時系列プロット
        axes[1,1].plot(data_col, alpha=0.7)
        axes[1,1].set_title(f'{column_name} - 時系列')
        
        plt.tight_layout()
        plt.show()
        
        return data_col
    
    def time_period_analysis(self, start_date, end_date):
        """特定期間の分析"""
        if self.datetime_col is None:
            print("日時列がないため、期間分析はできません")
            return
        
        # 期間でフィルタリング
        mask = (self.data[self.datetime_col] >= start_date) & (self.data[self.datetime_col] <= end_date)
        period_data = self.data[mask]
        
        if len(period_data) == 0:
            print("指定期間にデータがありません")
            return
        
        print(f"\n=== {start_date} ～ {end_date} の分析 ===")
        print(f"期間データ数: {len(period_data)}件")
        
        # 基本統計の比較
        if len(self.numeric_cols) > 0:
            print("\n期間統計 vs 全体統計:")
            
            period_stats = period_data[self.numeric_cols].mean()
            overall_stats = self.data[self.numeric_cols].mean()
            
            comparison = pd.DataFrame({
                '期間平均': period_stats,
                '全体平均': overall_stats,
                '差分': period_stats - overall_stats,
                '差分率(%)': ((period_stats - overall_stats) / overall_stats * 100)
            })
            
            print(comparison.round(3))
        
        return period_data
    
    def generate_comprehensive_insights(self):
        """包括的な洞察レポートの生成"""
        print("\n" + "=" * 80)
        print("【高度分析 - 包括的洞察レポート】")
        print("=" * 80)
        
        if not self.insights:
            print("特筆すべき洞察はありませんでした")
            return
        
        print("■ 高度分析で得られた洞察:")
        for i, insight in enumerate(self.insights, 1):
            print(f"{i:2d}. {insight}")
        
        print(f"\n■ 洞察総数: {len(self.insights)}件")
        
        # カテゴリ別の集計
        categories = {
            '周期性・季節性': [i for i in self.insights if any(kw in i for kw in ['周期', '季節', 'FFT'])],
            '変化点・異常': [i for i in self.insights if any(kw in i for kw in ['変化点', '異常'])],
            '相関・関連性': [i for i in self.insights if any(kw in i for kw in ['相関', '関連'])],
            '稼働パターン': [i for i in self.insights if any(kw in i for kw in ['稼働', 'ピーク', '時間'])],
            'プロセス特性': [i for i in self.insights if any(kw in i for kw in ['プロセス', '成分', '変動'])]
        }
        
        print("\n■ カテゴリ別洞察:")
        for category, items in categories.items():
            if items:
                print(f"\n{category} ({len(items)}件):")
                for item in items:
                    print(f"  • {item}")
        
        return self.insights
    
    def run_advanced_analysis(self):
        """高度分析の一括実行"""
        print("🔬 高度なデータ洞察分析を開始します...")
        
        self.operational_pattern_analysis()
        self.periodicity_detection()
        self.seasonal_decomposition_analysis()
        self.change_point_detection()
        self.cross_correlation_analysis()
        self.process_state_analysis()
        
        # 総合レポート
        self.generate_comprehensive_insights()
        
        print("\n✅ 高度分析が完了しました")
        print("\n💡 さらなる探索のために interactive_exploration_guide() を実行してください")
        
        return self.insights

# 使用例とクイックスタート
def quick_advanced_analysis(data_path, datetime_col=None):
    """クイックスタート用の関数"""
    try:
        # データ読み込み
        data = pd.read_csv(data_path)
        
        # 日時列の自動変換
        if datetime_col and datetime_col in data.columns:
            data[datetime_col] = pd.to_datetime(data[datetime_col])
        
        # 高度分析の実行
        analyzer = AdvancedFactoryDataInsights(data, datetime_col)
        insights = analyzer.run_advanced_analysis()
        
        return analyzer, insights
        
    except Exception as e:
        print(f"エラー: {e}")
        return None, None

if __name__ == "__main__":
    print("高度データ洞察分析ツールの使用方法:")
    print("1. データを読み込み: data = pd.read_csv('your_data.csv')")
    print("2. 分析器を初期化: analyzer = AdvancedFactoryDataInsights(data)")
    print("3. 一括分析実行: analyzer.run_advanced_analysis()")
    print("4. または個別分析: analyzer.operational_pattern_analysis() など")
    print("\nクイックスタート:")
    print("analyzer, insights = quick_advanced_analysis('your_data.csv', 'datetime_column')") 