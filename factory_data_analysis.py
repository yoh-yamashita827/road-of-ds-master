# 工場タグデータ分析サンプルコード
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

# =====================================
# 1. データ読み込み・基本情報確認
# =====================================

def load_and_explore_data(file_path):
    """データ読み込みと基本情報の確認"""
    # CSVファイル読み込み
    df = pd.read_csv(file_path)
    
    # タイムスタンプをdatetime型に変換
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    print("=== データ基本情報 ===")
    print(f"データ期間: {df.index.min()} ～ {df.index.max()}")
    print(f"データ数: {len(df)}")
    print(f"カラム数: {len(df.columns)}")
    print("\n=== 欠損値情報 ===")
    print(df.isnull().sum())
    print("\n=== 基本統計量 ===")
    print(df.describe())
    
    return df

# =====================================
# 2. データ可視化・探索
# =====================================

def visualize_data(df, target_col='water_meter'):
    """データの可視化"""
    plt.figure(figsize=(15, 10))
    
    # 水分計の時系列プロット
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df[target_col])
    plt.title('水分計の時系列変化')
    plt.xlabel('時間')
    plt.ylabel('水分計値')
    
    # 水分計の分布
    plt.subplot(2, 2, 2)
    plt.hist(df[target_col].dropna(), bins=50, alpha=0.7)
    plt.title('水分計値の分布')
    plt.xlabel('水分計値')
    plt.ylabel('頻度')
    
    # 相関マトリックス（主要タグのみ）
    plt.subplot(2, 2, 3)
    corr_matrix = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', center=0)
    plt.title('相関マトリックス（主要タグ）')
    
    # 水分計と他タグの相関
    plt.subplot(2, 2, 4)
    correlations = df.corrwith(df[target_col]).abs().sort_values(ascending=False)
    correlations[1:11].plot(kind='bar')
    plt.title('水分計との相関（上位10タグ）')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

# =====================================
# 3. データ前処理
# =====================================

def preprocess_data(df, target_col='water_meter'):
    """データ前処理"""
    df_processed = df.copy()
    
    # 1. 欠損値補完（線形補間）
    df_processed = df_processed.interpolate(method='linear')
    
    # 2. 異常値検出・処理（IQR法）
    def remove_outliers_iqr(series, factor=1.5):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        return series.clip(lower_bound, upper_bound)
    
    # 数値カラムの異常値処理
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_processed[col] = remove_outliers_iqr(df_processed[col])
    
    print(f"前処理後のデータ数: {len(df_processed)}")
    print(f"欠損値数: {df_processed.isnull().sum().sum()}")
    
    return df_processed

# =====================================
# 4. 特徴量エンジニアリング
# =====================================

def create_features(df, target_col='water_meter', lag_minutes=30):
    """特徴量作成"""
    df_features = df.copy()
    
    # 1. 時間遅れ特徴量（30分前のデータ）
    lag_steps = lag_minutes  # 1分単位なので30ステップ
    for col in df.columns:
        if col != target_col:
            df_features[f'{col}_lag{lag_minutes}m'] = df[col].shift(lag_steps)
    
    # 2. 移動平均特徴量
    windows = [5, 15, 30, 60]  # 5分、15分、30分、1時間
    for window in windows:
        for col in df.columns:
            if col != target_col:
                df_features[f'{col}_ma{window}m'] = df[col].rolling(window=window, min_periods=1).mean()
    
    # 3. 移動標準偏差
    for window in [15, 30]:
        for col in df.columns:
            if col != target_col:
                df_features[f'{col}_std{window}m'] = df[col].rolling(window=window, min_periods=1).std()
    
    # 4. 差分特徴量
    for col in df.columns:
        if col != target_col:
            df_features[f'{col}_diff1'] = df[col].diff(1)  # 1分前との差
            df_features[f'{col}_diff5'] = df[col].diff(5)  # 5分前との差
    
    # 5. PV-SV差（設定値との乖離）
    pv_cols = [col for col in df.columns if 'PV' in col]
    sv_cols = [col for col in df.columns if 'SV' in col]
    
    for pv_col in pv_cols:
        tag_name = pv_col.replace('_PV', '')
        sv_col = f'{tag_name}_SV'
        if sv_col in df.columns:
            df_features[f'{tag_name}_PV_SV_diff'] = df[pv_col] - df[sv_col]
    
    # 6. 時間特徴量
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['is_weekend'] = (df_features.index.dayofweek >= 5).astype(int)
    
    # 欠損値を削除
    df_features = df_features.dropna()
    
    print(f"特徴量作成後のカラム数: {len(df_features.columns)}")
    print(f"データ数: {len(df_features)}")
    
    return df_features

# =====================================
# 5. 特徴選択
# =====================================

def feature_selection(X, y, method='correlation', top_k=50):
    """特徴選択"""
    if method == 'correlation':
        # 相関係数による選択
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected_features = correlations.head(top_k).index.tolist()
        
    elif method == 'mutual_info':
        # 相互情報量による選択
        mi_scores = mutual_info_regression(X, y)
        mi_df = pd.DataFrame({'feature': X.columns, 'score': mi_scores})
        mi_df = mi_df.sort_values('score', ascending=False)
        selected_features = mi_df.head(top_k)['feature'].tolist()
        
    elif method == 'rf_importance':
        # Random Forestの特徴量重要度
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        selected_features = importance_df.head(top_k)['feature'].tolist()
    
    print(f"選択された特徴量数: {len(selected_features)}")
    print("上位10特徴量:")
    for i, feature in enumerate(selected_features[:10], 1):
        print(f"{i}. {feature}")
    
    return selected_features

# =====================================
# 6. モデル構築・評価
# =====================================

def train_and_evaluate_model(X, y, model_type='rf'):
    """モデル訓練と評価"""
    # 時系列分割
    tscv = TimeSeriesSplit(n_splits=5)
    
    # データスケーリング
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # モデル選択
    if model_type == 'rf':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    
    # 交差検証
    cv_scores = []
    predictions = []
    actuals = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # モデル訓練
        model.fit(X_train, y_train)
        
        # 予測
        y_pred = model.predict(X_test)
        
        # 評価
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        cv_scores.append({'fold': fold+1, 'RMSE': rmse, 'MAE': mae})
        predictions.extend(y_pred)
        actuals.extend(y_test)
        
        print(f"Fold {fold+1}: RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    # 全体の評価
    overall_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    overall_mae = mean_absolute_error(actuals, predictions)
    
    print(f"\n=== 全体評価 ===")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE: {overall_mae:.4f}")
    
    return model, scaler, cv_scores

# =====================================
# 7. 予測結果可視化
# =====================================

def visualize_predictions(y_true, y_pred, timestamps):
    """予測結果の可視化"""
    plt.figure(figsize=(15, 8))
    
    # 時系列プロット
    plt.subplot(2, 1, 1)
    plt.plot(timestamps, y_true, label='実測値', alpha=0.7)
    plt.plot(timestamps, y_pred, label='予測値', alpha=0.7)
    plt.title('水分計予測結果')
    plt.xlabel('時間')
    plt.ylabel('水分計値')
    plt.legend()
    
    # 散布図
    plt.subplot(2, 1, 2)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('実測値')
    plt.ylabel('予測値')
    plt.title('実測値 vs 予測値')
    
    plt.tight_layout()
    plt.show()

# =====================================
# 8. メイン実行関数
# =====================================

def main_analysis_pipeline(file_path, target_col='water_meter'):
    """メイン分析パイプライン"""
    print("=== 工場データ分析開始 ===\n")
    
    # 1. データ読み込み
    print("1. データ読み込み・探索")
    df = load_and_explore_data(file_path)
    
    # 2. データ可視化
    print("\n2. データ可視化")
    visualize_data(df, target_col)
    
    # 3. データ前処理
    print("\n3. データ前処理")
    df_processed = preprocess_data(df, target_col)
    
    # 4. 特徴量作成
    print("\n4. 特徴量エンジニアリング")
    df_features = create_features(df_processed, target_col)
    
    # 5. 特徴選択
    print("\n5. 特徴選択")
    X = df_features.drop(columns=[target_col])
    y = df_features[target_col]
    
    selected_features = feature_selection(X, y, method='rf_importance', top_k=30)
    X_selected = X[selected_features]
    
    # 6. モデル構築・評価
    print("\n6. モデル構築・評価")
    model, scaler, cv_scores = train_and_evaluate_model(X_selected, y)
    
    print("\n=== 分析完了 ===")
    
    return model, scaler, selected_features, df_features

# =====================================
# 使用例
# =====================================

if __name__ == "__main__":
    # 使用例
    file_path = "factory_data.csv"  # あなたのCSVファイルパス
    target_column = "water_meter"   # 水分計のカラム名
    
    # 分析実行
    model, scaler, features, data = main_analysis_pipeline(file_path, target_column)
    
    # 新しいデータでの予測例
    def predict_water_meter(new_data, model, scaler, selected_features):
        """新しいデータに対する予測"""
        new_data_scaled = scaler.transform(new_data[selected_features])
        prediction = model.predict(new_data_scaled)
        return prediction 