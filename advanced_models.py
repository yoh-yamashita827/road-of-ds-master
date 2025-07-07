# 高度なモデル実装サンプル
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =====================================
# 1. LSTM用データ準備
# =====================================

def create_lstm_dataset(data, target_col, sequence_length=60, forecast_horizon=1):
    """LSTM用のシーケンスデータ作成"""
    
    # スケーリング
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    features = data.drop(columns=[target_col])
    target = data[target_col]
    
    scaled_features = scaler_X.fit_transform(features)
    scaled_target = scaler_y.fit_transform(target.values.reshape(-1, 1)).flatten()
    
    X, y = [], []
    
    for i in range(sequence_length, len(scaled_features) - forecast_horizon + 1):
        X.append(scaled_features[i-sequence_length:i])
        y.append(scaled_target[i + forecast_horizon - 1])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"LSTM Dataset Shape: X={X.shape}, y={y.shape}")
    
    return X, y, scaler_X, scaler_y

# =====================================
# 2. LSTMモデル構築
# =====================================

def build_lstm_model(input_shape, units=[64, 32], dropout_rate=0.2):
    """LSTMモデル構築"""
    
    model = Sequential()
    
    # 第1LSTM層
    model.add(LSTM(
        units[0],
        return_sequences=True if len(units) > 1 else False,
        input_shape=input_shape
    ))
    model.add(Dropout(dropout_rate))
    
    # 追加のLSTM層
    for i, unit in enumerate(units[1:], 1):
        return_seq = i < len(units) - 1
        model.add(LSTM(unit, return_sequences=return_seq))
        model.add(Dropout(dropout_rate))
    
    # 出力層
    model.add(Dense(1))
    
    # コンパイル
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    print("LSTM Model Summary:")
    model.summary()
    
    return model

def train_lstm_model(X, y, sequence_length=60, epochs=50, batch_size=32):
    """LSTMモデル訓練"""
    
    # 時系列分割
    tscv = TimeSeriesSplit(n_splits=3)
    
    histories = []
    models = []
    scores = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n=== Fold {fold + 1} ===")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # モデル構築
        model = build_lstm_model(
            input_shape=(X.shape[1], X.shape[2]),
            units=[64, 32],
            dropout_rate=0.2
        )
        
        # 訓練
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # 評価
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        histories.append(history)
        models.append(model)
        scores.append({'fold': fold + 1, 'rmse': rmse, 'mae': mae})
    
    return models, histories, scores

# =====================================
# 3. XGBoostモデル
# =====================================

def train_xgboost_model(X, y, params=None):
    """XGBoostモデル訓練"""
    
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
    
    # 時系列分割
    tscv = TimeSeriesSplit(n_splits=5)
    
    models = []
    scores = []
    predictions = []
    actuals = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        print(f"\n=== XGBoost Fold {fold + 1} ===")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # モデル訓練
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # 予測
        y_pred = model.predict(X_test)
        
        # 評価
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"Fold {fold + 1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
        
        models.append(model)
        scores.append({'fold': fold + 1, 'rmse': rmse, 'mae': mae})
        predictions.extend(y_pred)
        actuals.extend(y_test)
    
    # 全体評価
    overall_rmse = np.sqrt(mean_squared_error(actuals, predictions))
    overall_mae = mean_absolute_error(actuals, predictions)
    
    print(f"\n=== XGBoost 全体評価 ===")
    print(f"RMSE: {overall_rmse:.4f}")
    print(f"MAE: {overall_mae:.4f}")
    
    return models, scores

# =====================================
# 4. 特徴量重要度可視化
# =====================================

def plot_feature_importance(model, feature_names, top_k=20):
    """特徴量重要度の可視化"""
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        print("このモデルは特徴量重要度をサポートしていません")
        return
    
    # 重要度データフレーム作成
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 可視化
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_k)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('重要度')
    plt.title(f'特徴量重要度 (Top {top_k})')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df

# =====================================
# 5. アンサンブルモデル
# =====================================

def ensemble_prediction(models, X_data, weights=None):
    """複数モデルのアンサンブル予測"""
    
    if weights is None:
        weights = [1/len(models)] * len(models)
    
    predictions = []
    
    for model in models:
        if hasattr(model, 'predict'):
            pred = model.predict(X_data)
            predictions.append(pred)
    
    # 重み付き平均
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    
    return ensemble_pred

# =====================================
# 6. ハイパーパラメータ最適化
# =====================================

def optimize_xgboost_params(X, y, n_trials=100):
    """Optunaを使ったXGBoostハイパーパラメータ最適化"""
    
    import optuna
    
    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42
        }
        
        # 時系列分割での交差検証
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=False)
            
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            scores.append(rmse)
        
        return np.mean(scores)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)
    
    print("=== 最適化結果 ===")
    print("最適パラメータ:")
    for key, value in study.best_params.items():
        print(f"{key}: {value}")
    print(f"最適スコア: {study.best_value:.4f}")
    
    return study.best_params

# =====================================
# 7. 実時間予測システム
# =====================================

class RealTimePredictionSystem:
    """リアルタイム予測システム"""
    
    def __init__(self, model, scaler, selected_features, sequence_length=60):
        self.model = model
        self.scaler = scaler
        self.selected_features = selected_features
        self.sequence_length = sequence_length
        self.data_buffer = []
    
    def add_new_data(self, new_row):
        """新しいデータを追加"""
        self.data_buffer.append(new_row)
        
        # バッファサイズを制限
        if len(self.data_buffer) > self.sequence_length:
            self.data_buffer.pop(0)
    
    def predict(self):
        """現在のバッファデータで予測"""
        if len(self.data_buffer) < self.sequence_length:
            return None, "データが不足しています"
        
        # データ準備
        data_df = pd.DataFrame(self.data_buffer)
        features = data_df[self.selected_features]
        scaled_features = self.scaler.transform(features)
        
        # LSTMの場合
        if hasattr(self.model, 'predict') and len(scaled_features.shape) == 2:
            X = scaled_features.reshape(1, scaled_features.shape[0], scaled_features.shape[1])
            prediction = self.model.predict(X)[0][0]
        else:
            # 他のモデルの場合（最新データのみ使用）
            prediction = self.model.predict(scaled_features[-1:].reshape(1, -1))[0]
        
        return prediction, "予測成功"
    
    def get_prediction_confidence(self, n_bootstrap=100):
        """ブートストラップによる予測信頼区間"""
        if len(self.data_buffer) < self.sequence_length:
            return None, None, "データが不足しています"
        
        predictions = []
        
        for _ in range(n_bootstrap):
            # ブートストラップサンプリング
            bootstrap_indices = np.random.choice(
                len(self.data_buffer), 
                size=len(self.data_buffer), 
                replace=True
            )
            bootstrap_data = [self.data_buffer[i] for i in bootstrap_indices]
            
            # 予測
            data_df = pd.DataFrame(bootstrap_data)
            features = data_df[self.selected_features]
            scaled_features = self.scaler.transform(features)
            
            if hasattr(self.model, 'predict') and len(scaled_features.shape) == 2:
                X = scaled_features.reshape(1, scaled_features.shape[0], scaled_features.shape[1])
                pred = self.model.predict(X)[0][0]
            else:
                pred = self.model.predict(scaled_features[-1:].reshape(1, -1))[0]
            
            predictions.append(pred)
        
        # 信頼区間計算
        confidence_interval = np.percentile(predictions, [2.5, 97.5])
        mean_prediction = np.mean(predictions)
        
        return mean_prediction, confidence_interval, "成功"

# =====================================
# 使用例
# =====================================

def advanced_analysis_example():
    """高度な分析の使用例"""
    
    # データ読み込み（仮想データ）
    # df = pd.read_csv("factory_data.csv")
    # df = preprocess_data(df)
    # df_features = create_features(df)
    
    print("=== 高度なモデル分析例 ===")
    
    # 1. XGBoostモデル
    print("\n1. XGBoostモデル訓練")
    # models_xgb, scores_xgb = train_xgboost_model(X, y)
    
    # 2. LSTMモデル
    print("\n2. LSTMモデル訓練")
    # X_lstm, y_lstm, scaler_X, scaler_y = create_lstm_dataset(df_features, 'water_meter')
    # models_lstm, histories, scores_lstm = train_lstm_model(X_lstm, y_lstm)
    
    # 3. ハイパーパラメータ最適化
    print("\n3. ハイパーパラメータ最適化")
    # best_params = optimize_xgboost_params(X, y, n_trials=50)
    
    # 4. リアルタイム予測システム
    print("\n4. リアルタイム予測システム")
    # prediction_system = RealTimePredictionSystem(
    #     model=models_xgb[0],
    #     scaler=scaler,
    #     selected_features=selected_features
    # )
    
    print("分析コードの準備完了！")

if __name__ == "__main__":
    advanced_analysis_example() 