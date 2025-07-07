# 工場データ分析プロジェクト

## 概要
工場の各種タグ（PV, SV, MV値）から水分計の予測を行うAIモデル構築プロジェクトです。

## プロジェクト構成
```
├── factory_data_analysis.py    # 基本的な分析パイプライン
├── advanced_models.py          # 高度なモデル（LSTM、XGBoost等）
├── requirements.txt            # 必要ライブラリ
└── README.md                  # このファイル
```

## セットアップ

### 1. 環境構築
```bash
pip install -r requirements.txt
```

### 2. データ準備
CSVファイルは以下の形式を想定しています：
- タイムスタンプ列（`timestamp`）
- 各種タグのPV, SV, MV値
- 目的変数となる水分計の値

## 使用方法

### 基本分析
```python
from factory_data_analysis import main_analysis_pipeline

# 分析実行
model, scaler, features, data = main_analysis_pipeline(
    file_path="your_data.csv",
    target_col="water_meter"
)
```

### 高度なモデル
```python
from advanced_models import train_xgboost_model, train_lstm_model

# XGBoostモデル
models_xgb, scores = train_xgboost_model(X, y)

# LSTMモデル
X_lstm, y_lstm, scaler_X, scaler_y = create_lstm_dataset(data, 'water_meter')
models_lstm, histories, scores = train_lstm_model(X_lstm, y_lstm)
```

## 分析の流れ

### 1. データ探索
- 基本統計量の確認
- 欠損値・異常値の検出
- 相関分析
- 時系列可視化

### 2. 前処理
- 欠損値補完（線形補間）
- 異常値処理（IQR法）
- データ正規化

### 3. 特徴量エンジニアリング
- **時間遅れ特徴量**: 30分前のデータ（前工程の遅れ時間を考慮）
- **移動平均**: 5分、15分、30分、1時間の移動平均
- **移動標準偏差**: 15分、30分の変動性指標
- **差分特徴量**: 1分前、5分前との差分
- **PV-SV差**: 設定値との乖離
- **時間特徴量**: 時刻、曜日、週末フラグ

### 4. 特徴選択
- 相関係数による選択
- 相互情報量による選択
- Random Forestの特徴量重要度

### 5. モデル構築
- **Random Forest**: ベースラインモデル
- **XGBoost**: 勾配ブースティング
- **LSTM**: 時系列深層学習
- **アンサンブル**: 複数モデルの組み合わせ

### 6. 評価
- 時系列分割による交差検証
- RMSE、MAEによる評価
- 予測結果の可視化

## 高度な機能

### ハイパーパラメータ最適化
```python
from advanced_models import optimize_xgboost_params

best_params = optimize_xgboost_params(X, y, n_trials=100)
```

### リアルタイム予測システム
```python
from advanced_models import RealTimePredictionSystem

prediction_system = RealTimePredictionSystem(
    model=trained_model,
    scaler=scaler,
    selected_features=features
)

# 新しいデータを追加
prediction_system.add_new_data(new_data_row)

# 予測実行
prediction, status = prediction_system.predict()
```

## 注意事項

1. **データ形式**: CSVファイルのカラム名は実際のデータに合わせて調整してください
2. **時間遅れ**: 30分の遅れ時間は工程によって調整が必要な場合があります
3. **特徴量**: 工場の特性に応じて特徴量を追加・修正してください
4. **計算資源**: LSTMモデルは計算時間が長くなる場合があります

## カスタマイズ例

### 遅れ時間の調整
```python
df_features = create_features(df, target_col='water_meter', lag_minutes=45)
```

### ウィンドウサイズの変更
```python
# create_features関数内のwindowsパラメータを変更
windows = [3, 10, 20, 40]  # カスタマイズした移動平均ウィンドウ
```

### カスタム特徴量の追加
```python
# 温度差特徴量の例
df_features['temp_diff'] = df['temp_inlet_PV'] - df['temp_outlet_PV']

# 比率特徴量の例
df_features['flow_ratio'] = df['flow_rate_PV'] / df['flow_rate_SV']
```

## トラブルシューティング

### よくある問題
1. **メモリ不足**: データサイズが大きい場合はチャンクサイズを調整
2. **収束しない**: 学習率やバッチサイズを調整
3. **過学習**: 正則化パラメータを調整

### パフォーマンス改善
- 特徴量数を削減
- 時系列分割数を調整
- 並列処理の活用

## 今後の拡張案

1. **Prophet**や**ARIMA**等の時系列専用モデルの追加
2. **Attention機構**を持つTransformerモデルの実装
3. **AutoML**による自動モデル選択
4. **ドリフト検出**による継続的学習
5. **説明可能AI**による予測根拠の可視化 