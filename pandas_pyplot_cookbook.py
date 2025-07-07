#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pandas & Pyplot 実用的使い方講座
=====================================

工場データ分析でよく使うpandas・matplotlib操作を
ユースケース別に整理したコード集です。

目次:
1. データ読み込み・基本操作
2. データ結合・マージ
3. データ変換・前処理
4. 時系列データ操作
5. グループ集計・統計
6. 可視化（matplotlib/pyplot）
7. データ出力・保存
8. トラブルシューティング

各セクションは独立して実行可能です。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import japanize_matplotlib

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

# =============================================================================
# 1. データ読み込み・基本操作
# =============================================================================

def section_01_data_loading():
    """データ読み込み・基本操作のコード例"""
    print("=" * 60)
    print("1. データ読み込み・基本操作")
    print("=" * 60)
    
    # ケース1: CSVファイルの読み込み
    print("■ ケース1: CSVファイルの基本読み込み")
    print("# 基本的な読み込み")
    print("df = pd.read_csv('data.csv')")
    print("# エンコーディング指定")
    print("df = pd.read_csv('data.csv', encoding='utf-8')")
    print("# 特定列のみ読み込み")
    print("df = pd.read_csv('data.csv', usecols=['timestamp', 'temp', 'pressure'])")
    print("# 日時列を自動で変換")
    print("df = pd.read_csv('data.csv', parse_dates=['timestamp'])")
    
    # ケース2: データの基本情報確認
    print("\n■ ケース2: データの基本情報確認")
    # サンプルデータ作成
    sample_data = {
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='5T'),
        'temperature': np.random.normal(85, 2, 100),
        'pressure': np.random.normal(2.0, 0.1, 100),
        'status': np.random.choice(['運転中', '停止中', 'メンテ'], 100)
    }
    df = pd.DataFrame(sample_data)
    
    print("# データの形状確認")
    print(f"df.shape: {df.shape}")
    print("# 基本統計量")
    print("df.describe()")
    print(df.describe().round(3))
    print("# データ型確認")
    print("df.dtypes")
    print(df.dtypes)
    print("# 欠損値確認")
    print("df.isnull().sum()")
    print(df.isnull().sum())
    
    # ケース3: 列の選択・操作
    print("\n■ ケース3: 列の選択・操作")
    print("# 単一列選択")
    print("temperature = df['temperature']")
    print("# 複数列選択")
    print("numeric_cols = df[['temperature', 'pressure']]")
    print("# 列名変更")
    print("df.rename(columns={'temperature': 'temp', 'pressure': 'press'}, inplace=True)")
    
    # ケース4: 行の選択・フィルタリング
    print("\n■ ケース4: 行の選択・フィルタリング")
    print("# 条件でフィルタリング")
    print("high_temp = df[df['temperature'] > 85]")
    high_temp = df[df['temperature'] > 85]
    print(f"高温データ: {len(high_temp)}件")
    
    print("# 複数条件")
    print("filtered = df[(df['temperature'] > 85) & (df['pressure'] < 2.0)]")
    filtered = df[(df['temperature'] > 85) & (df['pressure'] < 2.0)]
    print(f"条件該当データ: {len(filtered)}件")
    
    print("# 特定値を含む行")
    print("maintenance = df[df['status'] == 'メンテ']")
    maintenance = df[df['status'] == 'メンテ']
    print(f"メンテナンス中: {len(maintenance)}件")
    
    return df

# =============================================================================
# 2. データ結合・マージ
# =============================================================================

def section_02_data_merging():
    """データ結合・マージのコード例"""
    print("\n" + "=" * 60)
    print("2. データ結合・マージ")
    print("=" * 60)
    
    # サンプルデータ作成
    df1 = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=50, freq='10T'),
        'temperature': np.random.normal(85, 2, 50),
        'line_id': 'A'
    })
    
    df2 = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01 00:05:00', periods=50, freq='10T'),
        'pressure': np.random.normal(2.0, 0.1, 50),
        'line_id': 'A'
    })
    
    # ケース1: 時間軸での結合
    print("■ ケース1: 時間軸でのデータ結合")
    print("# 内部結合（完全一致のみ）")
    print("merged_inner = pd.merge(df1, df2, on='timestamp', how='inner')")
    merged_inner = pd.merge(df1, df2, on='timestamp', how='inner')
    print(f"内部結合結果: {len(merged_inner)}行")
    
    print("# 外部結合（全データ保持）")
    print("merged_outer = pd.merge(df1, df2, on='timestamp', how='outer')")
    merged_outer = pd.merge(df1, df2, on='timestamp', how='outer')
    print(f"外部結合結果: {len(merged_outer)}行")
    
    print("# 左結合（df1ベース）")
    print("merged_left = pd.merge(df1, df2, on='timestamp', how='left')")
    merged_left = pd.merge(df1, df2, on='timestamp', how='left')
    print(f"左結合結果: {len(merged_left)}行")
    
    # ケース2: 近似時間での結合
    print("\n■ ケース2: 時間が完全一致しない場合の結合")
    print("# merge_asofを使用（最も近い過去の値）")
    print("merged_asof = pd.merge_asof(df1.sort_values('timestamp'), df2.sort_values('timestamp'), on='timestamp')")
    merged_asof = pd.merge_asof(df1.sort_values('timestamp'), df2.sort_values('timestamp'), on='timestamp')
    print(f"merge_asof結果: {len(merged_asof)}行")
    
    # ケース3: インデックスでの結合
    print("\n■ ケース3: インデックスでの結合")
    df1_indexed = df1.set_index('timestamp')
    df2_indexed = df2.set_index('timestamp')
    
    print("# join（インデックスベース）")
    print("joined = df1_indexed.join(df2_indexed, how='outer')")
    joined = df1_indexed.join(df2_indexed, how='outer')
    print(f"join結果: {len(joined)}行")
    
    # ケース4: 縦方向の結合
    print("\n■ ケース4: 縦方向の結合（concat）")
    df3 = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-02', periods=30, freq='10T'),
        'temperature': np.random.normal(87, 2, 30),
        'line_id': 'B'
    })
    
    print("# 縦方向結合")
    print("concatenated = pd.concat([df1, df3], ignore_index=True)")
    concatenated = pd.concat([df1, df3], ignore_index=True)
    print(f"concat結果: {len(concatenated)}行")
    
    print("# キーを追加して結合")
    print("concatenated_with_key = pd.concat([df1, df3], keys=['LineA', 'LineB'])")
    concatenated_with_key = pd.concat([df1, df3], keys=['LineA', 'LineB'])
    print(f"キー付きconcat結果: {len(concatenated_with_key)}行")
    
    return merged_left

# =============================================================================
# 3. データ変換・前処理
# =============================================================================

def section_03_data_transformation():
    """データ変換・前処理のコード例"""
    print("\n" + "=" * 60)
    print("3. データ変換・前処理")
    print("=" * 60)
    
    # サンプルデータ作成
    np.random.seed(42)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=200, freq='5T'),
        'temperature': np.random.normal(85, 3, 200),
        'pressure': np.random.normal(2.0, 0.2, 200),
        'flow_rate': np.random.normal(100, 15, 200)
    })
    
    # いくつかの欠損値を作成
    df.loc[10:15, 'temperature'] = np.nan
    df.loc[50:52, 'pressure'] = np.nan
    
    # ケース1: 欠損値処理
    print("■ ケース1: 欠損値の処理")
    print("# 欠損値を前の値で埋める")
    print("df['temperature'].fillna(method='ffill', inplace=True)")
    df_filled = df.copy()
    df_filled['temperature'].fillna(method='ffill', inplace=True)
    print(f"前方埋め後の欠損値: {df_filled['temperature'].isnull().sum()}個")
    
    print("# 欠損値を平均値で埋める")
    print("df['pressure'].fillna(df['pressure'].mean(), inplace=True)")
    df_filled['pressure'].fillna(df_filled['pressure'].mean(), inplace=True)
    print(f"平均値埋め後の欠損値: {df_filled['pressure'].isnull().sum()}個")
    
    print("# 線形補間")
    print("df['temperature'].interpolate(method='linear', inplace=True)")
    
    # ケース2: 外れ値処理
    print("\n■ ケース2: 外れ値の処理")
    print("# IQR法での外れ値検出")
    print("""
Q1 = df['temperature'].quantile(0.25)
Q3 = df['temperature'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = (df['temperature'] < lower_bound) | (df['temperature'] > upper_bound)
""")
    
    Q1 = df_filled['temperature'].quantile(0.25)
    Q3 = df_filled['temperature'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (df_filled['temperature'] < lower_bound) | (df_filled['temperature'] > upper_bound)
    print(f"外れ値検出: {outliers.sum()}個")
    
    # ケース3: 新しい列の作成
    print("\n■ ケース3: 新しい列の作成")
    print("# 移動平均")
    print("df['temp_ma_5'] = df['temperature'].rolling(window=5).mean()")
    df_filled['temp_ma_5'] = df_filled['temperature'].rolling(window=5).mean()
    
    print("# 差分計算")
    print("df['temp_diff'] = df['temperature'].diff()")
    df_filled['temp_diff'] = df_filled['temperature'].diff()
    
    print("# 条件に基づく列作成")
    print("df['temp_status'] = df['temperature'].apply(lambda x: '高' if x > 87 else '低' if x < 83 else '正常')")
    df_filled['temp_status'] = df_filled['temperature'].apply(lambda x: '高' if x > 87 else '低' if x < 83 else '正常')
    
    print("# 複数列を使った計算")
    print("df['efficiency'] = df['flow_rate'] / df['pressure']")
    df_filled['efficiency'] = df_filled['flow_rate'] / df_filled['pressure']
    
    # ケース4: データ型変換
    print("\n■ ケース4: データ型の変換")
    print("# 文字列をカテゴリ型に")
    print("df['temp_status'] = df['temp_status'].astype('category')")
    df_filled['temp_status'] = df_filled['temp_status'].astype('category')
    
    print("# 数値の精度変更")
    print("df['temperature'] = df['temperature'].round(2)")
    df_filled['temperature'] = df_filled['temperature'].round(2)
    
    return df_filled

# =============================================================================
# 4. 時系列データ操作
# =============================================================================

def section_04_time_series():
    """時系列データ操作のコード例"""
    print("\n" + "=" * 60)
    print("4. 時系列データ操作")
    print("=" * 60)
    
    # サンプルデータ作成
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    df = pd.DataFrame({
        'timestamp': dates,
        'value': np.random.normal(100, 10, 1000) + 10 * np.sin(np.arange(1000) * 2 * np.pi / 288)  # 日周期
    })
    
    # ケース1: 時間インデックスの設定
    print("■ ケース1: 時間インデックスの設定")
    print("df.set_index('timestamp', inplace=True)")
    df.set_index('timestamp', inplace=True)
    print("# インデックスがDatetimeIndexになりました")
    
    # ケース2: 時間範囲での抽出
    print("\n■ ケース2: 時間範囲での抽出")
    print("# 特定日のデータ")
    print("today_data = df['2024-01-01']")
    today_data = df['2024-01-01']
    print(f"1月1日のデータ: {len(today_data)}件")
    
    print("# 期間指定")
    print("week_data = df['2024-01-01':'2024-01-07']")
    week_data = df['2024-01-01':'2024-01-07']
    print(f"1週間のデータ: {len(week_data)}件")
    
    print("# 時間帯指定")
    print("morning_data = df.between_time('08:00', '12:00')")
    morning_data = df.between_time('08:00', '12:00')
    print(f"午前中のデータ: {len(morning_data)}件")
    
    # ケース3: リサンプリング
    print("\n■ ケース3: リサンプリング（時間間隔の変更）")
    print("# 1時間平均")
    print("hourly_avg = df.resample('1H').mean()")
    hourly_avg = df.resample('1H').mean()
    print(f"1時間平均データ: {len(hourly_avg)}件")
    
    print("# 日次統計")
    print("daily_stats = df.resample('1D').agg({'value': ['mean', 'min', 'max', 'std']})")
    daily_stats = df.resample('1D').agg({'value': ['mean', 'min', 'max', 'std']})
    print(f"日次統計: {len(daily_stats)}件")
    
    # ケース4: 時間特徴量の作成
    print("\n■ ケース4: 時間特徴量の作成")
    df_time = df.copy()
    print("# 時間要素の抽出")
    print("""
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['month'] = df.index.month
df['is_weekend'] = df.index.dayofweek >= 5
""")
    df_time['hour'] = df_time.index.hour
    df_time['day_of_week'] = df_time.index.dayofweek
    df_time['month'] = df_time.index.month
    df_time['is_weekend'] = df_time.index.dayofweek >= 5
    
    # ケース5: 時間シフト
    print("\n■ ケース5: 時間シフト（ラグ・リード）")
    print("# 1期間前の値")
    print("df['value_lag1'] = df['value'].shift(1)")
    df_time['value_lag1'] = df_time['value'].shift(1)
    
    print("# 1期間後の値")
    print("df['value_lead1'] = df['value'].shift(-1)")
    df_time['value_lead1'] = df_time['value'].shift(-1)
    
    print("# 30分前の値（6期間前、5分間隔なので）")
    print("df['value_lag30min'] = df['value'].shift(6)")
    df_time['value_lag30min'] = df_time['value'].shift(6)
    
    # ケース6: 別々の日時・時刻列を1つのタイムスタンプに結合
    print("\n■ ケース6: 日時・時刻列の結合")
    
    # サンプルデータ作成（日時と時刻が別々）
    sample_datetime_data = pd.DataFrame({
        '日時': ['2025/06/12', '2025/06/12', '2025/06/13', '2025/06/13'],
        '時刻': ['8:00:00', '14:30:00', '9:15:00', '16:45:00'],
        'temperature': [85.2, 87.1, 84.8, 86.3]
    })
    
    print("# 元データ:")
    print(sample_datetime_data)
    
    print("\n# 方法1: 文字列結合してから変換")
    print("df['timestamp'] = pd.to_datetime(df['日時'] + ' ' + df['時刻'])")
    sample_datetime_data['timestamp'] = pd.to_datetime(sample_datetime_data['日時'] + ' ' + sample_datetime_data['時刻'])
    print("結果:")
    print(sample_datetime_data[['timestamp', 'temperature']])
    
    print("\n# 方法2: pd.to_datetimeで複数列を指定")
    sample_datetime_data2 = pd.DataFrame({
        'year': [2025, 2025, 2025],
        'month': [6, 6, 6],
        'day': [12, 12, 13],
        'hour': [8, 14, 9],
        'minute': [0, 30, 15],
        'temperature': [85.2, 87.1, 84.8]
    })
    print("df['timestamp'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])")
    sample_datetime_data2['timestamp'] = pd.to_datetime(sample_datetime_data2[['year', 'month', 'day', 'hour', 'minute']])
    print("結果:")
    print(sample_datetime_data2[['timestamp', 'temperature']])
    
    print("\n# 方法3: combine関数を使用")
    sample_datetime_data3 = pd.DataFrame({
        'date': pd.to_datetime(['2025/06/12', '2025/06/13', '2025/06/14']),
        'time': pd.to_datetime(['8:00:00', '14:30:00', '9:15:00'], format='%H:%M:%S').dt.time,
        'temperature': [85.2, 87.1, 84.8]
    })
    print("df['timestamp'] = df['date'].dt.date + pd.to_timedelta(df['time'].astype(str))")
    # より簡単な方法
    print("# または")
    print("df['timestamp'] = pd.to_datetime(df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['time'].astype(str))")
    sample_datetime_data3['timestamp'] = pd.to_datetime(
        sample_datetime_data3['date'].dt.strftime('%Y-%m-%d') + ' ' + 
        sample_datetime_data3['time'].astype(str)
    )
    print("結果:")
    print(sample_datetime_data3[['timestamp', 'temperature']])
    
    print("\n# よくある問題と解決方法:")
    print("🔸 異なる日時形式の場合:")
    mixed_format_data = pd.DataFrame({
        'date_col': ['2025-06-12', '06/13/2025', '2025/6/14'],
        'time_col': ['08:00', '2:30 PM', '09:15:30'],
        'value': [1, 2, 3]
    })
    print("# 形式を統一してから結合")
    print("df['date_normalized'] = pd.to_datetime(df['date_col']).dt.strftime('%Y-%m-%d')")
    print("df['time_normalized'] = pd.to_datetime(df['time_col'], format='mixed').dt.strftime('%H:%M:%S')")
    print("df['timestamp'] = pd.to_datetime(df['date_normalized'] + ' ' + df['time_normalized'])")
    
    print("\n🔸 欠損値がある場合:")
    print("df['timestamp'] = pd.to_datetime(df['日時'] + ' ' + df['時刻'], errors='coerce')")
    print("# errors='coerce'で変換できない値はNaTになります")
    
    return df_time

# =============================================================================
# 5. グループ集計・統計
# =============================================================================

def section_05_groupby_stats():
    """グループ集計・統計のコード例"""
    print("\n" + "=" * 60)
    print("5. グループ集計・統計")
    print("=" * 60)
    
    # サンプルデータ作成
    np.random.seed(42)
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=500, freq='10T'),
        'line_id': np.random.choice(['A', 'B', 'C'], 500),
        'shift': np.random.choice(['朝', '昼', '夜'], 500),
        'temperature': np.random.normal(85, 5, 500),
        'pressure': np.random.normal(2.0, 0.3, 500),
        'production': np.random.normal(100, 20, 500)
    })
    
    # ケース1: 基本的なグループ集計
    print("■ ケース1: 基本的なグループ集計")
    print("# ライン別平均")
    print("line_avg = df.groupby('line_id').mean()")
    line_avg = df.groupby('line_id')[['temperature', 'pressure', 'production']].mean()
    print(line_avg.round(2))
    
    print("\n# シフト別統計")
    print("shift_stats = df.groupby('shift').agg({'temperature': ['mean', 'std'], 'production': ['sum', 'count']})")
    shift_stats = df.groupby('shift').agg({
        'temperature': ['mean', 'std'], 
        'production': ['sum', 'count']
    })
    print(shift_stats.round(2))
    
    # ケース2: 複数キーでのグループ化
    print("\n■ ケース2: 複数キーでのグループ化")
    print("# ライン×シフト別集計")
    print("multi_group = df.groupby(['line_id', 'shift']).agg({'production': 'mean', 'temperature': 'std'})")
    multi_group = df.groupby(['line_id', 'shift']).agg({
        'production': 'mean',
        'temperature': 'std'
    })
    print(multi_group.round(2))
    
    # ケース3: カスタム集計関数
    print("\n■ ケース3: カスタム集計関数")
    print("# 変動係数の計算")
    print("cv = lambda x: x.std() / x.mean()")
    cv = lambda x: x.std() / x.mean()
    print("df.groupby('line_id')['temperature'].agg(cv)")
    cv_result = df.groupby('line_id')['temperature'].agg(cv)
    print(cv_result.round(3))
    
    print("\n# 複数のカスタム関数")
    print("""
custom_agg = df.groupby('line_id')['production'].agg([
    ('平均', 'mean'),
    ('最大', 'max'),
    ('範囲', lambda x: x.max() - x.min()),
    ('変動係数', lambda x: x.std() / x.mean())
])
""")
    custom_agg = df.groupby('line_id')['production'].agg([
        ('平均', 'mean'),
        ('最大', 'max'),
        ('範囲', lambda x: x.max() - x.min()),
        ('変動係数', lambda x: x.std() / x.mean())
    ])
    print(custom_agg.round(2))
    
    # ケース4: 時間軸での集計
    print("\n■ ケース4: 時間軸での集計")
    df_indexed = df.set_index('timestamp')
    
    print("# 時間別平均（ライン別）")
    print("hourly_by_line = df_indexed.groupby(['line_id', pd.Grouper(freq='1H')]).mean()")
    hourly_by_line = df_indexed.groupby(['line_id', pd.Grouper(freq='1H')])[['temperature', 'production']].mean()
    print(f"時間別データ: {len(hourly_by_line)}件")
    
    # ケース5: 条件付き集計
    print("\n■ ケース5: 条件付き集計")
    print("# 高温時のみの統計")
    print("high_temp_stats = df[df['temperature'] > 85].groupby('line_id')['production'].mean()")
    high_temp_stats = df[df['temperature'] > 85].groupby('line_id')['production'].mean()
    print(high_temp_stats.round(2))
    
    return df

# =============================================================================
# 6. 可視化（matplotlib/pyplot）
# =============================================================================

def section_06_visualization():
    """可視化のコード例"""
    print("\n" + "=" * 60)
    print("6. 可視化（matplotlib/pyplot）")
    print("=" * 60)
    
    # サンプルデータ作成
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1H')
    df = pd.DataFrame({
        'timestamp': dates,
        'temperature': 85 + 5 * np.sin(np.arange(200) * 2 * np.pi / 24) + np.random.normal(0, 1, 200),
        'pressure': 2.0 + 0.3 * np.sin(np.arange(200) * 2 * np.pi / 24) + np.random.normal(0, 0.1, 200),
        'status': np.random.choice(['正常', '警告', '異常'], 200, p=[0.8, 0.15, 0.05])
    })
    
    # ケース1: 基本的な線グラフ
    print("■ ケース1: 基本的な線グラフ")
    print("""
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['temperature'], label='温度')
plt.title('温度の時系列変化')
plt.xlabel('時間')
plt.ylabel('温度 (°C)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
""")
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['temperature'], label='温度', alpha=0.7)
    plt.title('温度の時系列変化')
    plt.xlabel('時間')
    plt.ylabel('温度 (°C)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # ケース2: 複数系列の可視化
    print("\n■ ケース2: 複数系列の可視化")
    print("""
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# 上段: 温度
axes[0].plot(df['timestamp'], df['temperature'], color='red', alpha=0.7)
axes[0].set_title('温度')
axes[0].set_ylabel('温度 (°C)')

# 下段: 圧力
axes[1].plot(df['timestamp'], df['pressure'], color='blue', alpha=0.7)
axes[1].set_title('圧力')
axes[1].set_ylabel('圧力 (MPa)')
axes[1].set_xlabel('時間')

plt.tight_layout()
plt.show()
""")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 上段: 温度
    axes[0].plot(df['timestamp'], df['temperature'], color='red', alpha=0.7)
    axes[0].set_title('温度')
    axes[0].set_ylabel('温度 (°C)')
    
    # 下段: 圧力
    axes[1].plot(df['timestamp'], df['pressure'], color='blue', alpha=0.7)
    axes[1].set_title('圧力')
    axes[1].set_ylabel('圧力 (MPa)')
    axes[1].set_xlabel('時間')
    
    plt.tight_layout()
    plt.show()
    
    # ケース3: 散布図と相関
    print("\n■ ケース3: 散布図と相関")
    print("""
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(df['temperature'], df['pressure'], alpha=0.6)
plt.xlabel('温度 (°C)')
plt.ylabel('圧力 (MPa)')
plt.title('温度 vs 圧力')

plt.subplot(1, 2, 2)
correlation = df[['temperature', 'pressure']].corr()
plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar()
plt.title('相関行列')
plt.xticks([0, 1], ['温度', '圧力'])
plt.yticks([0, 1], ['温度', '圧力'])

plt.tight_layout()
plt.show()
""")
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(df['temperature'], df['pressure'], alpha=0.6)
    plt.xlabel('温度 (°C)')
    plt.ylabel('圧力 (MPa)')
    plt.title('温度 vs 圧力')
    
    plt.subplot(1, 2, 2)
    correlation = df[['temperature', 'pressure']].corr()
    plt.imshow(correlation, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title('相関行列')
    plt.xticks([0, 1], ['温度', '圧力'])
    plt.yticks([0, 1], ['温度', '圧力'])
    
    plt.tight_layout()
    plt.show()
    
    # ケース4: ヒストグラムと分布
    print("\n■ ケース4: ヒストグラムと分布")
    print("""
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df['temperature'], bins=20, alpha=0.7, color='red', edgecolor='black')
plt.xlabel('温度 (°C)')
plt.ylabel('頻度')
plt.title('温度の分布')

plt.subplot(1, 2, 2)
plt.boxplot([df['temperature'], df['pressure']*40], labels=['温度', '圧力×40'])
plt.title('ボックスプロット')
plt.ylabel('値')

plt.tight_layout()
plt.show()
""")
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['temperature'], bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('温度 (°C)')
    plt.ylabel('頻度')
    plt.title('温度の分布')
    
    plt.subplot(1, 2, 2)
    plt.boxplot([df['temperature'], df['pressure']*40], labels=['温度', '圧力×40'])
    plt.title('ボックスプロット')
    plt.ylabel('値')
    
    plt.tight_layout()
    plt.show()
    
    # ケース5: カテゴリ別の可視化
    print("\n■ ケース5: カテゴリ別の可視化")
    print("""
plt.figure(figsize=(12, 6))

# 状態別の温度分布
status_colors = {'正常': 'green', '警告': 'orange', '異常': 'red'}

for status in df['status'].unique():
    data = df[df['status'] == status]['temperature']
    plt.hist(data, alpha=0.7, label=status, color=status_colors[status], bins=15)

plt.xlabel('温度 (°C)')
plt.ylabel('頻度')
plt.title('状態別温度分布')
plt.legend()
plt.show()
""")
    
    plt.figure(figsize=(12, 6))
    
    # 状態別の温度分布
    status_colors = {'正常': 'green', '警告': 'orange', '異常': 'red'}
    
    for status in df['status'].unique():
        data = df[df['status'] == status]['temperature']
        plt.hist(data, alpha=0.7, label=status, color=status_colors[status], bins=15)
    
    plt.xlabel('温度 (°C)')
    plt.ylabel('頻度')
    plt.title('状態別温度分布')
    plt.legend()
    plt.show()
    
    return df

# =============================================================================
# 7. データ出力・保存
# =============================================================================

def section_07_data_export():
    """データ出力・保存のコード例"""
    print("\n" + "=" * 60)
    print("7. データ出力・保存")
    print("=" * 60)
    
    # サンプルデータ作成
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'temperature': np.random.normal(85, 2, 100),
        'pressure': np.random.normal(2.0, 0.1, 100)
    })
    
    # ケース1: CSV出力
    print("■ ケース1: CSV出力")
    print("# 基本的なCSV出力")
    print("df.to_csv('output.csv', index=False)")
    print("# エンコーディング指定")
    print("df.to_csv('output.csv', index=False, encoding='utf-8')")
    print("# 特定列のみ出力")
    print("df[['timestamp', 'temperature']].to_csv('temp_only.csv', index=False)")
    
    # ケース2: Excel出力
    print("\n■ ケース2: Excel出力")
    print("# 基本的なExcel出力")
    print("df.to_excel('output.xlsx', index=False)")
    print("# 複数シートに出力")
    print("""
with pd.ExcelWriter('multi_sheet.xlsx') as writer:
    df.to_excel(writer, sheet_name='データ', index=False)
    df.describe().to_excel(writer, sheet_name='統計')
""")
    
    # ケース3: 条件付き出力
    print("\n■ ケース3: 条件付き出力")
    print("# 高温データのみ出力")
    print("high_temp = df[df['temperature'] > 85]")
    print("high_temp.to_csv('high_temperature.csv', index=False)")
    
    # ケース4: 集計結果の出力
    print("\n■ ケース4: 集計結果の出力")
    print("# 時間別統計")
    df_indexed = df.set_index('timestamp')
    hourly_stats = df_indexed.resample('1D').agg({
        'temperature': ['mean', 'min', 'max'],
        'pressure': ['mean', 'std']
    })
    print("hourly_stats.to_csv('hourly_statistics.csv')")
    
    # ケース5: データ形式の変換
    print("\n■ ケース5: データ形式の変換")
    print("# JSON形式")
    print("df.to_json('data.json', orient='records', date_format='iso')")
    print("# Pickle形式（pandas専用）")
    print("df.to_pickle('data.pkl')")
    print("# 読み込み")
    print("df_loaded = pd.read_pickle('data.pkl')")
    
    return df

# =============================================================================
# 8. トラブルシューティング
# =============================================================================

def section_08_troubleshooting():
    """よくあるエラーと解決方法"""
    print("\n" + "=" * 60)
    print("8. トラブルシューティング")
    print("=" * 60)
    
    print("■ よくあるエラーと解決方法")
    
    print("\n🔸 エラー1: KeyError: 'column_name'")
    print("原因: 指定した列名が存在しない")
    print("解決方法:")
    print("# 列名の確認")
    print("print(df.columns.tolist())")
    print("# 列名の存在確認")
    print("if 'column_name' in df.columns:")
    print("    # 処理実行")
    
    print("\n🔸 エラー2: ValueError: could not convert string to float")
    print("原因: 数値に変換できない文字列が含まれている")
    print("解決方法:")
    print("# 数値変換（エラー値はNaNに）")
    print("df['column'] = pd.to_numeric(df['column'], errors='coerce')")
    print("# 文字列のクリーニング")
    print("df['column'] = df['column'].str.replace(',', '').astype(float)")
    
    print("\n🔸 エラー3: MemoryError")
    print("原因: メモリ不足（大きなファイル）")
    print("解決方法:")
    print("# チャンク読み込み")
    print("for chunk in pd.read_csv('large_file.csv', chunksize=10000):")
    print("    # チャンクごとに処理")
    print("# 必要な列のみ読み込み")
    print("df = pd.read_csv('file.csv', usecols=['col1', 'col2'])")
    
    print("\n🔸 エラー4: SettingWithCopyWarning")
    print("原因: DataFrameのコピーに対する曖昧な操作")
    print("解決方法:")
    print("# .copy()を明示的に使用")
    print("df_subset = df[df['column'] > 0].copy()")
    print("df_subset['new_column'] = value")
    print("# または.loc[]を使用")
    print("df.loc[df['column'] > 0, 'new_column'] = value")
    
    print("\n🔸 エラー5: 日時変換エラー")
    print("原因: 日時形式が認識できない")
    print("解決方法:")
    print("# 形式を明示的に指定")
    print("df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')")
    print("# エラー時はNaTに")
    print("df['date'] = pd.to_datetime(df['date'], errors='coerce')")
    
    print("\n■ パフォーマンス最適化のコツ")
    print("1. .apply()よりもベクトル化操作を使用")
    print("2. 大きなデータには.query()を使用")
    print("3. 不要な列は早めに削除")
    print("4. データ型を適切に設定（int64 → int32など）")
    print("5. インデックスを効果的に活用")

# =============================================================================
# メイン実行部分
# =============================================================================

def run_all_examples():
    """全ての例を順次実行"""
    print("🐼 Pandas & Pyplot 実用的使い方講座")
    print("=" * 80)
    
    # 各セクションを実行
    df1 = section_01_data_loading()
    df2 = section_02_data_merging()
    df3 = section_03_data_transformation()
    df4 = section_04_time_series()
    df5 = section_05_groupby_stats()
    df6 = section_06_visualization()
    df7 = section_07_data_export()
    section_08_troubleshooting()
    
    print("\n" + "=" * 80)
    print("✅ 全てのセクションが完了しました！")
    print("=" * 80)
    
    print("\n📚 学習のポイント:")
    print("1. 各コードは独立して実行可能です")
    print("2. 実際のデータで試してみてください")
    print("3. エラーが出たら Section 8 を参考にしてください")
    print("4. 可視化は用途に応じてカスタマイズしてください")
    print("\n💡 次のステップ:")
    print("- 実際の工場データでこれらの手法を試す")
    print("- Seabornやplotlyでより高度な可視化を学ぶ")
    print("- 機械学習ライブラリ（scikit-learn）との連携を学ぶ")

def show_quick_reference():
    """クイックリファレンス表示"""
    print("\n" + "=" * 60)
    print("📖 クイックリファレンス")
    print("=" * 60)
    
    reference = {
        "データ読み込み": [
            "pd.read_csv('file.csv')",
            "pd.read_excel('file.xlsx')",
            "pd.read_json('file.json')"
        ],
        "データ確認": [
            "df.head(), df.tail()",
            "df.info(), df.describe()",
            "df.shape, df.columns"
        ],
        "データ選択": [
            "df['column']",
            "df[['col1', 'col2']]",
            "df[df['col'] > value]"
        ],
        "データ結合": [
            "pd.merge(df1, df2, on='key')",
            "pd.concat([df1, df2])",
            "df1.join(df2)"
        ],
        "グループ集計": [
            "df.groupby('col').mean()",
            "df.groupby('col').agg({'col2': 'sum'})",
            "df.resample('1H').mean()"
        ],
        "データ変換": [
            "df['new_col'] = df['col'].apply(func)",
            "df['col'].fillna(value)",
            "df['col'].rolling(window=5).mean()"
        ],
        "可視化": [
            "plt.plot(x, y)",
            "plt.scatter(x, y)",
            "plt.hist(data, bins=20)"
        ],
        "データ出力": [
            "df.to_csv('file.csv')",
            "df.to_excel('file.xlsx')",
            "df.to_json('file.json')"
        ]
    }
    
    for category, commands in reference.items():
        print(f"\n■ {category}")
        for cmd in commands:
            print(f"  {cmd}")

if __name__ == "__main__":
    print("実行方法を選択してください:")
    print("1. 全セクション実行: run_all_examples()")
    print("2. 個別セクション実行: section_01_data_loading() など")
    print("3. クイックリファレンス: show_quick_reference()")
    print("\n例: python pandas_pyplot_cookbook.py")
    
    # 全セクション一括実行
    run_all_examples()

    # 個別セクション実行
    section_01_data_loading()  # データ読み込み
    section_02_data_merging()  # データ結合
    # など

    # クイックリファレンス表示
    show_quick_reference() 