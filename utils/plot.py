import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main(df : pd.DataFrame, continuous=False) -> matplotlib.figure.Figure:
    """
    データフレームのデフォルトのプロットを作成します。

    Args:
        df: データフレーム
        continuous: Trueの場合、連続時間表示（空白期間除去）
        
    """ 
    df.index = pd.to_datetime(df.index)
    col = 'T731.PV'
    
    if continuous:
        return plot_continuous_time(df, col)
    else:
        return plot_normal_time(df, col)

def plot_normal_time(df: pd.DataFrame, col: str) -> matplotlib.figure.Figure:
    """通常の時間軸でプロット"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index.to_pydatetime(), df[col], label=col, color='blue')

    unique_varieties = df['type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_varieties)))
    variety_colors_map = dict(zip(unique_varieties, colors))

    segments = []
    start_idx = 0
    for i in range(1, len(df)):
        if df['type'].iloc[i] != df['type'].iloc[i-1]:
            segments.append((df.index[start_idx], df.index[i-1], df['type'].iloc[i-1]))
            start_idx = i
    segments.append((df.index[start_idx], df.index[-1], df['type'].iloc[-1]))
    
    added_labels = set()
    for start, end, variety in segments:
        label = f'品種{variety}' if variety not in added_labels else None
        ax.axvspan(start.to_pydatetime(), end.to_pydatetime(), 
                  color=variety_colors_map[variety], alpha=0.3, label=label)
        added_labels.add(variety)
    
    ax.set_xlabel('Time')
    ax.set_ylabel(col)
    ax.set_title(f'{col} over Time')
    ax.legend()
    plt.show()
    
    return fig

def plot_continuous_time(df: pd.DataFrame, col: str) -> matplotlib.figure.Figure:
    """連続時間表示（空白期間除去）でプロット"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 品種の変化点を検出してセグメントに分割
    segments = []
    start_idx = 0
    
    for i in range(1, len(df)):
        if df['type'].iloc[i] != df['type'].iloc[i-1]:
            segments.append({
                'start_idx': start_idx,
                'end_idx': i-1,
                'variety': df['type'].iloc[i-1],
                'data': df.iloc[start_idx:i]
            })
            start_idx = i
    
    # 最後のセグメントを追加
    segments.append({
        'start_idx': start_idx,
        'end_idx': len(df)-1,
        'variety': df['type'].iloc[-1],
        'data': df.iloc[start_idx:]
    })
    
    # 色設定
    unique_varieties = df['type'].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_varieties)))
    variety_colors_map = dict(zip(unique_varieties, colors))
    
    # 連続位置でプロット
    current_pos = 0
    x_positions = []
    y_values = []
    segment_boundaries = []
    segment_labels = []
    added_labels = set()
    
    for i, segment in enumerate(segments):
        segment_data = segment['data']
        variety = segment['variety']
        
        # セグメントの開始位置を記録
        if current_pos > 0:
            segment_boundaries.append(current_pos)
        
        # このセグメントのX位置
        segment_x = list(range(current_pos, current_pos + len(segment_data)))
        
        # 背景色設定
        label = f'品種{variety}' if variety not in added_labels else None
        ax.axvspan(current_pos, current_pos + len(segment_data) - 1,
                  color=variety_colors_map[variety], alpha=0.3, label=label)
        added_labels.add(variety)
        
        # データプロット
        ax.plot(segment_x, segment_data[col], color='blue', linewidth=1.5)
        
        # セグメント境界に縦線
        if current_pos > 0:
            ax.axvline(current_pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # セグメントラベル用の位置
        mid_pos = current_pos + len(segment_data) // 2
        segment_labels.append((mid_pos, variety, segment_data.index[0], segment_data.index[-1]))
        
        # X位置とY値を記録
        x_positions.extend(segment_x)
        y_values.extend(segment_data[col])
        
        current_pos += len(segment_data)
    
    # X軸のカスタムラベル設定
    tick_positions = []
    tick_labels = []
    
    for mid_pos, variety, start_time, end_time in segment_labels:
        tick_positions.append(mid_pos)
        tick_labels.append(f'品種{variety}\n{start_time.strftime("%m/%d %H:%M")}\n～{end_time.strftime("%H:%M")}')
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_xlabel('連続データポイント（空白期間除去）')
    ax.set_ylabel(col)
    ax.set_title(f'{col} - 連続時間表示（品種別背景）')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_by_variety(df: pd.DataFrame, col: str) -> matplotlib.figure.Figure:
    """品種別にプロットを分割表示"""
    unique_varieties = sorted(df['type'].unique())
    n_varieties = len(unique_varieties)
    
    # サブプロット作成
    fig, axes = plt.subplots(n_varieties, 1, figsize=(12, 4 * n_varieties))
    if n_varieties == 1:
        axes = [axes]
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_varieties))
    
    for i, variety in enumerate(unique_varieties):
        ax = axes[i]
        variety_data = df[df['type'] == variety]
        
        ax.plot(variety_data.index.to_pydatetime(), variety_data[col], 
               color=colors[i], linewidth=1.5, label=f'品種{variety}')
        
        ax.set_title(f'品種{variety} - {col}', fontsize=12, fontweight='bold')
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 最後のサブプロットにのみX軸ラベル
        if i == n_varieties - 1:
            ax.set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_by_equipment(df: pd.DataFrame, col: str, equipment_col: str = 'equipment') -> matplotlib.figure.Figure:
    """設備別にプロットを分割表示"""
    if equipment_col not in df.columns:
        print(f"警告: '{equipment_col}' 列が見つかりません。")
        return plot_normal_time(df, col)
    
    unique_equipment = sorted(df[equipment_col].unique())
    n_equipment = len(unique_equipment)
    
    # サブプロット作成
    fig, axes = plt.subplots(n_equipment, 1, figsize=(12, 4 * n_equipment))
    if n_equipment == 1:
        axes = [axes]
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, equipment in enumerate(unique_equipment):
        ax = axes[i]
        equipment_data = df[df[equipment_col] == equipment]
        
        # 品種別背景色も追加
        unique_varieties = equipment_data['type'].unique()
        variety_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_varieties)))
        variety_colors_map = dict(zip(unique_varieties, variety_colors))
        
        # 背景塗り分け
        segments = []
        start_idx = 0
        for j in range(1, len(equipment_data)):
            if equipment_data['type'].iloc[j] != equipment_data['type'].iloc[j-1]:
                segments.append((equipment_data.index[start_idx], equipment_data.index[j-1], 
                               equipment_data['type'].iloc[j-1]))
                start_idx = j
        segments.append((equipment_data.index[start_idx], equipment_data.index[-1], 
                        equipment_data['type'].iloc[-1]))
        
        added_labels = set()
        for start, end, variety in segments:
            label = f'品種{variety}' if variety not in added_labels else None
            ax.axvspan(start.to_pydatetime(), end.to_pydatetime(),
                      color=variety_colors_map[variety], alpha=0.3, label=label)
            added_labels.add(variety)
        
        ax.plot(equipment_data.index.to_pydatetime(), equipment_data[col], 
               color=colors[i % len(colors)], linewidth=1.5, label=f'設備{equipment}')
        
        ax.set_title(f'設備{equipment} - {col}', fontsize=12, fontweight='bold')
        ax.set_ylabel(col)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 最後のサブプロットにのみX軸ラベル
        if i == n_equipment - 1:
            ax.set_xlabel('Time')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_equipment_continuous(df: pd.DataFrame, col: str, equipment_col: str = 'equipment') -> matplotlib.figure.Figure:
    """設備別連続時間表示"""
    if equipment_col not in df.columns:
        print(f"警告: '{equipment_col}' 列が見つかりません。")
        return plot_continuous_time(df, col)
    
    unique_equipment = sorted(df[equipment_col].unique())
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    current_global_pos = 0
    all_tick_positions = []
    all_tick_labels = []
    
    for eq_idx, equipment in enumerate(unique_equipment):
        equipment_data = df[df[equipment_col] == equipment].sort_index()
        
        if len(equipment_data) == 0:
            continue
        
        # 品種セグメント分割
        segments = []
        start_idx = 0
        
        for i in range(1, len(equipment_data)):
            if equipment_data['type'].iloc[i] != equipment_data['type'].iloc[i-1]:
                segments.append({
                    'start_idx': start_idx,
                    'end_idx': i-1,
                    'variety': equipment_data['type'].iloc[i-1],
                    'data': equipment_data.iloc[start_idx:i]
                })
                start_idx = i
        
        segments.append({
            'start_idx': start_idx,
            'end_idx': len(equipment_data)-1,
            'variety': equipment_data['type'].iloc[-1],
            'data': equipment_data.iloc[start_idx:]
        })
        
        # 品種色設定
        unique_varieties = equipment_data['type'].unique()
        variety_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_varieties)))
        variety_colors_map = dict(zip(unique_varieties, variety_colors))
        
        equipment_start_pos = current_global_pos
        
        for segment in segments:
            segment_data = segment['data']
            variety = segment['variety']
            
            # セグメントのX位置
            segment_x = list(range(current_global_pos, current_global_pos + len(segment_data)))
            
            # 背景色
            ax.axvspan(current_global_pos, current_global_pos + len(segment_data) - 1,
                      color=variety_colors_map[variety], alpha=0.2)
            
            # データプロット
            ax.plot(segment_x, segment_data[col], 
                   color=colors[eq_idx % len(colors)], linewidth=1.5, alpha=0.8)
            
            current_global_pos += len(segment_data)
        
        # 設備境界に縦線
        if equipment_start_pos > 0:
            ax.axvline(equipment_start_pos, color='black', linestyle='-', alpha=0.8, linewidth=3)
        
        # 設備ラベル
        equipment_mid_pos = equipment_start_pos + (current_global_pos - equipment_start_pos) // 2
        all_tick_positions.append(equipment_mid_pos)
        all_tick_labels.append(f'設備{equipment}')
        
        # 小さなギャップを追加
        current_global_pos += 10
    
    ax.set_xticks(all_tick_positions)
    ax.set_xticklabels(all_tick_labels, rotation=45, ha='right')
    ax.set_xlabel('連続データポイント（設備・品種別）')
    ax.set_ylabel(col)
    ax.set_title(f'{col} - 設備別連続表示（品種背景付き）')
    ax.grid(True, alpha=0.3)
    
    # 凡例作成
    equipment_legend = [plt.Line2D([0], [0], color=colors[i % len(colors)], linewidth=2, 
                                  label=f'設備{eq}') for i, eq in enumerate(unique_equipment)]
    ax.legend(handles=equipment_legend, loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig

# 使用例関数
def demo_all_plots(df: pd.DataFrame, col: str = 'T731.PV', equipment_col: str = 'equipment'):
    """全てのプロット機能のデモ"""
    print("🎨 全プロット機能デモ")
    print("=" * 50)
    
    print("1. 通常時間表示（品種背景）")
    plot_normal_time(df, col)
    
    print("2. 連続時間表示（品種背景）")
    plot_continuous_time(df, col)
    
    print("3. 品種別分割表示")
    plot_by_variety(df, col)
    
    if equipment_col in df.columns:
        print("4. 設備別分割表示（品種背景付き）")
        plot_by_equipment(df, col, equipment_col)
        
        print("5. 設備別連続時間表示")
        plot_equipment_continuous(df, col, equipment_col)
    else:
        print(f"設備列 '{equipment_col}' が見つからないため、設備別プロットをスキップ")
    
    print("🎉 デモ完了!")

if __name__ == "__main__":
    # テスト用のサンプルデータ作成
    import datetime
    
    dates = pd.date_range('2024-01-01 10:00', periods=1000, freq='5min')
    
    # サンプルデータ
    np.random.seed(42)
    sample_df = pd.DataFrame({
        'T731.PV': np.random.normal(100, 10, 1000),
        'type': np.random.choice(['H', 'L'], 1000),
        'equipment': np.random.choice([1, 2], 1000)
    }, index=dates)
    
    # デモ実行
    demo_all_plots(sample_df)