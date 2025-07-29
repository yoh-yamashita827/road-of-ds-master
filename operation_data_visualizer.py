#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稼働表データ可視化ツール - 品種別背景塗り分け機能付き

品種タグに応じて背景を塗り分けて、時系列データを可視化します。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import japanize_matplotlib
import warnings
warnings.filterwarnings('ignore')

class OperationDataVisualizer:
    """稼働データ可視化クラス"""
    
    def __init__(self):
        self.data = None
        self.operation_periods = []
        
    def load_extracted_data(self, csv_file):
        """抽出済みデータを読み込み"""
        try:
            df = pd.read_csv(csv_file, index_col=0)
            # インデックスをdatetimeに変換
            df.index = pd.to_datetime(df.index)
            
            # 日時関連の列もdatetimeに変換
            datetime_cols = ['稼働開始時刻', '稼働終了時刻']
            for col in datetime_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col])
            
            # 稼働日列の処理
            if '稼働日' in df.columns:
                df['稼働日'] = pd.to_datetime(df['稼働日']).dt.date
            
            self.data = df
            print(f"✅ データ読み込み完了: {df.shape}")
            print(f"期間: {df.index.min()} ～ {df.index.max()}")
            return df
        except Exception as e:
            print(f"❌ データ読み込みエラー: {e}")
            return None
    
    def extract_operation_periods(self):
        """稼働期間情報を抽出"""
        if self.data is None:
            print("❌ データが読み込まれていません")
            return
        
        # 稼働期間ごとにグループ化
        periods = []
        
        # 稼働開始時刻と品種でグループ化
        grouped = self.data.groupby(['稼働開始時刻', '品種', '設備タグ'])
        
        for (start_time, product, equipment_tag), group in grouped:
            end_time = group['稼働終了時刻'].iloc[0]
            
            # 文字列の場合はdatetimeに変換
            if isinstance(start_time, str):
                start_time = pd.to_datetime(start_time)
            if isinstance(end_time, str):
                end_time = pd.to_datetime(end_time)
            
            periods.append({
                'start': start_time,
                'end': end_time,
                'product': product,
                'equipment_tag': equipment_tag,
                'data_count': len(group)
            })
        
        self.operation_periods = sorted(periods, key=lambda x: x['start'])
        print(f"✅ 稼働期間抽出完了: {len(self.operation_periods)}期間")
        
        for i, period in enumerate(self.operation_periods):
            print(f"期間{i+1}: {period['start']} ～ {period['end']} "
                  f"品種{period['product']} 設備{period['equipment_tag']} ({period['data_count']}件)")
    
    def plot_with_product_background(self, columns=None, figsize=(15, 10), save_file=None):
        """
        品種別背景塗り分けプロット
        
        Parameters:
        -----------
        columns : list, optional
            プロットする列名のリスト（指定しない場合は数値列を自動選択）
        figsize : tuple
            図のサイズ
        save_file : str, optional
            保存ファイル名
        """
        if self.data is None:
            print("❌ データが読み込まれていません")
            return
        
        if not self.operation_periods:
            self.extract_operation_periods()
        
        # 数値列を自動選択
        if columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_cols if col not in ['設備タグ']][:4]
        
        # 品種別の色設定
        product_colors = {
            'H': {'color': 'lightblue', 'alpha': 0.3, 'label': '品種H'},
            'L': {'color': 'lightcoral', 'alpha': 0.3, 'label': '品種L'},
            'M': {'color': 'lightgreen', 'alpha': 0.3, 'label': '品種M'},
            'S': {'color': 'lightyellow', 'alpha': 0.3, 'label': '品種S'}
        }
        
        # サブプロット作成
        n_cols = min(len(columns), 2)
        n_rows = (len(columns) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.ravel()
        
        # 各列についてプロット
        for i, col in enumerate(columns):
            ax = axes[i]
            
            # 背景塗り分け
            for period in self.operation_periods:
                product = period['product']
                if product in product_colors:
                    ax.axvspan(period['start'], period['end'], 
                             color=product_colors[product]['color'],
                             alpha=product_colors[product]['alpha'],
                             label=product_colors[product]['label'] if i == 0 else "")
            
            # データプロット
            ax.plot(self.data.index, self.data[col], 
                   color='navy', linewidth=1.5, alpha=0.8)
            
            # 設備タグ別に点をプロット
            equipment_tags = self.data['設備タグ'].unique()
            equipment_colors = ['red', 'blue', 'green', 'orange']
            
            for j, tag in enumerate(equipment_tags):
                tag_data = self.data[self.data['設備タグ'] == tag]
                ax.scatter(tag_data.index, tag_data[col], 
                          color=equipment_colors[j % len(equipment_colors)],
                          s=10, alpha=0.6, label=f'設備{tag}' if i == 0 else "")
            
            # 軸設定
            ax.set_title(f'{col} - 品種別稼働状況', fontsize=12, fontweight='bold')
            ax.set_ylabel(col, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # X軸の日時フォーマット
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 凡例を最初のサブプロットに追加
        if len(axes) > 0:
            # 重複を除いた凡例作成
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[0].legend(by_label.values(), by_label.keys(), 
                          loc='upper right', fontsize=8)
        
        # 余分なサブプロットを非表示
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # 保存
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✅ 図を保存しました: {save_file}")
        
        plt.show()
    
    def plot_with_product_background_continuous(self, columns=None, figsize=(15, 10), save_file=None):
        """
        品種別背景塗り分けプロット（連続表示版）
        データがある期間だけを連続的に表示し、空白期間を除去
        
        Parameters:
        -----------
        columns : list, optional
            プロットする列名のリスト（指定しない場合は数値列を自動選択）
        figsize : tuple
            図のサイズ
        save_file : str, optional
            保存ファイル名
        """
        if self.data is None:
            print("❌ データが読み込まれていません")
            return
        
        if not self.operation_periods:
            self.extract_operation_periods()
        
        # 数値列を自動選択
        if columns is None:
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            columns = [col for col in numeric_cols if col not in ['設備タグ']][:4]
        
        # 品種別の色設定
        product_colors = {
            'H': {'color': 'lightblue', 'alpha': 0.3, 'label': '品種H'},
            'L': {'color': 'lightcoral', 'alpha': 0.3, 'label': '品種L'},
            'M': {'color': 'lightgreen', 'alpha': 0.3, 'label': '品種M'},
            'S': {'color': 'lightyellow', 'alpha': 0.3, 'label': '品種S'}
        }
        
        # 連続データの作成
        continuous_data = []
        position_map = {}  # 元の時刻と新しい位置のマッピング
        current_pos = 0
        
        for period in self.operation_periods:
            # 各期間のデータを取得
            mask = (self.data.index >= period['start']) & (self.data.index <= period['end'])
            period_data = self.data[mask].sort_index()
            
            if len(period_data) > 0:
                # 新しい連続位置を割り当て
                for i, (timestamp, row) in enumerate(period_data.iterrows()):
                    position_map[timestamp] = current_pos + i
                
                continuous_data.append(period_data)
                current_pos += len(period_data)
        
        # 連続データを結合
        if not continuous_data:
            print("❌ 表示するデータがありません")
            return
        
        combined_data = pd.concat(continuous_data)
        x_positions = [position_map[ts] for ts in combined_data.index]
        
        # サブプロット作成
        n_cols = min(len(columns), 2)
        n_rows = (len(columns) + 1) // 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.ravel()
        
        # 各列についてプロット
        for i, col in enumerate(columns):
            ax = axes[i]
            
            # 背景塗り分け（連続位置で）
            current_pos = 0
            for j, period in enumerate(self.operation_periods):
                mask = (self.data.index >= period['start']) & (self.data.index <= period['end'])
                period_data = self.data[mask].sort_index()
                
                if len(period_data) > 0:
                    start_pos = current_pos
                    end_pos = current_pos + len(period_data) - 1
                    
                    product = period['product']
                    if product in product_colors:
                        ax.axvspan(start_pos, end_pos, 
                                 color=product_colors[product]['color'],
                                 alpha=product_colors[product]['alpha'],
                                 label=product_colors[product]['label'] if i == 0 and j == 0 else "")
                    
                    current_pos += len(period_data)
            
            # データプロット（連続位置で）
            ax.plot(x_positions, combined_data[col], 
                   color='navy', linewidth=1.5, alpha=0.8)
            
            # 設備タグ別に点をプロット
            equipment_tags = combined_data['設備タグ'].unique()
            equipment_colors = ['red', 'blue', 'green', 'orange']
            
            for j, tag in enumerate(equipment_tags):
                tag_mask = combined_data['設備タグ'] == tag
                tag_x = [x_positions[k] for k, is_tag in enumerate(tag_mask) if is_tag]
                tag_y = combined_data[tag_mask][col]
                
                ax.scatter(tag_x, tag_y, 
                          color=equipment_colors[j % len(equipment_colors)],
                          s=10, alpha=0.6, label=f'設備{tag}' if i == 0 else "")
            
            # 軸設定
            ax.set_title(f'{col} - 品種別稼働状況（連続表示）', fontsize=12, fontweight='bold')
            ax.set_ylabel(col, fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # X軸のカスタムラベル（期間境界に時刻表示）
            tick_positions = []
            tick_labels = []
            current_pos = 0
            
            for period in self.operation_periods:
                mask = (self.data.index >= period['start']) & (self.data.index <= period['end'])
                period_data = self.data[mask].sort_index()
                
                if len(period_data) > 0:
                    # 期間の開始位置にラベル
                    tick_positions.append(current_pos)
                    tick_labels.append(f"品種{period['product']}\n{period['start'].strftime('%m/%d %H:%M')}")
                    current_pos += len(period_data)
            
            # 最後の期間の終了も追加
            if self.operation_periods:
                last_period = self.operation_periods[-1]
                tick_positions.append(current_pos - 1)
                tick_labels.append(f"{last_period['end'].strftime('%m/%d %H:%M')}")
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        
        # 凡例を最初のサブプロットに追加
        if len(axes) > 0:
            handles, labels = axes[0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            axes[0].legend(by_label.values(), by_label.keys(), 
                          loc='upper right', fontsize=8)
        
        # 余分なサブプロットを非表示
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # 保存
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✅ 連続表示図を保存しました: {save_file}")
        
        plt.show()
    
    def plot_operation_timeline(self, figsize=(15, 6), save_file=None):
        """
        稼働タイムライン表示（ガントチャート風）
        
        Parameters:
        -----------
        figsize : tuple
            図のサイズ
        save_file : str, optional
            保存ファイル名
        """
        if not self.operation_periods:
            self.extract_operation_periods()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 品種別の色設定
        product_colors = {
            'H': 'lightblue',
            'L': 'lightcoral',
            'M': 'lightgreen',
            'S': 'lightyellow'
        }
        
        # 各稼働期間を横棒グラフで表示
        for i, period in enumerate(self.operation_periods):
            start = period['start']
            end = period['end']
            duration = (end - start).total_seconds() / 3600  # 時間単位
            
            product = period['product']
            equipment_tag = period['equipment_tag']
            
            color = product_colors.get(product, 'lightgray')
            
            # 設備タグごとに縦位置を調整
            y_pos = equipment_tag - 0.4 + (i % 2) * 0.8
            
            ax.barh(y_pos, duration, left=mdates.date2num(start), 
                   height=0.3, color=color, alpha=0.7, 
                   edgecolor='black', linewidth=1)
            
            # ラベル追加
            mid_time = start + (end - start) / 2
            ax.text(mdates.date2num(mid_time), y_pos, 
                   f'品種{product}\n{start.strftime("%m/%d %H:%M")}\n～{end.strftime("%H:%M")}',
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 軸設定
        ax.set_xlabel('時間', fontsize=12)
        ax.set_ylabel('設備タグ', fontsize=12)
        ax.set_title('稼働スケジュール タイムライン', fontsize=14, fontweight='bold')
        
        # Y軸設定
        equipment_tags = sorted(set(p['equipment_tag'] for p in self.operation_periods))
        ax.set_yticks(equipment_tags)
        ax.set_yticklabels([f'設備{tag}' for tag in equipment_tags])
        
        # X軸の日時フォーマット
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax.grid(True, alpha=0.3)
        
        # 凡例追加
        legend_elements = []
        for product, color in product_colors.items():
            if any(p['product'] == product for p in self.operation_periods):
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=f'品種{product}'))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # 保存
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✅ タイムライン図を保存しました: {save_file}")
        
        plt.show()
    
    def plot_operation_gantt_chart(self, figsize=(15, 8), save_file=None):
        """
        稼働ガントチャート（改良版）
        データがある期間だけを表示し、期間間の空白を明示
        
        Parameters:
        -----------
        figsize : tuple
            図のサイズ
        save_file : str, optional
            保存ファイル名
        """
        if not self.operation_periods:
            self.extract_operation_periods()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 3])
        
        # 品種別の色設定
        product_colors = {
            'H': 'lightblue',
            'L': 'lightcoral',
            'M': 'lightgreen',
            'S': 'lightyellow'
        }
        
        # 上段：稼働タイムライン（実時間）
        for i, period in enumerate(self.operation_periods):
            start = period['start']
            end = period['end']
            duration = (end - start).total_seconds() / 3600  # 時間単位
            
            product = period['product']
            equipment_tag = period['equipment_tag']
            color = product_colors.get(product, 'lightgray')
            
            # 設備タグごとに縦位置を調整
            y_pos = equipment_tag - 0.4
            
            ax1.barh(y_pos, duration, left=mdates.date2num(start), 
                    height=0.3, color=color, alpha=0.7, 
                    edgecolor='black', linewidth=1)
            
            # ラベル追加
            mid_time = start + (end - start) / 2
            ax1.text(mdates.date2num(mid_time), y_pos, 
                    f'品種{product}',
                    ha='center', va='center', fontsize=10, fontweight='bold')
        
        # 上段の軸設定
        ax1.set_title('実時間での稼働スケジュール', fontsize=12, fontweight='bold')
        ax1.set_ylabel('設備タグ', fontsize=10)
        
        equipment_tags = sorted(set(p['equipment_tag'] for p in self.operation_periods))
        ax1.set_yticks(equipment_tags)
        ax1.set_yticklabels([f'設備{tag}' for tag in equipment_tags])
        
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax1.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 下段：連続表示でのデータ
        if self.data is not None:
            # 主要な数値列を選択
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            target_col = [col for col in numeric_cols if col not in ['設備タグ']][0]
            
            x_pos = 0
            all_x = []
            all_y = []
            colors_list = []
            
            for period in self.operation_periods:
                mask = (self.data.index >= period['start']) & (self.data.index <= period['end'])
                period_data = self.data[mask].sort_index()
                
                if len(period_data) > 0:
                    # X位置を連続的に設定
                    period_x = list(range(x_pos, x_pos + len(period_data)))
                    period_y = period_data[target_col].tolist()
                    
                    # 背景色設定
                    product = period['product']
                    color = product_colors.get(product, 'lightgray')
                    
                    # 背景塗り分け
                    ax2.axvspan(x_pos, x_pos + len(period_data) - 1, 
                               color=color, alpha=0.3)
                    
                    # データプロット
                    ax2.plot(period_x, period_y, 'navy-', linewidth=1.5, alpha=0.8)
                    
                    # 期間境界に縦線
                    if x_pos > 0:
                        ax2.axvline(x_pos, color='red', linestyle='--', alpha=0.7, linewidth=2)
                    
                    # 期間ラベル
                    mid_x = x_pos + len(period_data) // 2
                    ax2.text(mid_x, ax2.get_ylim()[1] * 0.9, 
                            f'品種{product}\n{period["start"].strftime("%m/%d %H:%M")}\n～{period["end"].strftime("%H:%M")}',
                            ha='center', va='top', fontsize=8, 
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
                    
                    x_pos += len(period_data)
        
        ax2.set_title(f'{target_col} - 連続表示（空白期間除去）', fontsize=12, fontweight='bold')
        ax2.set_xlabel('データポイント番号', fontsize=10)
        ax2.set_ylabel(target_col, fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 凡例追加
        legend_elements = []
        for product, color in product_colors.items():
            if any(p['product'] == product for p in self.operation_periods):
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=f'品種{product}'))
        
        if legend_elements:
            ax2.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # 保存
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✅ ガントチャートを保存しました: {save_file}")
        
        plt.show()
    
    def plot_detailed_analysis(self, target_column='temperature', figsize=(15, 12), save_file=None):
        """
        詳細分析プロット（統計情報付き）
        
        Parameters:
        -----------
        target_column : str
            分析対象の列名
        figsize : tuple
            図のサイズ
        save_file : str, optional
            保存ファイル名
        """
        if self.data is None:
            print("❌ データが読み込まれていません")
            return
        
        if not self.operation_periods:
            self.extract_operation_periods()
        
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        
        # 品種別の色設定
        product_colors = {
            'H': 'lightblue',
            'L': 'lightcoral',
            'M': 'lightgreen',
            'S': 'lightyellow'
        }
        
        # 1. 時系列プロット（背景塗り分け）
        ax1 = axes[0, 0]
        for period in self.operation_periods:
            product = period['product']
            ax1.axvspan(period['start'], period['end'], 
                       color=product_colors.get(product, 'lightgray'),
                       alpha=0.3)
        
        ax1.plot(self.data.index, self.data[target_column], color='navy', linewidth=1)
        ax1.set_title(f'{target_column} 時系列 - 品種別背景')
        ax1.set_ylabel(target_column)
        ax1.grid(True, alpha=0.3)
        
        # 2. 品種別ボックスプロット
        ax2 = axes[0, 1]
        products = self.data['品種'].unique()
        box_data = [self.data[self.data['品種'] == p][target_column] for p in products]
        box_colors = [product_colors.get(p, 'lightgray') for p in products]
        
        bp = ax2.boxplot(box_data, labels=[f'品種{p}' for p in products], patch_artist=True)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
        
        ax2.set_title(f'{target_column} 品種別分布')
        ax2.set_ylabel(target_column)
        ax2.grid(True, alpha=0.3)
        
        # 3. 設備タグ別時系列
        ax3 = axes[1, 0]
        equipment_tags = self.data['設備タグ'].unique()
        colors = ['red', 'blue', 'green', 'orange']
        
        for i, tag in enumerate(equipment_tags):
            tag_data = self.data[self.data['設備タグ'] == tag]
            ax3.scatter(tag_data.index, tag_data[target_column], 
                       color=colors[i % len(colors)], alpha=0.6, 
                       label=f'設備{tag}', s=20)
        
        ax3.set_title(f'{target_column} 設備別')
        ax3.set_ylabel(target_column)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 稼働期間別統計
        ax4 = axes[1, 1]
        period_stats = []
        period_labels = []
        
        for i, period in enumerate(self.operation_periods):
            mask = (self.data.index >= period['start']) & (self.data.index <= period['end'])
            period_data = self.data[mask][target_column]
            
            if len(period_data) > 0:
                period_stats.append([
                    period_data.mean(),
                    period_data.std(),
                    period_data.min(),
                    period_data.max()
                ])
                period_labels.append(f"期間{i+1}\n品種{period['product']}")
        
        if period_stats:
            stats_df = pd.DataFrame(period_stats, 
                                  columns=['平均', '標準偏差', '最小', '最大'],
                                  index=period_labels)
            
            stats_df[['平均', '標準偏差']].plot(kind='bar', ax=ax4, alpha=0.7)
            ax4.set_title(f'{target_column} 期間別統計')
            ax4.set_ylabel(target_column)
            ax4.legend()
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. ヒストグラム（品種別）
        ax5 = axes[2, 0]
        for product in products:
            product_data = self.data[self.data['品種'] == product][target_column]
            ax5.hist(product_data, alpha=0.6, bins=20, 
                    color=product_colors.get(product, 'lightgray'),
                    label=f'品種{product}')
        
        ax5.set_title(f'{target_column} 分布（品種別）')
        ax5.set_xlabel(target_column)
        ax5.set_ylabel('頻度')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 相関分析
        ax6 = axes[2, 1]
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['設備タグ']]
        
        if len(numeric_cols) > 1:
            corr_data = self.data[numeric_cols].corr()
            im = ax6.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
            ax6.set_xticks(range(len(numeric_cols)))
            ax6.set_yticks(range(len(numeric_cols)))
            ax6.set_xticklabels(numeric_cols, rotation=45)
            ax6.set_yticklabels(numeric_cols)
            ax6.set_title('変数間相関')
            
            # 相関係数を表示
            for i in range(len(numeric_cols)):
                for j in range(len(numeric_cols)):
                    ax6.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                            ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=ax6)
        
        plt.tight_layout()
        
        # 保存
        if save_file:
            plt.savefig(save_file, dpi=300, bbox_inches='tight')
            print(f"✅ 詳細分析図を保存しました: {save_file}")
        
        plt.show()


def main():
    """メイン実行関数"""
    print("🎨 稼働データ可視化ツール - 品種別背景塗り分け")
    print("=" * 60)
    
    # 可視化クラスのインスタンス作成
    visualizer = OperationDataVisualizer()
    
    # データ読み込み
    data_file = 'extracted_tag_data.csv'
    df = visualizer.load_extracted_data(data_file)
    
    if df is None:
        print("❌ データファイルが見つかりません")
        print("先に operation_data_extractor.py を実行してください")
        return
    
    # 稼働期間情報抽出
    visualizer.extract_operation_periods()
    
    print("\n📊 可視化実行中...")
    
    # 1. 品種別背景塗り分けプロット（連続表示版）
    print("1. 品種別背景塗り分けプロット（連続表示）作成中...")
    visualizer.plot_with_product_background_continuous(
        columns=['temperature', 'pressure', 'flow_rate', 'moisture'],
        save_file='product_background_continuous.png'
    )
    
    # 2. 稼働ガントチャート（改良版）
    print("2. 稼働ガントチャート（改良版）作成中...")
    visualizer.plot_operation_gantt_chart(save_file='operation_gantt_chart.png')
    
    # 3. 従来版（比較用）
    print("3. 従来版プロット（比較用）作成中...")
    visualizer.plot_with_product_background(
        columns=['temperature', 'pressure'],
        save_file='product_background_original.png'
    )
    
    # 4. 詳細分析プロット
    print("4. 詳細分析プロット作成中...")
    visualizer.plot_detailed_analysis(
        target_column='temperature',
        save_file='detailed_analysis.png'
    )
    
    print("\n🎉 可視化完了!")
    print("📁 生成されたファイル:")
    print("- product_background_continuous.png: 品種別背景塗り分け（連続表示）")
    print("- operation_gantt_chart.png: 稼働ガントチャート（改良版）")
    print("- product_background_original.png: 従来版（比較用）")
    print("- detailed_analysis.png: 詳細分析プロット")
    print("\n💡 連続表示版では空白期間が除去され、データがある期間だけが表示されます。")


if __name__ == "__main__":
    main() 