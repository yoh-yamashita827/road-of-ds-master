#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稼働表に基づくタグデータ抽出・結合ツール

稼働表の稼働時間に対応するタグデータを抽出し、
品種タグと設備タグを付加します。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import japanize_matplotlib
import warnings
warnings.filterwarnings('ignore')

class OperationDataExtractor:
    """稼働表に基づくタグデータ抽出クラス"""
    
    def __init__(self):
        self.operation_schedule = None
        self.tag_data = None
        self.extracted_data = None
        
    def load_operation_schedule(self, excel_file_path, sheet_name=None):
        """
        稼働表（Excel）を読み込み
        
        Parameters:
        -----------
        excel_file_path : str
            稼働表のExcelファイルパス
        sheet_name : str, optional
            シート名（指定しない場合は最初のシート）
        """
        try:
            # Excelファイル読み込み
            if sheet_name:
                df_raw = pd.read_excel(excel_file_path, sheet_name=sheet_name)
            else:
                df_raw = pd.read_excel(excel_file_path)
            
            print(f"✅ 稼働表読み込み完了: {df_raw.shape}")
            print(f"列名: {df_raw.columns.tolist()}")
            
            # データ構造を整理
            # 1行目はヘッダーなので、2行目以降を取得
            df = df_raw.iloc[1:].copy()
            
            # 列名を適切に設定
            df.columns = ['日付', '品種', 'タグ', '稼働開始時間', '区切り', '翌日フラグ', '稼働終了時間', '稼働時間合計', '備考']
            
            # インデックスをリセット
            df = df.reset_index(drop=True)
            
            print("\n整理後のデータ:")
            print(df)
            
            self.operation_schedule = df
            return df
            
        except Exception as e:
            print(f"❌ 稼働表読み込みエラー: {e}")
            return None
    
    def load_tag_data(self, csv_file_path):
        """
        タグデータ（CSV）を読み込み
        
        Parameters:
        -----------
        csv_file_path : str
            タグデータのCSVファイルパス
        """
        try:
            # CSVファイル読み込み
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            
            # タイムスタンプ列を自動検出
            timestamp_cols = []
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['time', 'timestamp', '時刻', '時間', 'date']):
                    timestamp_cols.append(col)
            
            if timestamp_cols:
                timestamp_col = timestamp_cols[0]
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df = df.set_index(timestamp_col)
                print(f"✅ タイムスタンプ列として {timestamp_col} を設定")
            
            print(f"✅ タグデータ読み込み完了: {df.shape}")
            print(f"期間: {df.index.min()} ～ {df.index.max()}")
            print(f"列名: {df.columns.tolist()}")
            
            self.tag_data = df
            return df
            
        except Exception as e:
            print(f"❌ タグデータ読み込みエラー: {e}")
            return None
    
    def parse_operation_times(self, start_time, end_time, next_day_flag):
        """
        稼働時間をパース
        
        Parameters:
        -----------
        start_time : datetime.time
            稼働開始時間
        end_time : datetime.time
            稼働終了時間
        next_day_flag : str
            翌日フラグ（'翌'など）
        
        Returns:
        --------
        tuple : (start_time_str, end_time_str) またはNone
        """
        try:
            if pd.isna(start_time) or pd.isna(end_time):
                return None
                
            # 開始時間
            if hasattr(start_time, 'hour'):
                start_time_str = f"{start_time.hour:02d}:{start_time.minute:02d}:00"
            else:
                return None
            
            # 終了時間
            if hasattr(end_time, 'hour'):
                end_time_str = f"{end_time.hour:02d}:{end_time.minute:02d}:00"
            else:
                return None
            
            # 翌日フラグがある場合の処理
            is_next_day = str(next_day_flag) == '翌'
            
            return (start_time_str, end_time_str, is_next_day)
            
        except Exception as e:
            print(f"⚠️ 時間パースエラー: {start_time}, {end_time} - {e}")
            return None
    
    def extract_tag_data_by_schedule(self):
        """
        稼働表に基づいてタグデータを抽出
        """
        if self.operation_schedule is None:
            print("❌ 稼働表が読み込まれていません")
            return None
            
        if self.tag_data is None:
            print("❌ タグデータが読み込まれていません")
            return None
        
        extracted_records = []
        
        print("🔄 稼働表に基づくデータ抽出開始...")
        
        for idx, row in self.operation_schedule.iterrows():
            try:
                # 各列の値を取得
                date_val = row['日付']
                product = row['品種']
                tag_value = row['タグ']
                start_time = row['稼働開始時間']
                end_time = row['稼働終了時間']
                next_day_flag = row['翌日フラグ']
                
                print(f"\n処理中: {date_val} - {product} - タグ{tag_value}")
                print(f"時間: {start_time} ～ {end_time} (翌日: {next_day_flag})")
                
                # 時間情報をパース
                time_info = self.parse_operation_times(start_time, end_time, next_day_flag)
                if time_info is None:
                    print(f"⚠️ 時間情報の解析に失敗")
                    continue
                
                start_time_str, end_time_str, is_next_day = time_info
                
                # 日付の処理
                if pd.isna(date_val):
                    print(f"⚠️ 日付情報がありません")
                    continue
                
                # 日付をdatetimeに変換
                if hasattr(date_val, 'date'):
                    date_part = date_val.date()
                else:
                    date_part = pd.to_datetime(date_val).date()
                
                # 開始・終了日時の作成
                start_datetime = pd.to_datetime(f"{date_part} {start_time_str}")
                
                if is_next_day:
                    # 翌日終了の場合
                    end_date = date_part + timedelta(days=1)
                    end_datetime = pd.to_datetime(f"{end_date} {end_time_str}")
                else:
                    # 同日終了の場合
                    end_datetime = pd.to_datetime(f"{date_part} {end_time_str}")
                
                print(f"抽出期間: {start_datetime} ～ {end_datetime}")
                
                # タグデータから該当時間帯を抽出
                mask = (self.tag_data.index >= start_datetime) & (self.tag_data.index <= end_datetime)
                period_data = self.tag_data[mask].copy()
                
                if len(period_data) > 0:
                    # 品種タグと設備タグを追加
                    period_data['品種'] = product
                    period_data['設備タグ'] = tag_value
                    period_data['稼働開始時刻'] = start_datetime
                    period_data['稼働終了時刻'] = end_datetime
                    period_data['稼働日'] = date_part
                    
                    extracted_records.append(period_data)
                    print(f"✅ 抽出完了: {len(period_data)}件")
                else:
                    print(f"⚠️ 該当時間帯にデータなし: {start_datetime} - {end_datetime}")
                
            except Exception as e:
                print(f"❌ 行処理エラー: {idx} - {e}")
                continue
        
        # 全ての抽出データを結合
        if extracted_records:
            self.extracted_data = pd.concat(extracted_records, ignore_index=False)
            print(f"\n🎉 データ抽出完了: {len(self.extracted_data)}件")
            return self.extracted_data
        else:
            print("❌ 抽出されたデータがありません")
            return None
    
    def analyze_extracted_data(self):
        """抽出されたデータの分析"""
        if self.extracted_data is None:
            print("❌ 抽出データがありません")
            return
        
        print("=" * 60)
        print("📊 抽出データ分析結果")
        print("=" * 60)
        
        # 基本統計
        print(f"総データ件数: {len(self.extracted_data):,}")
        print(f"期間: {self.extracted_data.index.min()} ～ {self.extracted_data.index.max()}")
        
        # 品種別集計
        print("\n■ 品種別データ件数")
        product_counts = self.extracted_data['品種'].value_counts()
        print(product_counts)
        
        # 設備タグ別集計
        print("\n■ 設備タグ別データ件数")
        tag_counts = self.extracted_data['設備タグ'].value_counts()
        print(tag_counts)
        
        # 稼働日別集計
        print("\n■ 稼働日別データ件数")
        day_counts = self.extracted_data['稼働日'].value_counts().sort_index()
        print(day_counts)
        
        # 数値列の統計
        numeric_cols = self.extracted_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['設備タグ']]
        if len(numeric_cols) > 0:
            print("\n■ 数値データ統計")
            print(self.extracted_data[numeric_cols].describe().round(3))
    
    def visualize_extracted_data(self):
        """抽出されたデータの可視化"""
        if self.extracted_data is None:
            print("❌ 抽出データがありません")
            return
        
        # 数値列を取得
        numeric_cols = self.extracted_data.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['設備タグ']]
        
        if len(numeric_cols) == 0:
            print("⚠️ 可視化可能な数値データがありません")
            return
        
        # 品種別時系列プロット
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        # 品種別に色分け
        products = self.extracted_data['品種'].unique()
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, col in enumerate(numeric_cols[:4]):
            ax = axes[i]
            
            for j, product in enumerate(products):
                product_data = self.extracted_data[self.extracted_data['品種'] == product]
                ax.scatter(product_data.index, product_data[col], 
                          label=f'品種{product}', color=colors[j % len(colors)], alpha=0.7)
            
            ax.set_title(f'{col} - 品種別')
            ax.set_ylabel(col)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 設備タグ別ボックスプロット
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(1, min(2, len(numeric_cols)), figsize=(12, 6))
            if len(numeric_cols) == 1:
                axes = [axes]
            
            for i, col in enumerate(numeric_cols[:2]):
                self.extracted_data.boxplot(column=col, by='設備タグ', ax=axes[i])
                axes[i].set_title(f'{col} - 設備タグ別分布')
                axes[i].set_xlabel('設備タグ')
            
            plt.tight_layout()
            plt.show()
    
    def save_extracted_data(self, output_file='extracted_tag_data.csv'):
        """抽出されたデータを保存"""
        if self.extracted_data is None:
            print("❌ 保存するデータがありません")
            return
        
        try:
            self.extracted_data.to_csv(output_file, encoding='utf-8', index=True)
            print(f"✅ データ保存完了: {output_file}")
            print(f"保存件数: {len(self.extracted_data):,}件")
        except Exception as e:
            print(f"❌ 保存エラー: {e}")


def main():
    """メイン実行関数"""
    print("🏭 稼働表に基づくタグデータ抽出ツール")
    print("=" * 60)
    
    # インスタンス作成
    extractor = OperationDataExtractor()
    
    # 稼働表読み込み
    print("📂 稼働表読み込み中...")
    operation_df = extractor.load_operation_schedule('test.xlsx')
    
    if operation_df is None:
        print("❌ 稼働表の読み込みに失敗しました")
        return
    
    # タグデータ読み込み（ファイル名は適宜変更）
    print("\n📂 タグデータ読み込み中...")
    # tag_df = extractor.load_tag_data('tag_data.csv')  # 実際のファイル名に変更
    
    # サンプルデータ作成（実際のデータがない場合）
    print("⚠️ サンプルタグデータを作成します...")
    sample_data = create_sample_tag_data()
    extractor.tag_data = sample_data
    
    # データ抽出実行
    print("\n🔄 データ抽出実行...")
    extracted_data = extractor.extract_tag_data_by_schedule()
    
    if extracted_data is not None:
        # 分析実行
        extractor.analyze_extracted_data()
        
        # 可視化
        extractor.visualize_extracted_data()
        
        # 保存
        extractor.save_extracted_data()
    
    print("\n🎉 処理完了!")


def create_sample_tag_data():
    """サンプルタグデータ作成"""
    print("📊 サンプルタグデータ作成中...")
    
    # 2025年6月1日-3日の5分間隔データ（実際の稼働表の日付に合わせる）
    start_date = '2025-06-01 00:00:00'
    end_date = '2025-06-04 23:59:59'
    
    timestamps = pd.date_range(start=start_date, end=end_date, freq='5min')
    
    # サンプルタグデータ
    np.random.seed(42)
    data = {
        'temperature': np.random.normal(80, 10, len(timestamps)),
        'pressure': np.random.normal(1.5, 0.2, len(timestamps)),
        'flow_rate': np.random.normal(100, 15, len(timestamps)),
        'moisture': np.random.normal(12, 2, len(timestamps))
    }
    
    df = pd.DataFrame(data, index=timestamps)
    print(f"✅ サンプルデータ作成完了: {df.shape}")
    
    return df


if __name__ == "__main__":
    main() 