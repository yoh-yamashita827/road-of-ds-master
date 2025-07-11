#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稼働表に基づくタグデータ抽出・結合ツール（実データ版）

実際のタグデータファイルを使用して、稼働表の稼働時間に対応する
タグデータを抽出し、品種タグと設備タグを付加します。
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import japanize_matplotlib
import warnings
warnings.filterwarnings('ignore')

def extract_tag_data_by_operation_schedule(operation_excel_file, tag_csv_file, output_file='extracted_tag_data.csv'):
    """
    稼働表に基づいてタグデータを抽出するメイン関数
    
    Parameters:
    -----------
    operation_excel_file : str
        稼働表のExcelファイルパス
    tag_csv_file : str
        タグデータのCSVファイルパス
    output_file : str
        出力ファイル名
    """
    print("🏭 稼働表に基づくタグデータ抽出ツール（実データ版）")
    print("=" * 70)
    
    # 1. 稼働表読み込み
    print("📂 稼働表読み込み中...")
    try:
        df_raw = pd.read_excel(operation_excel_file)
        
        # データ構造を整理（1行目はヘッダーなので、2行目以降を取得）
        df_operation = df_raw.iloc[1:].copy()
        df_operation.columns = ['日付', '品種', 'タグ', '稼働開始時間', '区切り', '翌日フラグ', '稼働終了時間', '稼働時間合計', '備考']
        df_operation = df_operation.reset_index(drop=True)
        
        print(f"✅ 稼働表読み込み完了: {len(df_operation)}件")
        print(df_operation)
        
    except Exception as e:
        print(f"❌ 稼働表読み込みエラー: {e}")
        return None
    
    # 2. タグデータ読み込み
    print(f"\n📂 タグデータ読み込み中: {tag_csv_file}")
    try:
        df_tag = pd.read_csv(tag_csv_file, encoding='utf-8')
        
        # タイムスタンプ列を自動検出
        timestamp_col = None
        for col in df_tag.columns:
            if any(keyword in col.lower() for keyword in ['time', 'timestamp', '時刻', '時間', 'date']):
                timestamp_col = col
                break
        
        if timestamp_col:
            df_tag[timestamp_col] = pd.to_datetime(df_tag[timestamp_col])
            df_tag = df_tag.set_index(timestamp_col)
            print(f"✅ タイムスタンプ列として '{timestamp_col}' を設定")
        else:
            print("⚠️ タイムスタンプ列が見つかりません。最初の列を使用します。")
            df_tag.iloc[:, 0] = pd.to_datetime(df_tag.iloc[:, 0])
            df_tag = df_tag.set_index(df_tag.columns[0])
        
        print(f"✅ タグデータ読み込み完了: {df_tag.shape}")
        print(f"期間: {df_tag.index.min()} ～ {df_tag.index.max()}")
        print(f"列名: {df_tag.columns.tolist()}")
        
    except Exception as e:
        print(f"❌ タグデータ読み込みエラー: {e}")
        return None
    
    # 3. データ抽出
    print("\n🔄 稼働表に基づくデータ抽出開始...")
    extracted_records = []
    
    for idx, row in df_operation.iterrows():
        try:
            # 各列の値を取得
            date_val = row['日付']
            product = row['品種']
            tag_value = row['タグ']
            start_time = row['稼働開始時間']
            end_time = row['稼働終了時間']
            next_day_flag = row['翌日フラグ']
            
            print(f"\n処理中: {date_val} - 品種{product} - 設備タグ{tag_value}")
            print(f"時間: {start_time} ～ {end_time} (翌日: {next_day_flag})")
            
            # 時間情報の処理
            if pd.isna(start_time) or pd.isna(end_time):
                print("⚠️ 時間情報が不完全です")
                continue
            
            # 開始・終了時間の文字列化
            if hasattr(start_time, 'hour'):
                start_time_str = f"{start_time.hour:02d}:{start_time.minute:02d}:00"
            else:
                print("⚠️ 開始時間の形式が不正です")
                continue
            
            if hasattr(end_time, 'hour'):
                end_time_str = f"{end_time.hour:02d}:{end_time.minute:02d}:00"
            else:
                print("⚠️ 終了時間の形式が不正です")
                continue
            
            # 日付の処理
            if pd.isna(date_val):
                print("⚠️ 日付情報がありません")
                continue
            
            if hasattr(date_val, 'date'):
                date_part = date_val.date()
            else:
                date_part = pd.to_datetime(date_val).date()
            
            # 開始・終了日時の作成
            start_datetime = pd.to_datetime(f"{date_part} {start_time_str}")
            
            # 翌日フラグの処理
            is_next_day = str(next_day_flag) == '翌'
            if is_next_day:
                end_date = date_part + timedelta(days=1)
                end_datetime = pd.to_datetime(f"{end_date} {end_time_str}")
            else:
                end_datetime = pd.to_datetime(f"{date_part} {end_time_str}")
            
            print(f"抽出期間: {start_datetime} ～ {end_datetime}")
            
            # タグデータから該当時間帯を抽出
            mask = (df_tag.index >= start_datetime) & (df_tag.index <= end_datetime)
            period_data = df_tag[mask].copy()
            
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
                print(f"⚠️ 該当時間帯にデータなし")
            
        except Exception as e:
            print(f"❌ 行処理エラー: {idx} - {e}")
            continue
    
    # 4. 結果の結合と保存
    if extracted_records:
        df_result = pd.concat(extracted_records, ignore_index=False)
        
        print(f"\n🎉 データ抽出完了: {len(df_result)}件")
        print("=" * 50)
        print("📊 抽出データサマリー")
        print("=" * 50)
        print(f"総データ件数: {len(df_result):,}")
        print(f"期間: {df_result.index.min()} ～ {df_result.index.max()}")
        
        # 品種別集計
        print("\n■ 品種別データ件数")
        print(df_result['品種'].value_counts())
        
        # 設備タグ別集計
        print("\n■ 設備タグ別データ件数")
        print(df_result['設備タグ'].value_counts())
        
        # 稼働日別集計
        print("\n■ 稼働日別データ件数")
        print(df_result['稼働日'].value_counts().sort_index())
        
        # ファイル保存
        try:
            df_result.to_csv(output_file, encoding='utf-8', index=True)
            print(f"\n✅ データ保存完了: {output_file}")
            print(f"保存件数: {len(df_result):,}件")
        except Exception as e:
            print(f"❌ 保存エラー: {e}")
        
        return df_result
    else:
        print("❌ 抽出されたデータがありません")
        return None


def main():
    """
    メイン実行関数
    
    使用方法:
    1. 稼働表ファイル名を指定
    2. タグデータファイル名を指定
    3. 出力ファイル名を指定（オプション）
    """
    
    # ファイル名設定
    operation_excel_file = 'test.xlsx'  # 稼働表ファイル
    tag_csv_file = 'tag_data.csv'      # タグデータファイル（実際のファイル名に変更）
    output_file = 'extracted_tag_data.csv'  # 出力ファイル
    
    # データ抽出実行
    result = extract_tag_data_by_operation_schedule(
        operation_excel_file=operation_excel_file,
        tag_csv_file=tag_csv_file,
        output_file=output_file
    )
    
    if result is not None:
        print("\n🎯 次のステップ:")
        print("1. 抽出されたデータを確認")
        print("2. 品種別・設備タグ別の分析実行")
        print("3. 異常値検出や予測モデルの構築")
        print("\n📁 生成されたファイル:")
        print(f"- {output_file}: 抽出されたタグデータ")
    
    print("\n🎉 処理完了!")


if __name__ == "__main__":
    main() 