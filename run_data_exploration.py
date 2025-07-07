#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工場データ包括的探索分析スクリプト

このスクリプトは工場のタグデータに対して包括的な探索分析を実行し、
データの特徴的な点を片っ端から探り出すツールです。

使用方法:
1. データファイルのパスを設定
2. 必要に応じて日時列名を設定
3. スクリプトを実行

出力:
- 基本的なデータ理解のための分析結果
- 高度な洞察分析結果
- ヒアリング推奨項目のリスト
- 発見事項の総合レポート
"""

import pandas as pd
import numpy as np
import warnings
from data_exploration import FactoryDataExplorer
from advanced_data_insights import AdvancedFactoryDataInsights
import os
import sys

warnings.filterwarnings('ignore')

def main():
    """メイン分析実行関数"""
    print("🏭 工場データ包括的探索分析システム")
    print("=" * 60)
    
    # =================================
    # 設定部分（ここを変更してください）
    # =================================
    
    # データファイルのパス
    data_path = "factory_data.csv"  # 実際のファイル名に変更
    
    # 日時列名（自動検出も可能）
    datetime_column = None  # 例: "timestamp", "datetime", "time" など
    
    # =================================
    # 分析実行部分
    # =================================
    
    try:
        # ファイル存在確認
        if not os.path.exists(data_path):
            print(f"❌ エラー: ファイル '{data_path}' が見つかりません")
            print("\n📝 設定確認:")
            print(f"   - 現在のディレクトリ: {os.getcwd()}")
            print(f"   - 指定されたファイル: {data_path}")
            print("\n💡 解決方法:")
            print("   1. 正しいファイルパスを run_data_exploration.py の data_path に設定してください")
            print("   2. ファイルが現在のディレクトリにあることを確認してください")
            return
        
        print(f"📂 データファイル: {data_path}")
        print(f"📅 日時列: {datetime_column if datetime_column else '自動検出'}")
        print()
        
        # ステップ1: 基本探索分析
        print("🔍 ステップ1: 基本データ探索分析を開始...")
        print("-" * 50)
        
        explorer = FactoryDataExplorer(data_path)
        basic_findings = explorer.run_comprehensive_analysis()
        
        print(f"\n✅ 基本分析完了: {len(basic_findings)}個の発見事項")
        
        # ステップ2: 高度洞察分析
        print("\n" + "=" * 60)
        print("🔬 ステップ2: 高度洞察分析を開始...")
        print("-" * 50)
        
        # 日時列の設定
        if datetime_column and datetime_column in explorer.data.columns:
            explorer.data[datetime_column] = pd.to_datetime(explorer.data[datetime_column])
        
        advanced_analyzer = AdvancedFactoryDataInsights(explorer.data, datetime_column)
        advanced_insights = advanced_analyzer.run_advanced_analysis()
        
        print(f"\n✅ 高度分析完了: {len(advanced_insights)}個の洞察")
        
        # ステップ3: 統合レポート生成
        print("\n" + "=" * 80)
        print("📊 ステップ3: 統合分析レポート")
        print("=" * 80)
        
        generate_integrated_report(basic_findings, advanced_insights, explorer, advanced_analyzer)
        
        # ステップ4: 次のステップガイド
        print("\n" + "=" * 80)
        print("🚀 次のステップ・推奨アクション")
        print("=" * 80)
        
        provide_next_steps(basic_findings, advanced_insights)
        
        print("\n✨ 包括的データ探索分析が完了しました！")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        print("\n🔧 トラブルシューティング:")
        print("1. データファイルの形式がCSVであることを確認")
        print("2. ファイルに適切な権限があることを確認")
        print("3. 必要なライブラリがインストールされていることを確認")
        print("   pip install -r requirements.txt")

def generate_integrated_report(basic_findings, advanced_insights, explorer, advanced_analyzer):
    """統合分析レポートの生成"""
    
    total_findings = len(basic_findings) + len(advanced_insights)
    
    print(f"📈 総合分析結果サマリー:")
    print(f"   データ行数: {len(explorer.data):,}行")
    print(f"   データ列数: {len(explorer.data.columns)}列")
    print(f"   数値列: {len(explorer.data.select_dtypes(include=[np.number]).columns)}個")
    print(f"   基本分析発見事項: {len(basic_findings)}件")
    print(f"   高度分析洞察: {len(advanced_insights)}件")
    print(f"   合計発見事項: {total_findings}件")
    
    # 重要度別の分類
    priority_classification = classify_findings_by_priority(basic_findings + advanced_insights)
    
    print(f"\n🎯 重要度別分類:")
    for priority, items in priority_classification.items():
        if items:
            print(f"   {priority}: {len(items)}件")
    
    # データ品質評価
    data_quality_score = calculate_data_quality_score(explorer, basic_findings)
    print(f"\n🎭 データ品質スコア: {data_quality_score}/100")
    
    # 分析の信頼性評価
    analysis_confidence = assess_analysis_confidence(explorer, total_findings)
    print(f"🔬 分析信頼性: {analysis_confidence}")

def classify_findings_by_priority(all_findings):
    """発見事項を重要度別に分類"""
    
    high_priority_keywords = ['外れ値', '異常', '欠損', '品質問題', '変化点']
    medium_priority_keywords = ['相関', '周期', '季節', 'トレンド', '分布']
    low_priority_keywords = ['稼働', 'パターン', 'クラスタ']
    
    classification = {
        '🔴 高優先度': [],
        '🟡 中優先度': [],
        '🟢 低優先度': [],
        '⚪ その他': []
    }
    
    for finding in all_findings:
        if any(keyword in finding for keyword in high_priority_keywords):
            classification['🔴 高優先度'].append(finding)
        elif any(keyword in finding for keyword in medium_priority_keywords):
            classification['🟡 中優先度'].append(finding)
        elif any(keyword in finding for keyword in low_priority_keywords):
            classification['🟢 低優先度'].append(finding)
        else:
            classification['⚪ その他'].append(finding)
    
    return classification

def calculate_data_quality_score(explorer, findings):
    """データ品質スコアの計算"""
    score = 100
    
    # 欠損値のペナルティ
    missing_ratio = explorer.data.isnull().sum().sum() / (len(explorer.data) * len(explorer.data.columns))
    score -= missing_ratio * 30
    
    # 重複データのペナルティ
    duplicate_ratio = explorer.data.duplicated().sum() / len(explorer.data)
    score -= duplicate_ratio * 20
    
    # 品質問題のペナルティ
    quality_issues = [f for f in findings if '品質問題' in f]
    score -= len(quality_issues) * 10
    
    # 外れ値のペナルティ
    outlier_issues = [f for f in findings if '外れ値' in f]
    score -= len(outlier_issues) * 5
    
    return max(0, min(100, int(score)))

def assess_analysis_confidence(explorer, total_findings):
    """分析の信頼性評価"""
    data_size = len(explorer.data)
    num_features = len(explorer.data.select_dtypes(include=[np.number]).columns)
    
    if data_size < 100:
        return "低（データ数不足）"
    elif data_size < 1000:
        if num_features < 5:
            return "中（データ・特徴量ともに限定的）"
        else:
            return "中（十分な特徴量）"
    else:
        if num_features < 3:
            return "中（特徴量不足）"
        elif total_findings < 5:
            return "中（発見事項少）"
        else:
            return "高（十分なデータと発見事項）"

def provide_next_steps(basic_findings, advanced_insights):
    """次のステップの提案"""
    
    all_findings = basic_findings + advanced_insights
    
    print("📋 推奨される次のアクション:")
    
    # データ品質改善
    quality_issues = [f for f in all_findings if any(kw in f for kw in ['欠損', '品質', '異常', '外れ値'])]
    if quality_issues:
        print("\n1️⃣ データ品質改善（最優先）:")
        print("   • 欠損値の発生原因をヒアリング")
        print("   • 外れ値の妥当性を確認")
        print("   • データ収集プロセスの見直し")
    
    # 特徴量エンジニアリング
    correlation_findings = [f for f in all_findings if '相関' in f]
    period_findings = [f for f in all_findings if any(kw in f for kw in ['周期', '季節'])]
    
    if correlation_findings or period_findings:
        print("\n2️⃣ 特徴量エンジニアリング:")
        if correlation_findings:
            print("   • 高相関変数の因果関係を調査")
            print("   • 派生変数の作成（比率、差分等）")
        if period_findings:
            print("   • 周期性を活用した時間特徴量の作成")
            print("   • 移動平均等の平滑化特徴量")
    
    # モデル構築準備
    print("\n3️⃣ 予測モデル構築準備:")
    print("   • 30分遅れ特徴量の効果検証")
    print("   • 特徴量選択手法の適用")
    print("   • 時系列分割による適切な検証設計")
    
    # ドメイン知識活用
    print("\n4️⃣ ドメイン知識との統合:")
    print("   • 工程エンジニアとの発見事項共有")
    print("   • 物理的制約の確認")
    print("   • 業界標準との比較")
    
    # 継続的改善
    print("\n5️⃣ 継続的改善:")
    print("   • 定期的なデータ品質モニタリング")
    print("   • 新規データでの再分析")
    print("   • 予測性能の継続評価")

if __name__ == "__main__":
    # コマンドライン引数での設定も可能
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        datetime_col = sys.argv[2] if len(sys.argv) > 2 else None
        
        # 一時的に設定を上書き
        import builtins
        builtins.data_path = data_path
        builtins.datetime_column = datetime_col
    
    main() 