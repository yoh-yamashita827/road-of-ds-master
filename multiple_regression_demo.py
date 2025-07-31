#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重回帰モデル学習過程可視化デモ

複数の説明変数による重回帰の学習プロセスと係数の重要度を可視化します。
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
import pandas as pd
import warnings
import os
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings('ignore')

class MultipleRegressionDemo:
    """重回帰モデル学習過程のデモクラス"""
    
    def __init__(self):
        self.history = []
        self.X = None
        self.y = None
        self.coefficients = None
        self.intercept = None
        self.feature_names = None
        
    def generate_sample_data(self, n_samples=100, noise=0.3, seed=42):
        """工場データをイメージしたサンプルデータ生成"""
        np.random.seed(seed)
        
        # 説明変数（工場の要因）
        self.feature_names = ['温度', '圧力', '流量', '触媒濃度']
        
        # 真の関係: y = 2.0*温度 + 1.5*圧力 + 0.8*流量 + 3.0*触媒濃度 + 10.0 + noise
        self.true_coefficients = [2.0, 1.5, 0.8, 3.0]
        self.true_intercept = 10.0
        
        # 特徴量生成（標準化済み）
        self.X = np.random.normal(0, 1, (n_samples, 4))
        
        # 目的変数計算
        self.y = (self.X @ self.true_coefficients + self.true_intercept + 
                 np.random.normal(0, noise, n_samples))
        
        # パラメータの初期化
        self.coefficients = np.random.uniform(-1, 1, 4)
        self.intercept = np.random.uniform(-1, 1)
        
        print(f"真の係数: {dict(zip(self.feature_names, self.true_coefficients))}")
        print(f"真の切片: {self.true_intercept}")
        print(f"初期係数: {dict(zip(self.feature_names, self.coefficients))}")
        print(f"初期切片: {self.intercept:.3f}")
        
    def predict(self, X=None, coefficients=None, intercept=None):
        """予測"""
        if X is None:
            X = self.X
        if coefficients is None:
            coefficients = self.coefficients
        if intercept is None:
            intercept = self.intercept
        return X @ coefficients + intercept
    
    def compute_loss(self, coefficients=None, intercept=None):
        """平均二乗誤差を計算"""
        if coefficients is None:
            coefficients = self.coefficients
        if intercept is None:
            intercept = self.intercept
        
        y_pred = self.predict(self.X, coefficients, intercept)
        loss = np.mean((self.y - y_pred) ** 2)
        return loss
    
    def compute_gradients(self):
        """勾配を計算"""
        y_pred = self.predict()
        error = y_pred - self.y
        n_samples = len(self.y)
        
        # 各係数の勾配
        dw = (2/n_samples) * (self.X.T @ error)
        # 切片の勾配
        db = (2/n_samples) * np.sum(error)
        
        return dw, db
    
    def train_step(self, learning_rate=0.01):
        """1ステップの学習"""
        # 勾配計算
        dw, db = self.compute_gradients()
        
        # パラメータ更新
        self.coefficients -= learning_rate * dw
        self.intercept -= learning_rate * db
        
        # 履歴保存
        loss = self.compute_loss()
        self.history.append({
            'coefficients': self.coefficients.copy(),
            'intercept': self.intercept,
            'loss': loss,
            'dw': dw.copy(),
            'db': db
        })
        
        return loss
    
    def train(self, epochs=200, learning_rate=0.01):
        """学習実行"""
        self.history = []
        
        for epoch in range(epochs):
            loss = self.train_step(learning_rate)
            
            if epoch % 40 == 0:
                print(f"Epoch {epoch:3d}: Loss={loss:.4f}")
        
        print(f"最終係数: {dict(zip(self.feature_names, self.coefficients))}")
        print(f"最終切片: {self.intercept:.3f}")

def plot_multiple_regression_comprehensive():
    """重回帰の包括的な可視化"""
    model = MultipleRegressionDemo()
    model.generate_sample_data(n_samples=100, noise=0.5)
    
    # 学習実行
    model.train(epochs=200, learning_rate=0.1)
    
    # imagesディレクトリを作成
    os.makedirs('images', exist_ok=True)
    
    # 大きな図を作成
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 予測 vs 実測値
    ax1 = plt.subplot(2, 3, 1)
    y_pred = model.predict()
    ax1.scatter(model.y, y_pred, alpha=0.6, color='blue')
    
    # 完全予測線
    min_val = min(model.y.min(), y_pred.min())
    max_val = max(model.y.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    ax1.set_xlabel('実測値', fontsize=12)
    ax1.set_ylabel('予測値', fontsize=12)
    ax1.set_title('1. 予測精度', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # R²を計算して表示
    ss_res = np.sum((model.y - y_pred) ** 2)
    ss_tot = np.sum((model.y - np.mean(model.y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 2. 係数の重要度（最終結果）
    ax2 = plt.subplot(2, 3, 2)
    colors = ['red', 'orange', 'green', 'purple']
    bars = ax2.bar(model.feature_names, model.coefficients, color=colors, alpha=0.7)
    
    # 真の係数を点線で表示
    for i, (name, true_coef) in enumerate(zip(model.feature_names, model.true_coefficients)):
        ax2.axhline(y=true_coef, xmin=i/4-0.1, xmax=i/4+0.1, 
                   color='black', linestyle='--', linewidth=2)
    
    ax2.set_ylabel('係数の値', fontsize=12)
    ax2.set_title('2. 各要因の影響度（係数）', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # 係数の値をバーの上に表示
    for bar, coef in zip(bars, model.coefficients):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{coef:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. 損失関数の変化
    ax3 = plt.subplot(2, 3, 3)
    losses = [h['loss'] for h in model.history]
    ax3.plot(losses, 'b-', linewidth=2)
    ax3.set_xlabel('エポック', fontsize=12)
    ax3.set_ylabel('損失 (MSE)', fontsize=12)
    ax3.set_title('3. 学習の進行', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. 係数の収束過程
    ax4 = plt.subplot(2, 3, 4)
    for i, (name, color) in enumerate(zip(model.feature_names, colors)):
        coef_history = [h['coefficients'][i] for h in model.history]
        ax4.plot(coef_history, color=color, linewidth=2, label=name)
        ax4.axhline(y=model.true_coefficients[i], color=color, 
                   linestyle='--', alpha=0.7)
    
    ax4.set_xlabel('エポック', fontsize=12)
    ax4.set_ylabel('係数の値', fontsize=12)
    ax4.set_title('4. 係数の学習過程', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 残差分析
    ax5 = plt.subplot(2, 3, 5)
    residuals = model.y - y_pred
    ax5.scatter(y_pred, residuals, alpha=0.6, color='green')
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.set_xlabel('予測値', fontsize=12)
    ax5.set_ylabel('残差', fontsize=12)
    ax5.set_title('5. 残差分析', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. 数式の説明
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 学習された式を表示
    equation_parts = []
    for name, coef in zip(model.feature_names, model.coefficients):
        if coef >= 0:
            equation_parts.append(f"+ {coef:.2f}×{name}")
        else:
            equation_parts.append(f"- {abs(coef):.2f}×{name}")
    
    equation = f"予測値 = {model.intercept:.2f} " + " ".join(equation_parts)
    
    formula_text = f"""
重回帰モデルの学習結果

{equation}

各係数の意味:
• 大きい係数 → その要因の影響が大きい
• 正の係数 → 要因が増えると予測値も増加
• 負の係数 → 要因が増えると予測値は減少

R² = {r2:.3f} (1.0に近いほど精度が高い)
    """
    
    ax6.text(0.05, 0.95, formula_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    
    # 画像を保存
    filename = 'images/multiple_regression_comprehensive.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📁 包括的な重回帰分析図を保存しました: {filename}")
    
    plt.show()
    
    return model

def plot_simple_multiple_regression():
    """シンプルな重回帰結果表示"""
    model = MultipleRegressionDemo()
    model.generate_sample_data(n_samples=100, noise=0.5)
    
    # 学習実行
    model.train(epochs=200, learning_rate=0.1)
    
    # imagesディレクトリを作成
    os.makedirs('images', exist_ok=True)
    
    # 図を作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左側: 予測 vs 実測値
    y_pred = model.predict()
    ax1.scatter(model.y, y_pred, alpha=0.7, color='blue', s=60)
    
    # 完全予測線
    min_val = min(model.y.min(), y_pred.min())
    max_val = max(model.y.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r-', linewidth=3)
    
    ax1.set_xlabel('実測値', fontsize=14)
    ax1.set_ylabel('予測値', fontsize=14)
    ax1.set_title('予測精度', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # R²を計算して表示
    ss_res = np.sum((model.y - y_pred) ** 2)
    ss_tot = np.sum((model.y - np.mean(model.y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 右側: 係数の重要度
    colors = ['red', 'orange', 'green', 'purple']
    bars = ax2.bar(model.feature_names, np.abs(model.coefficients), 
                   color=colors, alpha=0.7)
    
    ax2.set_ylabel('係数の絶対値', fontsize=14)
    ax2.set_title('各要因の影響度', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='x', rotation=45)
    
    # 係数の値をバーの上に表示
    for bar, coef in zip(bars, model.coefficients):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{coef:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 画像を保存
    filename = 'images/multiple_regression_simple.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📁 シンプルな重回帰分析図を保存しました: {filename}")
    
    plt.show()
    
    return model

def main():
    """メイン実行関数"""
    print("🎓 重回帰モデル学習デモ")
    print("=" * 50)
    
    print("\n1. 包括的な重回帰分析")
    model1 = plot_multiple_regression_comprehensive()
    
    print("\n2. シンプルな重回帰結果")
    model2 = plot_simple_multiple_regression()
    
    print("\n🎉 デモ完了!")
    print("\n📋 重回帰の特徴:")
    print("- 複数の要因（説明変数）から予測")
    print("- 係数の大きさで各要因の影響度がわかる")
    print("- R²で予測精度を評価")
    print("- 実際の工場データに近い設定")

if __name__ == "__main__":
    main() 