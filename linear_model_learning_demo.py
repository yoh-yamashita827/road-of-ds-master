#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
線形モデル学習過程可視化デモ

線形回帰の学習プロセス（勾配降下法）を段階的に可視化します。
"""

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from matplotlib.animation import FuncAnimation
import pandas as pd
import warnings
import os
warnings.filterwarnings('ignore')

class LinearModelDemo:
    """線形モデル学習過程のデモクラス"""
    
    def __init__(self):
        self.history = []
        self.X = None
        self.y = None
        self.w = None
        self.b = None
        
    def generate_sample_data(self, n_samples=50, noise=0.5, seed=42):
        """サンプルデータ生成"""
        np.random.seed(seed)
        
        # 真の関係: y = 2x + 1 + noise
        self.X = np.random.uniform(-2, 2, n_samples)
        self.y = 2 * self.X + 1 + np.random.normal(0, noise, n_samples)
        
        # パラメータの初期化（適当な値から開始）
        self.w = np.random.uniform(-1, 1)  # 重み
        self.b = np.random.uniform(-1, 1)  # バイアス
        
        print(f"真の関係: y = 2x + 1")
        print(f"初期パラメータ: w={self.w:.3f}, b={self.b:.3f}")
        
    def predict(self, X, w=None, b=None):
        """予測"""
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        return w * X + b
    
    def compute_loss(self, w=None, b=None):
        """平均二乗誤差を計算"""
        if w is None:
            w = self.w
        if b is None:
            b = self.b
        
        y_pred = self.predict(self.X, w, b)
        loss = np.mean((self.y - y_pred) ** 2)
        return loss
    
    def compute_gradients(self):
        """勾配を計算"""
        y_pred = self.predict(self.X)
        error = y_pred - self.y
        
        dw = np.mean(2 * error * self.X)  # ∂L/∂w
        db = np.mean(2 * error)           # ∂L/∂b
        
        return dw, db
    
    def train_step(self, learning_rate=0.01):
        """1ステップの学習"""
        # 勾配計算
        dw, db = self.compute_gradients()
        
        # パラメータ更新
        self.w -= learning_rate * dw
        self.b -= learning_rate * db
        
        # 履歴保存
        loss = self.compute_loss()
        self.history.append({
            'w': self.w,
            'b': self.b,
            'loss': loss,
            'dw': dw,
            'db': db
        })
        
        return loss
    
    def train(self, epochs=100, learning_rate=0.01):
        """学習実行"""
        self.history = []
        
        for epoch in range(epochs):
            loss = self.train_step(learning_rate)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Loss={loss:.4f}, w={self.w:.3f}, b={self.b:.3f}")
        
        print(f"最終: w={self.w:.3f}, b={self.b:.3f} (真値: w=2.0, b=1.0)")

def plot_learning_process_comprehensive():
    """包括的な学習過程の可視化"""
    model = LinearModelDemo()
    model.generate_sample_data(n_samples=50, noise=0.3)
    
    # 学習実行
    model.train(epochs=100, learning_rate=0.1)
    
    # 大きな図を作成
    fig = plt.figure(figsize=(20, 15))
    
    # 1. データと学習結果
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(model.X, model.y, alpha=0.6, color='blue', label='実データ')
    
    # 真の関数
    x_line = np.linspace(-2, 2, 100)
    y_true = 2 * x_line + 1
    ax1.plot(x_line, y_true, 'g--', linewidth=2, label='真の関係 (y=2x+1)')
    
    # 学習された関数
    y_learned = model.predict(x_line)
    ax1.plot(x_line, y_learned, 'r-', linewidth=2, label=f'学習結果 (y={model.w:.2f}x+{model.b:.2f})')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('1. データと学習結果')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 損失関数の変化
    ax2 = plt.subplot(3, 3, 2)
    losses = [h['loss'] for h in model.history]
    ax2.plot(losses, 'b-', linewidth=2)
    ax2.set_xlabel('エポック')
    ax2.set_ylabel('損失 (MSE)')
    ax2.set_title('2. 損失関数の変化')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. パラメータの変化
    ax3 = plt.subplot(3, 3, 3)
    weights = [h['w'] for h in model.history]
    biases = [h['b'] for h in model.history]
    
    ax3.plot(weights, 'r-', linewidth=2, label='重み (w)')
    ax3.plot(biases, 'b-', linewidth=2, label='バイアス (b)')
    ax3.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='真の重み (2.0)')
    ax3.axhline(y=1, color='b', linestyle='--', alpha=0.7, label='真のバイアス (1.0)')
    ax3.set_xlabel('エポック')
    ax3.set_ylabel('パラメータ値')
    ax3.set_title('3. パラメータの収束')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 勾配の変化
    ax4 = plt.subplot(3, 3, 4)
    dws = [h['dw'] for h in model.history]
    dbs = [h['db'] for h in model.history]
    
    ax4.plot(dws, 'r-', linewidth=2, label='∂L/∂w')
    ax4.plot(dbs, 'b-', linewidth=2, label='∂L/∂b')
    ax4.set_xlabel('エポック')
    ax4.set_ylabel('勾配')
    ax4.set_title('4. 勾配の変化')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 損失関数の3D表面
    ax5 = plt.subplot(3, 3, 5, projection='3d')
    
    # パラメータ空間でのグリッド
    w_range = np.linspace(-1, 4, 50)
    b_range = np.linspace(-2, 3, 50)
    W, B = np.meshgrid(w_range, b_range)
    
    # 各点での損失を計算
    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Z[i, j] = model.compute_loss(W[i, j], B[i, j])
    
    ax5.plot_surface(W, B, Z, alpha=0.6, cmap='viridis')
    
    # 学習経路をプロット
    w_path = [h['w'] for h in model.history]
    b_path = [h['b'] for h in model.history]
    loss_path = [h['loss'] for h in model.history]
    
    ax5.plot(w_path, b_path, loss_path, 'r-', linewidth=3, label='学習経路')
    ax5.scatter([w_path[0]], [b_path[0]], [loss_path[0]], color='red', s=100, label='開始点')
    ax5.scatter([w_path[-1]], [b_path[-1]], [loss_path[-1]], color='green', s=100, label='終了点')
    
    ax5.set_xlabel('重み (w)')
    ax5.set_ylabel('バイアス (b)')
    ax5.set_zlabel('損失')
    ax5.set_title('5. 損失関数の表面と学習経路')
    
    # 6. 予測の改善過程
    ax6 = plt.subplot(3, 3, 6)
    ax6.scatter(model.X, model.y, alpha=0.6, color='blue', label='実データ')
    
    # 学習の各段階での予測線を表示
    epochs_to_show = [0, 10, 30, 60, 99]
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    
    for i, epoch in enumerate(epochs_to_show):
        if epoch < len(model.history):
            w_epoch = model.history[epoch]['w']
            b_epoch = model.history[epoch]['b']
            y_pred_epoch = w_epoch * x_line + b_epoch
            ax6.plot(x_line, y_pred_epoch, color=colors[i], linewidth=2, 
                    alpha=0.7, label=f'エポック {epoch}')
    
    ax6.plot(x_line, y_true, 'k--', linewidth=2, label='真の関係')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('6. 予測の改善過程')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 学習率の影響
    ax7 = plt.subplot(3, 3, 7)
    
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    colors_lr = ['blue', 'green', 'orange', 'red']
    
    for lr, color in zip(learning_rates, colors_lr):
        model_lr = LinearModelDemo()
        model_lr.generate_sample_data(n_samples=50, noise=0.3, seed=42)
        model_lr.train(epochs=100, learning_rate=lr)
        
        losses_lr = [h['loss'] for h in model_lr.history]
        ax7.plot(losses_lr, color=color, linewidth=2, label=f'学習率 {lr}')
    
    ax7.set_xlabel('エポック')
    ax7.set_ylabel('損失')
    ax7.set_title('7. 学習率の影響')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_yscale('log')
    
    # 8. 残差の分析
    ax8 = plt.subplot(3, 3, 8)
    
    y_pred_final = model.predict(model.X)
    residuals = model.y - y_pred_final
    
    ax8.scatter(y_pred_final, residuals, alpha=0.6)
    ax8.axhline(y=0, color='red', linestyle='--')
    ax8.set_xlabel('予測値')
    ax8.set_ylabel('残差')
    ax8.set_title('8. 残差プロット')
    ax8.grid(True, alpha=0.3)
    
    # 9. 学習の数式説明
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    formula_text = """
    線形回帰の学習過程
    
    1. モデル: ŷ = wx + b
    
    2. 損失関数: L = 1/n Σ(y - ŷ)²
    
    3. 勾配計算:
       ∂L/∂w = 2/n Σ(ŷ - y)x
       ∂L/∂b = 2/n Σ(ŷ - y)
    
    4. パラメータ更新:
       w ← w - α(∂L/∂w)
       b ← b - α(∂L/∂b)
    
    α: 学習率
    """
    
    ax9.text(0.1, 0.9, formula_text, transform=ax9.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return model

def plot_gradient_descent_animation():
    """勾配降下法のアニメーション"""
    print("📹 勾配降下法アニメーション作成中...")
    
    model = LinearModelDemo()
    model.generate_sample_data(n_samples=30, noise=0.2)
    
    # 学習実行
    model.train(epochs=50, learning_rate=0.1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    def animate(frame):
        ax1.clear()
        ax2.clear()
        
        if frame < len(model.history):
            # 現在のパラメータ
            current_w = model.history[frame]['w']
            current_b = model.history[frame]['b']
            current_loss = model.history[frame]['loss']
            
            # 左側: データと現在の予測線
            ax1.scatter(model.X, model.y, alpha=0.6, color='blue', label='データ')
            x_line = np.linspace(-2, 2, 100)
            y_true = 2 * x_line + 1
            y_current = current_w * x_line + current_b
            
            ax1.plot(x_line, y_true, 'g--', linewidth=2, label='真の関係')
            ax1.plot(x_line, y_current, 'r-', linewidth=2, 
                    label=f'現在: y={current_w:.2f}x+{current_b:.2f}')
            
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title(f'エポック {frame}: 損失 = {current_loss:.4f}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_xlim(-2, 2)
            ax1.set_ylim(-4, 6)
            
            # 右側: 損失の変化
            losses = [h['loss'] for h in model.history[:frame+1]]
            ax2.plot(range(len(losses)), losses, 'b-', linewidth=2)
            ax2.scatter([frame], [current_loss], color='red', s=100, zorder=5)
            ax2.set_xlabel('エポック')
            ax2.set_ylabel('損失')
            ax2.set_title('損失の変化')
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(0, len(model.history))
            ax2.set_ylim(0, max([h['loss'] for h in model.history]) * 1.1)
    
    # アニメーション作成（表示のみ）
    frames = min(50, len(model.history))
    for frame in range(0, frames, 5):  # 5フレームごとに表示
        animate(frame)
        plt.pause(0.5)
    
    # 最終フレーム
    animate(len(model.history) - 1)
    plt.show()

def plot_different_scenarios():
    """様々なシナリオでの学習"""
    print("📊 様々なシナリオでの学習比較")
    
    scenarios = [
        {'noise': 0.1, 'title': 'ノイズ小'},
        {'noise': 0.5, 'title': 'ノイズ中'},
        {'noise': 1.0, 'title': 'ノイズ大'},
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, scenario in enumerate(scenarios):
        model = LinearModelDemo()
        model.generate_sample_data(n_samples=50, noise=scenario['noise'], seed=42)
        model.train(epochs=100, learning_rate=0.1)
        
        # データと結果
        ax1 = axes[0, i]
        ax1.scatter(model.X, model.y, alpha=0.6, color='blue')
        x_line = np.linspace(-2, 2, 100)
        y_true = 2 * x_line + 1
        y_learned = model.predict(x_line)
        
        ax1.plot(x_line, y_true, 'g--', linewidth=2, label='真の関係')
        ax1.plot(x_line, y_learned, 'r-', linewidth=2, label='学習結果')
        ax1.set_title(f'{scenario["title"]}\n最終: w={model.w:.2f}, b={model.b:.2f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 損失の変化
        ax2 = axes[1, i]
        losses = [h['loss'] for h in model.history]
        ax2.plot(losses, 'b-', linewidth=2)
        ax2.set_xlabel('エポック')
        ax2.set_ylabel('損失')
        ax2.set_title(f'損失変化 (最終: {losses[-1]:.4f})')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_simple_learning_result():
    """データと学習結果のシンプルな表示"""
    model = LinearModelDemo()
    model.generate_sample_data(n_samples=50, noise=0.3)
    
    # 学習実行
    model.train(epochs=100, learning_rate=0.1)
    
    # imagesディレクトリを作成
    os.makedirs('images', exist_ok=True)
    
    # シンプルな図を作成
    plt.figure(figsize=(10, 8))
    
    # データと学習結果の表示
    plt.scatter(model.X, model.y, alpha=0.7, color='blue', s=60, label='実データ', zorder=3)
    
    # 学習された関数のみ
    x_line = np.linspace(-2, 2, 100)
    y_learned = model.predict(x_line)
    plt.plot(x_line, y_learned, 'r-', linewidth=3, label='学習結果', zorder=2)
    
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title('ソフトセンサーの学習イメージ', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 画像を保存
    filename = 'images/soft_sensor_learning.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📁 画像を保存しました: {filename}")
    
    plt.show()
    
    return model

def main():
    """メイン実行関数"""
    print("🎓 線形モデル学習結果表示")
    print("=" * 40)
    
    model = plot_simple_learning_result()
    
    print("\n🎉 学習完了!")
    print(f"📊 結果: w={model.w:.3f}, b={model.b:.3f}")
    print(f"📉 最終損失: {model.history[-1]['loss']:.4f}")

if __name__ == "__main__":
    main() 