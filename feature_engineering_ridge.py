#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
30分先の水分値予測（噴霧時点での判断を想定）向けの特徴量エンジニアリング

想定:
- 線形モデル（Ridge）で学習
- 時間方向のリーク防止（t時点の説明変数は t 以前のみ）
- 多スケールのラグ/ローリング、比率・交互作用・ラインダミー等

使い方:
  python feature_engineering_ridge.py \
    --input extracted_tag_data.csv \
    --output features_ridge.csv \
    --target_col moisture \
    --horizon_min 30

列名マッピング例（最低限）:
  - 噴霧流量: spray_flow → データ例では 'flow_rate'
  - 噴霧温度: spray_temp → データ例では 'temperature'
  - 加温出口温度: preheat_out_temp → データ例では（未定: 無ければスキップ）
  - 原料温度: feed_temp → データ例では（未定: 無ければスキップ）
  - 乾燥空気流量: air_flow → データ例では（未定: 無ければスキップ）
  - 制御: SV/PV/MV → データ例では（未定: 無ければスキップ）

注意:
- 実データでは上記の物理的に意味のある列にマッピングしてください。
- 列が存在しない場合は自動的にスキップします（警告表示）。
"""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ============================================================
# ユーティリティ
# ============================================================

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """インデックスが日時でなければ自動検出して設定する。

    Returns: インデックスがDatetimeIndexのDataFrame
    """
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # 候補列を探索
    for col in df.columns:
        low = str(col).lower()
        if any(k in low for k in ["time", "timestamp", "date", "時刻", "時間"]):
            try:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                return df
            except Exception:
                continue

    # 最初の列を強制的に日時とみなす（最終手段）
    try:
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
        df = df.set_index(df.columns[0])
    except Exception:
        raise ValueError("日時インデックスを設定できませんでした。日時列を指定してください。")
    return df


def add_line_dummies(df: pd.DataFrame, line_col_candidates: Iterable[str] = ("設備タグ", "line_id")) -> pd.DataFrame:
    """ライン／設備識別のダミー変数を付与（存在すれば）。"""
    out = df.copy()
    for col in line_col_candidates:
        if col in out.columns:
            # 数値ならそのままカテゴリに変換
            dummies = pd.get_dummies(out[col].astype("category"), prefix=str(col))
            out = pd.concat([out, dummies], axis=1)
            break
    return out


def safe_cols(df: pd.DataFrame, cols: Iterable[str]) -> List[str]:
    """存在する列のみ返す。"""
    present = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[warn] 欠落列をスキップ: {missing}")
    return present


def compute_time_lags(df: pd.DataFrame, cols: List[str], lag_minutes: Iterable[int]) -> pd.DataFrame:
    """時間ベースのラグを作成（t以前のみ）。

    注: インデックスはDatetimeIndexであること。
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex が必要です")

    result = []
    base = df[cols]
    for lag in lag_minutes:
        shifted = base.shift(freq=pd.Timedelta(minutes=lag))
        shifted.columns = [f"{c}_lag{lag}m" for c in cols]
        result.append(shifted)
    return pd.concat(result, axis=1)


def compute_time_rolling(df: pd.DataFrame, cols: List[str], windows_min: Iterable[int]) -> pd.DataFrame:
    """時間ベースのローリング統計（平均・標準偏差）。"""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex が必要です")

    out = []
    for w in windows_min:
        roll = df[cols].rolling(f"{w}min", min_periods=max(1, int(w/2)))
        out.append(roll.mean().add_suffix(f"_mean_{w}m"))
        out.append(roll.std().add_suffix(f"_std_{w}m"))
    return pd.concat(out, axis=1)


def compute_interactions(df: pd.DataFrame, pairs: Iterable[Tuple[str, str]]) -> pd.DataFrame:
    """交互作用（積）を作成。"""
    out = {}
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            out[f"{a}__x__{b}"] = df[a] * df[b]
    return pd.DataFrame(out, index=df.index)


def compute_ratios(df: pd.DataFrame, pairs: Iterable[Tuple[str, str]], eps: float = 1e-6) -> pd.DataFrame:
    """比率（a/b）を作成。"""
    out = {}
    for a, b in pairs:
        if a in df.columns and b in df.columns:
            out[f"{a}__div__{b}"] = df[a] / (df[b].replace(0, np.nan) + eps)
    return pd.DataFrame(out, index=df.index)


def hinge_transform(df: pd.DataFrame, cols: Iterable[str], knots: Iterable[float]) -> pd.DataFrame:
    """ヒンジ変換 max(0, x - k)。"""
    out = {}
    for c in cols:
        if c not in df.columns:
            continue
        for k in knots:
            out[f"{c}__hinge_{k}"] = (df[c] - k).clip(lower=0.0)
    return pd.DataFrame(out, index=df.index)


# ============================================================
# 特徴量生成の本体
# ============================================================

def build_feature_matrix(
    df_raw: pd.DataFrame,
    target_col: str,
    horizon_min: int = 30,
    mapping: Optional[Dict[str, str]] = None,
    lag_minutes: Iterable[int] = (0, 3, 10),
    rolling_minutes: Iterable[int] = (3,),
    add_interactions: bool = True,
    add_ratios: bool = True,
    add_hinges: bool = True,
    hinge_knots: Optional[Dict[str, List[float]]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """特徴量行列Xと目的変数yを作成。

    df_raw: インデックスが時刻。列に各種タグ。
    target_col: 目的変数の列名（例: moisture）。
    horizon_min: 予測ホライズン（分）。y(t) = target_col(t + horizon)
    mapping: 物理名→実データ列名の対応辞書
    """
    if mapping is None:
        mapping = {}

    df = ensure_datetime_index(df_raw.copy())
    df = df.sort_index()
    df = add_line_dummies(df)

    # 物理キー（存在すれば使用）
    phys_to_col = {
        # 最小セット（存在しなければスキップ）
        "spray_flow": mapping.get("spray_flow", "flow_rate"),
        "spray_temp": mapping.get("spray_temp", "temperature"),
        "preheat_out_temp": mapping.get("preheat_out_temp", None),
        "feed_temp": mapping.get("feed_temp", None),
        "air_flow": mapping.get("air_flow", None),
        "SV": mapping.get("SV", None),
        "PV": mapping.get("PV", None),
        "MV": mapping.get("MV", None),
    }

    base_cols = [c for c in phys_to_col.values() if c]
    base_cols = safe_cols(df, base_cols)

    # ラグ・ローリング
    lag_df = compute_time_lags(df, base_cols, lag_minutes) if base_cols else pd.DataFrame(index=df.index)
    roll_df = compute_time_rolling(df, base_cols, rolling_minutes) if base_cols else pd.DataFrame(index=df.index)

    # 制御誤差 e = SV - PV（存在すれば）
    control_df = pd.DataFrame(index=df.index)
    if phys_to_col.get("SV") in df.columns and phys_to_col.get("PV") in df.columns:
        control_df["control_error"] = df[phys_to_col["SV"]] - df[phys_to_col["PV"]]
        # そのラグ
        ce_lag = compute_time_lags(control_df, ["control_error"], lag_minutes)
        control_df = pd.concat([control_df, ce_lag], axis=1)

    # 比率
    ratio_df = pd.DataFrame(index=df.index)
    if add_ratios:
        ratio_pairs: List[Tuple[str, str]] = []
        if phys_to_col.get("spray_flow") and phys_to_col.get("air_flow"):
            ratio_pairs.append((phys_to_col["spray_flow"], phys_to_col["air_flow"]))
        if phys_to_col.get("preheat_out_temp") and phys_to_col.get("feed_temp") and phys_to_col.get("air_flow"):
            # 乾燥ポテンシャル的な代理: (温度差)/空気流量
            ratio_pairs.append((phys_to_col["preheat_out_temp"], phys_to_col["feed_temp"]))
        if ratio_pairs:
            ratio_df = compute_ratios(df, ratio_pairs)

    # 交互作用
    inter_df = pd.DataFrame(index=df.index)
    if add_interactions:
        inter_pairs: List[Tuple[str, str]] = []
        if phys_to_col.get("spray_temp") and phys_to_col.get("spray_flow"):
            inter_pairs.append((phys_to_col["spray_temp"], phys_to_col["spray_flow"]))
        if phys_to_col.get("preheat_out_temp") and phys_to_col.get("feed_temp"):
            inter_pairs.append((phys_to_col["preheat_out_temp"], phys_to_col["feed_temp"]))
        if inter_pairs:
            inter_df = compute_interactions(df, inter_pairs)

    # ヒンジ（代表的に噴霧温度・流量にしきい値）
    hinge_df = pd.DataFrame(index=df.index)
    if add_hinges:
        if hinge_knots is None:
            hinge_knots = {}
        hinge_cols: List[str] = []
        for phys_key in ("spray_temp", "spray_flow"):
            colname = phys_to_col.get(phys_key)
            if colname and colname in df.columns:
                hinge_cols.append(colname)
        if hinge_cols:
            # データ分位点から自動ノット（例: 25, 50, 75%）
            auto_knots: Dict[str, List[float]] = {}
            for c in hinge_cols:
                knots = hinge_knots.get(c)
                if not knots:
                    q = df[c].quantile([0.25, 0.5, 0.75]).values.tolist()
                    knots = sorted(set(float(x) for x in q))
                auto_knots[c] = knots
            parts = []
            for c, ks in auto_knots.items():
                parts.append(hinge_transform(df, [c], ks))
            if parts:
                hinge_df = pd.concat(parts, axis=1)

    # ライン/設備ダミー
    line_dummy_cols = [c for c in df.columns if c.startswith("設備タグ_") or c.startswith("line_id_")]
    line_df = df[line_dummy_cols] if line_dummy_cols else pd.DataFrame(index=df.index)

    # 速度項（微分近似）: MVや流量など代表1ステップ差分
    diff_df = pd.DataFrame(index=df.index)
    for col in safe_cols(df, [phys_to_col.get("MV", ""), phys_to_col.get("spray_flow", "")]):
        if col:
            diff_df[f"{col}__diff1"] = df[col] - df[col].shift(1)

    # 特徴量を結合
    feature_parts = [lag_df, roll_df, control_df, ratio_df, inter_df, hinge_df, line_df, diff_df]
    X = pd.concat([p for p in feature_parts if p is not None and p.shape[1] > 0], axis=1)

    # 目的変数（t+horizon）: ラベル整列。学習時点では t の特徴のみを使用。
    if target_col not in df.columns:
        raise ValueError(f"target_col '{target_col}' がデータに存在しません")
    y = df[target_col].shift(freq=pd.Timedelta(minutes=horizon_min))

    # 学習に使える行のみ残す
    valid_mask = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    return X, y


def fit_ridge_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """標準化 + RidgeCV のパイプラインを返す。"""
    alphas = np.logspace(-3, 3, 25)
    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", RidgeCV(alphas=alphas, cv=5, store_cv_values=False))
    ])
    model.fit(X, y)
    return model


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ridge向け特徴量作成と学習")
    p.add_argument("--input", type=str, default="extracted_tag_data.csv", help="入力CSV（抽出済み）")
    p.add_argument("--output", type=str, default="features_ridge.csv", help="特徴量の出力CSV")
    p.add_argument("--target_col", type=str, default="moisture", help="目的変数列名")
    p.add_argument("--horizon_min", type=int, default=30, help="予測ホライズン（分）")
    return p.parse_args()


def main():
    args = parse_args()
    print("🔧 特徴量エンジニアリング実行")
    print(f"入力: {args.input}")
    print(f"出力: {args.output}")
    print(f"目的変数: {args.target_col}, horizon: {args.horizon_min}分")

    df = pd.read_csv(args.input)
    df = ensure_datetime_index(df)

    # 必要であれば列名マッピングをここで指定（実データに合わせて変更）
    mapping = {
        # 'spray_flow': 'your_spray_flow_col',
        # 'spray_temp': 'your_spray_temp_col',
        # 'preheat_out_temp': 'your_preheat_outlet_temp_col',
        # 'feed_temp': 'your_feed_temp_col',
        # 'air_flow': 'your_air_flow_col',
        # 'SV': 'your_sv_col',
        # 'PV': 'your_pv_col',
        # 'MV': 'your_mv_col',
    }

    X, y = build_feature_matrix(
        df_raw=df,
        target_col=args.target_col,
        horizon_min=args.horizon_min,
        mapping=mapping,
        lag_minutes=(0, 3, 10),
        rolling_minutes=(3,),
        add_interactions=True,
        add_ratios=True,
        add_hinges=True,
    )

    # 特徴量を保存
    out_df = X.copy()
    out_df[args.target_col] = y
    out_df.to_csv(args.output, index=True)
    print(f"✅ 特徴量を保存しました: {args.output} (行数: {len(out_df):,}, 列数: {out_df.shape[1]})")

    # 簡易学習（参考）
    try:
        model = fit_ridge_model(X, y)
        ridge = model.named_steps["ridge"]
        print(f"🧠 RidgeCV 学習完了: best_alpha={getattr(ridge, 'alpha_', None)}")
        # 係数トップを参照
        coefs = model.named_steps["ridge"].coef_
        coef_df = pd.Series(coefs, index=X.columns).sort_values(key=lambda s: s.abs(), ascending=False)
        print("重要特徴トップ10:\n" + coef_df.head(10).to_string())
    except Exception as e:
        print(f"[warn] Ridge学習はスキップ: {e}")


if __name__ == "__main__":
    main()


