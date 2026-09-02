#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""大粒径モデル: 粒径変化点の120分前付近におけるプロセスタグ挙動の分析。

前提
----
- 入力DataFrameは、目的変数の各測定点に対して120分前付近の説明変数が
  すでに行方向に整列された状態を想定する。
- そのため時刻を120分shiftするのではなく、変化点の行から直前N行を取得する。
- 粒径変化が大きいケースと安定ケースを比較し、どのタグの状態・変動が異なるかを見る。

主な出力
--------
1. 目的変数の変化量(target_diff)
2. 大変動点 / 安定点の抽出
3. 各変化点直前window_rows行についてタグごとの mean/std/range/change
4. 大変動 vs 安定の比較表
5. target_diff と各タグchangeの相関

列名は実データに合わせて target_col / feature_cols を指定してください。
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd


def add_target_change(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """目的変数の前回測定点からの変化量を追加する。"""
    out = df.copy()
    out["target_diff"] = out[target_col].diff()
    out["abs_target_diff"] = out["target_diff"].abs()
    return out


def split_change_points(
    df: pd.DataFrame,
    upper_quantile: float = 0.75,
    lower_quantile: float = 0.25,
) -> Tuple[pd.Index, pd.Index]:
    """粒径の大変動点と安定点を分位点で抽出する。"""
    valid = df["abs_target_diff"].dropna()
    high_threshold = valid.quantile(upper_quantile)
    low_threshold = valid.quantile(lower_quantile)

    large_points = df.index[df["abs_target_diff"] >= high_threshold]
    stable_points = df.index[df["abs_target_diff"] <= low_threshold]

    print(f"large threshold : {high_threshold:.4f}")
    print(f"stable threshold: {low_threshold:.4f}")
    print(f"large points    : {len(large_points)}")
    print(f"stable points   : {len(stable_points)}")
    return large_points, stable_points


def summarize_preceding_windows(
    df: pd.DataFrame,
    points: Iterable,
    feature_cols: Iterable[str],
    window_rows: int = 30,
) -> pd.DataFrame:
    """各変化点の直前N行についてタグ挙動を要約する。

    入力dfが120分前付近に整列済みであることを前提とする。
    point自身を含め、そこから上方向にwindow_rows行を取得する。
    """
    feature_cols = list(feature_cols)
    rows = []

    for point in points:
        loc = df.index.get_loc(point)
        if not isinstance(loc, (int, np.integer)):
            raise ValueError("indexは一意である必要があります。")

        start = max(0, loc - window_rows + 1)
        window = df.iloc[start : loc + 1]

        row = {"changepoint": point}
        if "target_diff" in df.columns:
            row["target_diff"] = df.iloc[loc]["target_diff"]
        if "abs_target_diff" in df.columns:
            row["abs_target_diff"] = df.iloc[loc]["abs_target_diff"]

        for col in feature_cols:
            if col not in df.columns:
                continue
            s = window[col].dropna()
            if s.empty:
                continue

            row[f"{col}_mean"] = s.mean()
            row[f"{col}_std"] = s.std()
            row[f"{col}_range"] = s.max() - s.min()
            row[f"{col}_change"] = (
                s.iloc[-1] - s.iloc[0] if len(s) >= 2 else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows).set_index("changepoint")


def compare_large_and_stable(
    large_summary: pd.DataFrame,
    stable_summary: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.DataFrame:
    """大変動時と安定時のタグ挙動を比較する。

    mean/std/rangeは平均値を比較し、changeは絶対値平均を比較する。
    ratio > 1 なら、大変動時の方がその統計量が大きい。
    """
    rows = []

    for col in feature_cols:
        for stat in ("mean", "std", "range", "change"):
            name = f"{col}_{stat}"
            if name not in large_summary or name not in stable_summary:
                continue

            large = large_summary[name]
            stable = stable_summary[name]
            if stat == "change":
                large = large.abs()
                stable = stable.abs()

            large_value = large.mean()
            stable_value = stable.mean()
            ratio = large_value / stable_value if stable_value != 0 else np.nan

            rows.append(
                {
                    "feature": col,
                    "stat": stat,
                    "large": large_value,
                    "stable": stable_value,
                    "ratio": ratio,
                }
            )

    return pd.DataFrame(rows).sort_values("ratio", ascending=False)


def change_correlations(
    summary: pd.DataFrame,
    feature_cols: Iterable[str],
) -> pd.Series:
    """目的変数の変化方向と各タグの30行内変化方向の相関を見る。"""
    result = {}
    for col in feature_cols:
        change_col = f"{col}_change"
        if change_col in summary.columns:
            result[col] = summary[["target_diff", change_col]].corr().iloc[0, 1]
    return pd.Series(result, name="corr_with_target_diff").sort_values(ascending=False)


def run_analysis(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: Iterable[str],
    window_rows: int = 30,
) -> dict:
    """一連の大粒径変化点分析を実行する。"""
    work = add_target_change(df, target_col)
    large_points, stable_points = split_change_points(work)

    large_summary = summarize_preceding_windows(
        work, large_points, feature_cols, window_rows
    )
    stable_summary = summarize_preceding_windows(
        work, stable_points, feature_cols, window_rows
    )

    comparison = compare_large_and_stable(
        large_summary, stable_summary, feature_cols
    )

    all_points = work.index[work["target_diff"].notna()]
    all_summary = summarize_preceding_windows(
        work, all_points, feature_cols, window_rows
    )
    correlations = change_correlations(all_summary, feature_cols)

    return {
        "data": work,
        "large_points": large_points,
        "stable_points": stable_points,
        "large_summary": large_summary,
        "stable_summary": stable_summary,
        "comparison": comparison,
        "change_correlations": correlations,
    }


if __name__ == "__main__":
    # 使用例:
    # df = pd.read_csv("your_aligned_large_particle_data.csv", index_col=0)
    #
    # result = run_analysis(
    #     df,
    #     target_col="target",
    #     feature_cols=[
    #         "LPG",
    #         "O2_FLOW",
    #         "O2_RECYCLE",
    #         "STEAM1",
    #         "UNREACTED",
    #         "DP",
    #     ],
    #     window_rows=30,
    # )
    #
    # print(result["comparison"])
    # print(result["change_correlations"])
    pass
