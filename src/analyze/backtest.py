"""
バックテストモジュール

過去の出店データを用いてスコアリングモデルの妥当性を検証する。

- スコア上位エリアに実際に出店した店舗の存続率・売上推移を分析
- スコアと実績指標の相関を評価
"""

from __future__ import annotations

import logging

import pandas as pd

from src.analyze.scoring import compute_opportunity_score
from src.preprocess.cleaner import assign_mesh_code, map_genre

logger = logging.getLogger(__name__)


def load_historical_openings(filepath: str) -> pd.DataFrame:
    """過去の出店実績データを読み込む。

    Parameters
    ----------
    filepath : str
        出店実績 CSV のパス

    Returns
    -------
    pd.DataFrame
        出店実績データ（店舗名, 開店日, ジャンル, 緯度, 経度, 存続フラグ 等）
    """
    df = pd.read_csv(filepath)

    date_col = next(
        (c for c in ("opening_date", "open_date", "date", "opened_at") if c in df.columns),
        None,
    )
    if date_col is None:
        raise KeyError("opening_date / open_date / date / opened_at のいずれかが必要です")

    out = df.copy()
    out["opening_date"] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=["opening_date"]).reset_index(drop=True)

    # lat/lng カラム名の正規化
    if "lat" not in out.columns and "latitude" in out.columns:
        out["lat"] = out["latitude"]
    if "lng" not in out.columns:
        for candidate in ("longitude", "lon"):
            if candidate in out.columns:
                out["lng"] = out[candidate]
                break

    out = map_genre(out)
    if "mesh_code" not in out.columns and {"lat", "lng"}.issubset(out.columns):
        out = assign_mesh_code(out)

    return out


def simulate_scoring_at_date(
    target_date: str,
    df_aggregated: pd.DataFrame,
) -> pd.DataFrame:
    """指定時点のデータでスコアリングを再現する。

    Parameters
    ----------
    target_date : str
        バックテスト対象日 (YYYY-MM-DD)
    df_aggregated : pd.DataFrame
        その時点の集約データ

    Returns
    -------
    pd.DataFrame
        スコア付きデータ
    """
    target_ts = pd.to_datetime(target_date)
    out = df_aggregated.copy()

    if "snapshot_date" in out.columns:
        out["snapshot_date"] = pd.to_datetime(out["snapshot_date"], errors="coerce")
        out = out[out["snapshot_date"] <= target_ts].copy()

    if out.empty:
        return out

    scored = compute_opportunity_score(out)
    scored["target_date"] = target_ts.normalize()
    return scored


def evaluate_accuracy(
    scored_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
) -> dict[str, float]:
    """スコアと実績を突き合わせて精度を評価する。

    Parameters
    ----------
    scored_df : pd.DataFrame
        バックテスト時点でのスコアリング結果
    actuals_df : pd.DataFrame
        実績データ（存続率、売上等）

    Returns
    -------
    dict[str, float]
        評価指標の辞書 (correlation, precision_at_k, etc.)
    """
    if "opportunity_score" not in scored_df.columns:
        raise KeyError("scored_df に opportunity_score カラムが必要です")

    scored = scored_df.copy()
    if "unified_genre" not in scored.columns:
        scored = map_genre(scored)
    if "mesh_code" not in scored.columns and {"lat", "lng"}.issubset(scored.columns):
        scored = assign_mesh_code(scored)

    actuals = actuals_df.copy()
    if "unified_genre" not in actuals.columns:
        actuals = map_genre(actuals)
    if "mesh_code" not in actuals.columns and {"lat", "lng"}.issubset(actuals.columns):
        actuals = assign_mesh_code(actuals)

    if "mesh_code" not in scored.columns or "mesh_code" not in actuals.columns:
        raise KeyError("scored_df / actuals_df の両方に mesh_code（または lat/lng）が必要です")

    actual_counts = (
        actuals.groupby(["mesh_code", "unified_genre"], dropna=False)
        .size()
        .rename("actual_openings")
        .reset_index()
    )

    merged = scored.merge(actual_counts, on=["mesh_code", "unified_genre"], how="left")
    merged["actual_openings"] = merged["actual_openings"].fillna(0.0)

    if merged.empty:
        return {"correlation": 0.0, "precision_at_k": 0.0, "recall_at_k": 0.0,
                "hit_count": 0.0, "k": 0.0}

    score = pd.to_numeric(merged["opportunity_score"], errors="coerce").fillna(0.0)
    actual = pd.to_numeric(merged["actual_openings"], errors="coerce").fillna(0.0)

    correlation = float(score.corr(actual)) if score.nunique() > 1 and actual.nunique() > 1 else 0.0

    k = min(20, len(merged))
    top = merged.nlargest(k, "opportunity_score")
    hits = int((top["actual_openings"] > 0).sum())
    precision_at_k = float(hits / k) if k > 0 else 0.0
    positives = int((merged["actual_openings"] > 0).sum())
    recall_at_k = float(hits / positives) if positives > 0 else 0.0

    return {
        "correlation": correlation,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "hit_count": float(hits),
        "k": float(k),
    }


def run_backtest(
    historical_filepath: str,
    test_dates: list[str] | None = None,
) -> pd.DataFrame:
    """バックテストを一括実行する。

    Parameters
    ----------
    historical_filepath : str
        出店実績データのパス
    test_dates : list[str] | None, optional
        評価対象日リスト（None の場合は自動設定）

    Returns
    -------
    pd.DataFrame
        各テスト日ごとの評価結果
    """
    historical = load_historical_openings(historical_filepath)
    if historical.empty:
        return pd.DataFrame(
            columns=["test_date", "correlation", "precision_at_k", "recall_at_k",
                     "hit_count", "k", "train_rows", "scored_rows"]
        )

    if test_dates is None:
        unique_dates = sorted(pd.to_datetime(historical["opening_date"]).dt.normalize().unique())
        test_dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in unique_dates]

    results: list[dict] = []

    for date_str in test_dates:
        target_ts = pd.to_datetime(date_str)
        train = historical[historical["opening_date"] < target_ts].copy()
        if train.empty:
            continue

        train["_review_count"] = pd.to_numeric(
            train.get("review_count", train.get("user_ratings_total", pd.Series(0, index=train.index))),
            errors="coerce"
        ).fillna(0.0)
        train["_rating"] = pd.to_numeric(
            train.get("rating", train.get("avg_rating", pd.Series(0, index=train.index))),
            errors="coerce"
        ).fillna(0.0)

        for col in ("lat", "lng"):
            if col not in train.columns:
                train[col] = pd.NA

        aggregated = (
            train.groupby(["mesh_code", "unified_genre"], dropna=False)
            .agg(
                restaurant_count=("mesh_code", "size"),
                review_count=("_review_count", "sum"),
                avg_rating=("_rating", "mean"),
                lat=("lat", "mean"),
                lng=("lng", "mean"),
            )
            .reset_index()
        )

        scored = simulate_scoring_at_date(target_ts.strftime("%Y-%m-%d"), aggregated)

        window_end = target_ts + pd.Timedelta(days=30)
        actuals = historical[
            (historical["opening_date"] >= target_ts) & (historical["opening_date"] < window_end)
        ].copy()

        metrics = evaluate_accuracy(scored_df=scored, actuals_df=actuals)
        metrics["test_date"] = target_ts.strftime("%Y-%m-%d")
        metrics["train_rows"] = float(len(train))
        metrics["scored_rows"] = float(len(scored))
        results.append(metrics)

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values("test_date").reset_index(drop=True)
    return result_df
