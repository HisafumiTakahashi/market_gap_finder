"""スコアリング結果のバックテストを行うモジュール。

過去の開店データを学習期間と評価期間に分け、各時点で算出した機会スコアが
実際の開店実績とどの程度整合するかを定量評価する。
"""

from __future__ import annotations

import logging

import pandas as pd

from src.analyze.scoring import compute_opportunity_score
from src.preprocess.cleaner import assign_mesh_code, map_genre

logger = logging.getLogger(__name__)

_DATE_COLUMN_CANDIDATES = ("opening_date", "open_date", "date", "opened_at")
_RATING_COLUMN_CANDIDATES = ("rating", "avg_rating")
_REVIEW_COLUMN_CANDIDATES = ("total_reviews", "review_count", "user_ratings_total", "reviews")


def _find_first_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    """候補列名のうち DataFrame に最初に存在する列名を返す。

    Args:
        df: 対象 DataFrame。
        candidates: 優先順に並べた列名候補。

    Returns:
        見つかった列名。存在しない場合は `None`。
    """
    return next((column for column in candidates if column in df.columns), None)


def _normalize_lat_lng(df: pd.DataFrame) -> pd.DataFrame:
    """緯度経度列名を `lat` / `lng` に正規化する。

    `latitude` や `longitude`、`lon` といった別名の列が存在する場合に、
    後続処理で扱いやすい列名へ統一する。

    Args:
        df: 緯度経度に関する列を含む可能性がある DataFrame。

    Returns:
        列名を正規化した DataFrame のコピー。
    """
    normalized = df.copy()

    if "lat" not in normalized.columns and "latitude" in normalized.columns:
        normalized = normalized.rename(columns={"latitude": "lat"})

    if "lng" not in normalized.columns:
        if "longitude" in normalized.columns:
            normalized = normalized.rename(columns={"longitude": "lng"})
        elif "lon" in normalized.columns:
            normalized = normalized.rename(columns={"lon": "lng"})

    return normalized


def _get_numeric_series(
    df: pd.DataFrame,
    candidates: tuple[str, ...],
    default: float = 0.0,
) -> pd.Series:
    """候補列から数値系列を取得し、なければ既定値系列を返す。

    Args:
        df: 対象 DataFrame。
        candidates: 優先順に並べた列名候補。
        default: 列が存在しない場合や欠損値に用いる既定値。

    Returns:
        数値化済みの系列。
    """
    column = _find_first_column(df, candidates)
    if column is None:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def _aggregate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """学習用の開店履歴をメッシュ・ジャンル単位に集約する。

    評価点数やレビュー数の候補列を吸収しつつ、スコアリング関数が期待する
    入力形式へ変換する。

    Args:
        df: 学習期間に属する開店履歴 DataFrame。

    Returns:
        スコアリング用の集計済み DataFrame。
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "mesh_code",
                "unified_genre",
                "restaurant_count",
                "avg_rating",
                "total_reviews",
                "lat",
                "lng",
            ]
        )

    aggregated_input = df.copy()
    aggregated_input["avg_rating_source"] = _get_numeric_series(
        aggregated_input,
        _RATING_COLUMN_CANDIDATES,
        default=0.0,
    )
    aggregated_input["total_reviews_source"] = _get_numeric_series(
        aggregated_input,
        _REVIEW_COLUMN_CANDIDATES,
        default=0.0,
    )
    aggregated_input["lat"] = pd.to_numeric(aggregated_input["lat"], errors="coerce")
    aggregated_input["lng"] = pd.to_numeric(aggregated_input["lng"], errors="coerce")

    aggregated = (
        aggregated_input.groupby(["mesh_code", "unified_genre"], dropna=False)
        .agg(
            restaurant_count=("mesh_code", "size"),
            avg_rating=("avg_rating_source", "mean"),
            total_reviews=("total_reviews_source", "sum"),
            lat=("lat", "mean"),
            lng=("lng", "mean"),
        )
        .reset_index()
    )

    return aggregated


def load_historical_openings(filepath: str) -> pd.DataFrame:
    """過去の開店履歴 CSV を読み込み、分析用に整形する。

    日付列候補から開店日を特定して日時型へ変換し、無効な日付を持つ行を除外する。
    その後、ジャンル正規化とメッシュコード付与を実施する。

    Args:
        filepath: 開店履歴 CSV のパス。

    Returns:
        前処理済みの開店履歴 DataFrame。

    Raises:
        KeyError: 開店日を表す列が存在しない場合。
    """
    df = pd.read_csv(filepath)
    date_column = _find_first_column(df, _DATE_COLUMN_CANDIDATES)
    if date_column is None:
        raise KeyError("opening_date/open_date/date/opened_at column is required")

    loaded = _normalize_lat_lng(df)
    loaded["opening_date"] = pd.to_datetime(loaded[date_column], errors="coerce")
    loaded = loaded.dropna(subset=["opening_date"]).reset_index(drop=True)

    logger.info("Loaded %s historical rows after dropping invalid dates", len(loaded))

    loaded = map_genre(loaded)
    loaded = assign_mesh_code(loaded)
    return loaded


def simulate_scoring_at_date(target_date: str, df_aggregated: pd.DataFrame) -> pd.DataFrame:
    """指定日時点の情報だけを使って機会スコアを再計算する。

    `snapshot_date` 列が存在する場合は対象日時以前の行に限定し、その状態で
    スコアリングを実行して `target_date` 列を付与する。

    Args:
        target_date: 評価基準日を表す文字列。
        df_aggregated: スコアリング対象の集計済み DataFrame。

    Returns:
        指定日時点でのスコア計算結果。
    """
    target_ts = pd.to_datetime(target_date).normalize()
    scoring_input = df_aggregated.copy()

    if "snapshot_date" in scoring_input.columns:
        scoring_input["snapshot_date"] = pd.to_datetime(
            scoring_input["snapshot_date"],
            errors="coerce",
        )
        scoring_input = scoring_input[scoring_input["snapshot_date"] <= target_ts].copy()

    scored = compute_opportunity_score(scoring_input)
    scored["target_date"] = target_ts
    return scored


def evaluate_accuracy(scored_df: pd.DataFrame, actuals_df: pd.DataFrame) -> dict[str, float]:
    """スコア結果と実績開店数を比較して評価指標を算出する。

    メッシュ・ジャンルごとの実績開店数を結合し、相関係数と上位 K 件に対する
    precision / recall を返す。K は最大 20 件またはスコア対象件数の少ない方。

    Args:
        scored_df: 機会スコア計算済み DataFrame。
        actuals_df: 評価期間の実績開店データ。

    Returns:
        相関係数、precision@k、recall@k、ヒット件数、K を含む辞書。
    """
    if scored_df.empty:
        return {
            "correlation": 0.0,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_count": 0.0,
            "k": 0.0,
        }

    actual_counts = (
        actuals_df.groupby(["mesh_code", "unified_genre"], dropna=False)
        .size()
        .rename("actual_openings")
        .reset_index()
    )

    merged = scored_df.merge(
        actual_counts,
        on=["mesh_code", "unified_genre"],
        how="left",
    )
    merged["actual_openings"] = pd.to_numeric(
        merged["actual_openings"],
        errors="coerce",
    ).fillna(0.0)

    score_series = pd.to_numeric(merged["opportunity_score"], errors="coerce").fillna(0.0)
    actual_series = merged["actual_openings"]

    correlation = 0.0
    if score_series.nunique(dropna=False) > 1 and actual_series.nunique(dropna=False) > 1:
        correlation_value = score_series.corr(actual_series)
        correlation = 0.0 if pd.isna(correlation_value) else float(correlation_value)

    k = min(20, len(scored_df))
    if k == 0:
        return {
            "correlation": correlation,
            "precision_at_k": 0.0,
            "recall_at_k": 0.0,
            "hit_count": 0.0,
            "k": 0.0,
        }

    top_k = merged.nlargest(k, "opportunity_score")
    hit_count = float((top_k["actual_openings"] > 0).sum())
    positive_count = float((merged["actual_openings"] > 0).sum())

    precision_at_k = hit_count / float(k)
    recall_at_k = hit_count / positive_count if positive_count > 0 else 0.0

    return {
        "correlation": correlation,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "hit_count": hit_count,
        "k": float(k),
    }


def run_backtest(
    historical_filepath: str,
    test_dates: list[str] | None = None,
) -> pd.DataFrame:
    """複数の評価日でバックテストを実行し、結果を表形式で返す。

    各評価日について、過去データだけを学習用に集計し、その日から 30 日間の
    開店実績を正解データとして評価指標を計算する。

    Args:
        historical_filepath: 開店履歴 CSV のパス。
        test_dates: 評価日文字列の一覧。`None` の場合は履歴に含まれる日付を使用する。

    Returns:
        各評価日の指標をまとめた DataFrame。
    """
    historical = load_historical_openings(historical_filepath)
    result_columns = [
        "test_date",
        "correlation",
        "precision_at_k",
        "recall_at_k",
        "hit_count",
        "k",
        "train_rows",
        "scored_rows",
    ]
    if historical.empty:
        return pd.DataFrame(columns=result_columns)

    if test_dates is None:
        normalized_dates = pd.to_datetime(historical["opening_date"], errors="coerce").dt.normalize()
        test_dates = [timestamp.strftime("%Y-%m-%d") for timestamp in sorted(normalized_dates.dropna().unique())]

    results: list[dict[str, float | str | int]] = []

    for test_date in test_dates:
        target_ts = pd.to_datetime(test_date).normalize()
        train_df = historical[historical["opening_date"] < target_ts].copy()

        aggregated_train = _aggregate_training_data(train_df)
        scored_df = simulate_scoring_at_date(
            target_date=target_ts.strftime("%Y-%m-%d"),
            df_aggregated=aggregated_train,
        )

        window_end = target_ts + pd.Timedelta(days=30)
        actuals_df = historical[
            (historical["opening_date"] >= target_ts) & (historical["opening_date"] < window_end)
        ].copy()

        metrics = evaluate_accuracy(scored_df=scored_df, actuals_df=actuals_df)
        metrics["test_date"] = target_ts.strftime("%Y-%m-%d")
        metrics["train_rows"] = int(len(train_df))
        metrics["scored_rows"] = int(len(scored_df))
        results.append(metrics)

        logger.info(
            "Backtest %s: train_rows=%s scored_rows=%s actual_rows=%s",
            target_ts.strftime("%Y-%m-%d"),
            len(train_df),
            len(scored_df),
            len(actuals_df),
        )

    if not results:
        return pd.DataFrame(columns=result_columns)

    return pd.DataFrame(results, columns=result_columns)
