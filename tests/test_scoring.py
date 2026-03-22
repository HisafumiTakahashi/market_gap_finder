"""
スコアリングモジュールのユニットテスト

scoring.py の各関数について、正常系・異常系をテストする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analyze.scoring import (
    compute_demand_score,
    compute_demand_score_v2,
    compute_opportunity_score,
    compute_opportunity_score_v2,
    compute_opportunity_score_v3b,
    generate_reason,
    get_top_recommendations,
    rank_opportunities,
)


# ──────────────────────────────────────
# フィクスチャ
# ──────────────────────────────────────

@pytest.fixture()
def base_df() -> pd.DataFrame:
    """基本テスト用 DataFrame（4行）。"""
    return pd.DataFrame(
        {
            "mesh_code": ["m001", "m002", "m003", "m004"],
            "unified_genre": ["cafe", "izakaya", "ramen", "washoku"],
            "restaurant_count": [10, 30, 5, 50],
            "lat": [35.66, 35.69, 35.73, 35.67],
            "lng": [139.70, 139.71, 139.72, 139.76],
        }
    )


@pytest.fixture()
def base_df_v2() -> pd.DataFrame:
    """v2 テスト用 DataFrame（人口あり）。"""
    return pd.DataFrame(
        {
            "mesh_code": ["m001", "m002", "m003", "m004"],
            "unified_genre": ["cafe", "izakaya", "ramen", "washoku"],
            "restaurant_count": [10, 30, 5, 50],
            "population": [50000, 80000, 30000, 20000],
            "lat": [35.66, 35.69, 35.73, 35.67],
            "lng": [139.70, 139.71, 139.72, 139.76],
        }
    )


@pytest.fixture()
def empty_df() -> pd.DataFrame:
    return pd.DataFrame()


# ──────────────────────────────────────
# compute_demand_score
# ──────────────────────────────────────

class TestComputeDemandScore:
    def test_returns_series(self, base_df: pd.DataFrame) -> None:
        result = compute_demand_score(base_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(base_df)

    def test_values_between_0_and_1(self, base_df: pd.DataFrame) -> None:
        result = compute_demand_score(base_df)
        assert result.between(0.0, 1.0).all(), f"範囲外の値: {result.tolist()}"

    def test_name_is_demand_score(self, base_df: pd.DataFrame) -> None:
        result = compute_demand_score(base_df)
        assert result.name == "demand_score"

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = compute_demand_score(empty_df)
        assert result.empty

    def test_all_same_returns_zeros(self) -> None:
        df = pd.DataFrame(
            {"mesh_code": ["m1", "m1"], "restaurant_count": [5, 5]}
        )
        result = compute_demand_score(df)
        assert (result == 0.0).all()


# ──────────────────────────────────────
# compute_demand_score_v2
# ──────────────────────────────────────

class TestComputeDemandScoreV2:
    def test_uses_population(self, base_df_v2: pd.DataFrame) -> None:
        result = compute_demand_score_v2(base_df_v2)
        assert isinstance(result, pd.Series)
        # 人口80000が最高 → demand_score=1.0
        assert result.iloc[1] == pytest.approx(1.0)

    def test_fallback_to_restaurant_count(self, base_df: pd.DataFrame) -> None:
        """population がない場合は restaurant_count にフォールバック。"""
        result = compute_demand_score_v2(base_df)
        assert len(result) == len(base_df)

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = compute_demand_score_v2(empty_df)
        assert result.empty


# ──────────────────────────────────────
# compute_opportunity_score
# ──────────────────────────────────────

class TestComputeOpportunityScore:
    def test_score_column_exists(self, base_df: pd.DataFrame) -> None:
        result = compute_opportunity_score(base_df)
        assert "opportunity_score" in result.columns

    def test_intermediate_columns_exist(self, base_df: pd.DataFrame) -> None:
        result = compute_opportunity_score(base_df)
        for col in ("demand_score", "competitor_density"):
            assert col in result.columns, f"{col} が存在しない"

    def test_scores_between_0_and_1(self, base_df: pd.DataFrame) -> None:
        result = compute_opportunity_score(base_df)
        opp = result["opportunity_score"]
        assert opp.between(0.0, 1.0).all(), f"範囲外: {opp.tolist()}"

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = compute_opportunity_score(empty_df)
        assert result.empty

    def test_original_df_not_modified(self, base_df: pd.DataFrame) -> None:
        original_cols = set(base_df.columns)
        compute_opportunity_score(base_df)
        assert set(base_df.columns) == original_cols


# ──────────────────────────────────────
# compute_opportunity_score_v2
# ──────────────────────────────────────

class TestComputeOpportunityScoreV2:
    def test_score_column_exists(self, base_df_v2: pd.DataFrame) -> None:
        result = compute_opportunity_score_v2(base_df_v2)
        assert "opportunity_score" in result.columns

    def test_scores_between_0_and_1(self, base_df_v2: pd.DataFrame) -> None:
        result = compute_opportunity_score_v2(base_df_v2)
        opp = result["opportunity_score"]
        assert opp.between(0.0, 1.0).all()

    def test_high_population_low_competitor_scores_high(self) -> None:
        df = pd.DataFrame(
            {
                "mesh_code": ["m1", "m2"],
                "unified_genre": ["cafe", "cafe"],
                "restaurant_count": [1, 50],
                "population": [80000, 10000],
            }
        )
        result = compute_opportunity_score_v2(df)
        assert result["opportunity_score"].iloc[0] > result["opportunity_score"].iloc[1]

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = compute_opportunity_score_v2(empty_df)
        assert result.empty


class TestComputeOpportunityScoreV3B:
    def test_v3b_basic(self) -> None:
        df = pd.DataFrame(
            {
                "mesh_code": ["m1", "m2", "m3"],
                "unified_genre": ["cafe", "cafe", "ramen"],
                "restaurant_count": [8, 12, 5],
                "other_genre_count": [40, 15, 20],
                "genre_hhi": [0.3, 0.5, 0.4],
                "google_avg_rating": [4.2, 3.8, 4.0],
                "genre_diversity": [6, 4, 5],
                "nearest_station_distance": [0.2, 0.7, 0.5],
                "saturation_index": [0.3, 0.8, 0.4],
                "neighbor_avg_restaurants": [5, 12, 7],
                "land_price": [100, 130, 110],
            }
        )
        result = compute_opportunity_score_v3b(df)
        assert result["opportunity_score"].between(0.0, 1.0).all()

    def test_v3b_genre_baseline(self) -> None:
        df = pd.DataFrame(
            {
                "mesh_code": ["m1", "m2", "m3", "m4"],
                "unified_genre": ["cafe", "cafe", "cafe", "ramen"],
                "restaurant_count": [5, 10, 20, 1],
                "other_genre_count": [30, 30, 30, 0],
                "genre_hhi": [0.4, 0.4, 0.4, 0.1],
                "google_avg_rating": [4.0, 4.0, 4.0, 3.0],
                "genre_diversity": [5, 5, 5, 1],
                "nearest_station_distance": [0.3, 0.3, 0.3, 2.0],
                "saturation_index": [0.5, 0.5, 0.5, 0.9],
                "neighbor_avg_restaurants": [8, 8, 8, 20],
                "land_price": [100, 100, 100, 200],
            }
        )
        result = compute_opportunity_score_v3b(df)
        # cafe median is 10, so restaurant_count=5 has positive deficit and 20 does not.
        assert result.loc[0, "opportunity_score"] > result.loc[2, "opportunity_score"]

    def test_v3b_missing_columns(self) -> None:
        df = pd.DataFrame(
            {
                "mesh_code": ["m1", "m2"],
                "unified_genre": ["cafe", "ramen"],
                "restaurant_count": [5, 10],
            }
        )
        result = compute_opportunity_score_v3b(df)
        assert "opportunity_score" in result.columns
        assert result["opportunity_score"].between(0.0, 1.0).all()

    def test_v3b_empty_df(self, empty_df: pd.DataFrame) -> None:
        result = compute_opportunity_score_v3b(empty_df)
        assert result.empty


# ──────────────────────────────────────
# rank_opportunities
# ──────────────────────────────────────

class TestRankOpportunities:
    def test_returns_top_n(self, base_df: pd.DataFrame) -> None:
        scored = compute_opportunity_score(base_df)
        result = rank_opportunities(scored, top_n=2)
        assert len(result) == 2

    def test_sorted_descending(self, base_df: pd.DataFrame) -> None:
        scored = compute_opportunity_score(base_df)
        result = rank_opportunities(scored, top_n=len(base_df))
        scores = result["opportunity_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_top_n_zero_returns_empty(self, base_df: pd.DataFrame) -> None:
        scored = compute_opportunity_score(base_df)
        result = rank_opportunities(scored, top_n=0)
        assert result.empty

    def test_missing_score_column_raises(self, base_df: pd.DataFrame) -> None:
        with pytest.raises(KeyError, match="opportunity_score"):
            rank_opportunities(base_df)

    def test_index_reset(self, base_df: pd.DataFrame) -> None:
        scored = compute_opportunity_score(base_df)
        result = rank_opportunities(scored, top_n=3)
        assert list(result.index) == [0, 1, 2]


# ──────────────────────────────────────
# generate_reason
# ──────────────────────────────────────

class TestGenerateReason:
    def _make_row(self) -> pd.Series:
        return pd.Series(
            {
                "mesh_code": "53394663",
                "unified_genre": "cafe",
                "opportunity_score": 0.85,
                "demand_score": 0.75,
                "competitor_density": 2,
                "restaurant_count": 2,
            }
        )

    def test_returns_string(self) -> None:
        reason = generate_reason(self._make_row())
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_contains_genre(self) -> None:
        reason = generate_reason(self._make_row())
        assert "カフェ" in reason

    def test_contains_restaurant_info(self) -> None:
        reason = generate_reason(self._make_row())
        assert "2" in reason  # restaurant_count

    def test_missing_optional_fields(self) -> None:
        row = pd.Series({"opportunity_score": 0.5})
        reason = generate_reason(row)
        assert isinstance(reason, str)


# ──────────────────────────────────────
# get_top_recommendations
# ──────────────────────────────────────

class TestGetTopRecommendations:
    def test_returns_top_20_by_default(self, base_df: pd.DataFrame) -> None:
        scored = compute_opportunity_score(base_df)
        result = get_top_recommendations(scored)
        assert len(result) <= 20

    def test_reason_column_exists(self, base_df: pd.DataFrame) -> None:
        scored = compute_opportunity_score(base_df)
        result = get_top_recommendations(scored, top_n=2)
        assert "reason" in result.columns

    def test_reason_is_string(self, base_df: pd.DataFrame) -> None:
        scored = compute_opportunity_score(base_df)
        result = get_top_recommendations(scored, top_n=2)
        assert result["reason"].apply(lambda x: isinstance(x, str)).all()

    def test_empty_input_returns_empty(self, empty_df: pd.DataFrame) -> None:
        scored = compute_opportunity_score(empty_df)
        result = get_top_recommendations(scored)
        assert result.empty
