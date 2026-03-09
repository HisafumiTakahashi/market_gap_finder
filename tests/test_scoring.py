"""
スコアリングモジュールのユニットテスト

scoring.py の各関数について、正常系・異常系をテストする。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analyze.scoring import (
    compute_competitor_density,
    compute_demand_score,
    compute_opportunity_score,
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
            "area": ["渋谷", "新宿", "池袋", "銀座"],
            "genre": ["カフェ"] * 4,
            "review_count": [500, 200, 50, 800],
            "avg_rating": [4.5, 4.0, 3.5, 4.8],
            "congestion_index": [1.2, 1.0, 0.8, 1.5],
            "restaurant_count": [10, 30, 5, 50],
            "population_10k": [5.0, 8.0, 3.0, 2.0],
            "population_density": [18000, 22000, 12000, 10000],
            "land_price": [3000, 2500, 1500, 5000],
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

    def test_high_reviews_high_rating_scores_higher(self) -> None:
        """口コミ数・評価が高い行がより高いスコアになること。"""
        df = pd.DataFrame(
            {
                "review_count": [1000, 10],
                "avg_rating": [4.9, 3.0],
                "congestion_index": [1.5, 0.5],
            }
        )
        result = compute_demand_score(df)
        assert result.iloc[0] > result.iloc[1]

    def test_alias_columns_accepted(self) -> None:
        """別名カラム（user_ratings_total, rating）でも動作すること。"""
        df = pd.DataFrame(
            {
                "user_ratings_total": [100, 200],
                "rating": [4.0, 4.5],
            }
        )
        result = compute_demand_score(df)
        assert len(result) == 2
        assert result.between(0.0, 1.0).all()

    def test_all_zero_returns_zeros(self) -> None:
        df = pd.DataFrame({"review_count": [0, 0], "avg_rating": [0.0, 0.0]})
        result = compute_demand_score(df)
        assert (result == 0.0).all()


# ──────────────────────────────────────
# compute_competitor_density
# ──────────────────────────────────────

class TestComputeCompetitorDensity:
    def test_returns_series(self, base_df: pd.DataFrame) -> None:
        result = compute_competitor_density(base_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(base_df)

    def test_name_is_competitor_density(self, base_df: pd.DataFrame) -> None:
        result = compute_competitor_density(base_df)
        assert result.name == "competitor_density"

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = compute_competitor_density(empty_df)
        assert result.empty

    def test_density_formula(self) -> None:
        """store_count / population_10k で計算されること。"""
        df = pd.DataFrame({"restaurant_count": [20.0], "population_10k": [4.0]})
        result = compute_competitor_density(df)
        assert result.iloc[0] == pytest.approx(5.0)

    def test_raw_population_converted(self) -> None:
        """population カラム（生値）は 1万人単位に変換されること。"""
        df = pd.DataFrame({"restaurant_count": [20.0], "population": [40_000.0]})
        result = compute_competitor_density(df)
        # 20 / (40000/10000) = 20/4 = 5.0
        assert result.iloc[0] == pytest.approx(5.0)

    def test_no_population_column(self) -> None:
        """人口カラムがない場合は 1万人でゼロ除算せず動作すること。"""
        df = pd.DataFrame({"restaurant_count": [10.0]})
        result = compute_competitor_density(df)
        assert np.isfinite(result.iloc[0])


# ──────────────────────────────────────
# compute_opportunity_score
# ──────────────────────────────────────

class TestComputeOpportunityScore:
    def test_score_column_exists(self, base_df: pd.DataFrame) -> None:
        result = compute_opportunity_score(base_df)
        assert "opportunity_score" in result.columns

    def test_intermediate_columns_exist(self, base_df: pd.DataFrame) -> None:
        result = compute_opportunity_score(base_df)
        for col in ("demand_score", "competitor_density", "population_density_norm", "land_price_norm"):
            assert col in result.columns, f"{col} が存在しない"

    def test_scores_between_0_and_1(self, base_df: pd.DataFrame) -> None:
        result = compute_opportunity_score(base_df)
        opp = result["opportunity_score"]
        assert opp.between(0.0, 1.0).all(), f"範囲外: {opp.tolist()}"

    def test_high_demand_low_competitor_scores_high(self) -> None:
        """需要高・競合少・人口高・地価低の行が最高スコアになること。"""
        df = pd.DataFrame(
            {
                "review_count": [1000, 10],
                "avg_rating": [4.9, 3.0],
                "congestion_index": [1.5, 0.5],
                "restaurant_count": [2, 50],
                "population_10k": [5.0, 5.0],
                "population_density": [20000, 5000],
                "land_price": [1000, 8000],
            }
        )
        result = compute_opportunity_score(df)
        assert result["opportunity_score"].iloc[0] > result["opportunity_score"].iloc[1]

    def test_custom_weights_applied(self, base_df: pd.DataFrame) -> None:
        """重みを変えると結果が変わること。"""
        default = compute_opportunity_score(base_df.copy())
        custom = compute_opportunity_score(base_df.copy(), w_demand=5.0)
        # 少なくとも順位が同じではない（重みで変化する）ことを確認
        assert not (default["opportunity_score"].values == custom["opportunity_score"].values).all()

    def test_laplace_correction_prevents_zero_division(self) -> None:
        """競合密度が 0 でもゼロ除算が起きないこと。"""
        df = pd.DataFrame(
            {
                "review_count": [100],
                "avg_rating": [4.0],
                "restaurant_count": [0],
                "population_10k": [5.0],
                "population_density": [10000],
                "land_price": [2000],
            }
        )
        result = compute_opportunity_score(df)
        assert np.isfinite(result["opportunity_score"].iloc[0])

    def test_empty_dataframe(self, empty_df: pd.DataFrame) -> None:
        result = compute_opportunity_score(empty_df)
        assert result.empty

    def test_original_df_not_modified(self, base_df: pd.DataFrame) -> None:
        """元の DataFrame が変更されないこと（コピーが返る）。"""
        original_cols = set(base_df.columns)
        compute_opportunity_score(base_df)
        assert set(base_df.columns) == original_cols


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
                "area": "渋谷",
                "genre": "カフェ",
                "opportunity_score": 0.85,
                "demand_score": 0.75,
                "competitor_density": 0.5,
                "population_density_norm": 0.8,
                "land_price_norm": 0.3,
            }
        )

    def test_returns_string(self) -> None:
        reason = generate_reason(self._make_row())
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_contains_area_and_genre(self) -> None:
        reason = generate_reason(self._make_row())
        assert "渋谷" in reason
        assert "カフェ" in reason

    def test_contains_score(self) -> None:
        reason = generate_reason(self._make_row())
        assert "0.850" in reason

    def test_missing_optional_fields(self) -> None:
        """area/genre のみの最小行でもクラッシュしないこと。"""
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
        # base_df は4行なので全件返る
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
