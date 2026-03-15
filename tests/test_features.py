"""空間特徴量モジュールのユニットテスト。"""

from __future__ import annotations

import pandas as pd
import pytest

from src.analyze.features import (
    _neighbor_mesh_codes,
    add_all_features,
    add_genre_diversity,
    add_genre_hhi,
    add_neighbor_competition,
    add_saturation_index,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "jis_mesh3": ["53393586", "53393586", "53393587", "53393587", "53393587"],
            "unified_genre": ["cafe", "ramen", "cafe", "ramen", "izakaya"],
            "restaurant_count": [10, 5, 8, 3, 7],
            "population": [50000, 50000, 40000, 40000, 40000],
        }
    )


class TestNeighborMeshCodes:
    def test_returns_8_neighbors(self) -> None:
        neighbors = _neighbor_mesh_codes("53393586")
        assert len(neighbors) == 8

    def test_corner_returns_fewer(self) -> None:
        neighbors = _neighbor_mesh_codes("53393500")
        # (0,0) corner: only 3 neighbors
        assert len(neighbors) == 3

    def test_invalid_code(self) -> None:
        assert _neighbor_mesh_codes("abc") == []
        assert _neighbor_mesh_codes("123") == []


class TestAddGenreDiversity:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_diversity(sample_df)
        assert "genre_diversity" in result.columns

    def test_correct_counts(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_diversity(sample_df)
        # 53393586: cafe, ramen => 2 genres
        assert result.loc[result["jis_mesh3"] == "53393586", "genre_diversity"].iloc[0] == 2
        # 53393587: cafe, ramen, izakaya => 3 genres
        assert result.loc[result["jis_mesh3"] == "53393587", "genre_diversity"].iloc[0] == 3

    def test_empty_df(self) -> None:
        result = add_genre_diversity(pd.DataFrame())
        assert "genre_diversity" in result.columns


class TestAddGenreHHI:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_hhi(sample_df)
        assert "genre_hhi" in result.columns

    def test_hhi_range(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_hhi(sample_df)
        assert (result["genre_hhi"] >= 0).all()
        assert (result["genre_hhi"] <= 1).all()

    def test_single_genre_hhi_is_one(self) -> None:
        df = pd.DataFrame(
            {"jis_mesh3": ["m1"], "unified_genre": ["cafe"], "restaurant_count": [10]}
        )
        result = add_genre_hhi(df)
        assert result["genre_hhi"].iloc[0] == pytest.approx(1.0)


class TestAddNeighborCompetition:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_neighbor_competition(sample_df)
        assert "neighbor_avg_restaurants" in result.columns

    def test_neighbor_values_nonnegative(self, sample_df: pd.DataFrame) -> None:
        result = add_neighbor_competition(sample_df)
        assert (result["neighbor_avg_restaurants"] >= 0).all()


class TestAddSaturationIndex:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_saturation_index(sample_df)
        assert "saturation_index" in result.columns

    def test_saturation_nonnegative(self, sample_df: pd.DataFrame) -> None:
        result = add_saturation_index(sample_df)
        assert (result["saturation_index"] >= 0).all()


class TestAddAllFeatures:
    def test_adds_all_columns(self, sample_df: pd.DataFrame) -> None:
        result = add_all_features(sample_df)
        for col in ("genre_diversity", "genre_hhi", "neighbor_avg_restaurants", "saturation_index"):
            assert col in result.columns, f"{col} missing"

    def test_row_count_preserved(self, sample_df: pd.DataFrame) -> None:
        result = add_all_features(sample_df)
        assert len(result) == len(sample_df)
