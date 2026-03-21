from __future__ import annotations

import pandas as pd
import pytest

from src.analyze.features import (
    _neighbor_mesh_codes,
    add_all_features,
    add_genre_diversity,
    add_genre_hhi,
    add_genre_saturation,
    add_genre_share,
    add_neighbor_competition,
    add_neighbor_population,
    add_other_genre_count,
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

    def test_corner_wraps_across_boundary(self) -> None:
        # s=0,t=0 → ds=-1 wraps s→9 and q→q-1, dt=-1 wraps t→9 and r→r-1
        neighbors = _neighbor_mesh_codes("53393500")
        assert len(neighbors) == 8
        assert "53392499" in neighbors  # q:3→2, r:5→4, s:9, t:9
        assert "53393501" in neighbors

    def test_crosses_second_mesh_boundary(self) -> None:
        # q=3,r=0,s=0,t=0 → ds=-1,dt=-1 wraps s→9,q→2 and t→9,r→7,u→38
        neighbors = _neighbor_mesh_codes("53393000")
        assert "53382799" in neighbors  # u:39→38, q:3→2, r:0→7, s:9, t:9

    def test_invalid_code(self) -> None:
        assert _neighbor_mesh_codes("abc") == []
        assert _neighbor_mesh_codes("123") == []


class TestAddGenreDiversity:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_diversity(sample_df)
        assert "genre_diversity" in result.columns

    def test_correct_counts(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_diversity(sample_df)
        assert result.loc[result["jis_mesh3"] == "53393586", "genre_diversity"].iloc[0] == 2
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

    def test_single_genre_hhi_is_zero_when_no_other_rows_exist(self) -> None:
        df = pd.DataFrame({"jis_mesh3": ["m1"], "unified_genre": ["cafe"], "restaurant_count": [10]})
        result = add_genre_hhi(df)
        assert result["genre_hhi"].iloc[0] == pytest.approx(0.0)


class TestAddNeighborCompetition:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_neighbor_competition(sample_df)
        assert "neighbor_avg_restaurants" in result.columns

    def test_neighbor_values_nonnegative(self, sample_df: pd.DataFrame) -> None:
        result = add_neighbor_competition(sample_df)
        assert (result["neighbor_avg_restaurants"] >= 0).all()


class TestAddOtherGenreCount:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_other_genre_count(sample_df)
        assert "other_genre_count" in result.columns

    def test_counts_other_genres_within_mesh(self, sample_df: pd.DataFrame) -> None:
        result = add_other_genre_count(sample_df)
        mesh1 = result[result["jis_mesh3"] == "53393586"].sort_values("restaurant_count")
        assert mesh1["other_genre_count"].tolist() == [10, 5]

    def test_missing_mesh_column_defaults_zero(self) -> None:
        result = add_other_genre_count(pd.DataFrame({"restaurant_count": [3, 5]}))
        assert result["other_genre_count"].tolist() == [0, 0]


class TestAddGenreShare:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_share(sample_df)
        assert "genre_share" in result.columns

    def test_share_within_mesh(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_share(sample_df)
        mesh1 = result[result["jis_mesh3"] == "53393586"].sort_values("restaurant_count")
        assert mesh1["genre_share"].tolist() == pytest.approx([5 / 15, 10 / 15])


class TestAddSaturationIndex:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_saturation_index(sample_df)
        assert "saturation_index" in result.columns

    def test_saturation_nonnegative(self, sample_df: pd.DataFrame) -> None:
        result = add_saturation_index(sample_df)
        assert (result["saturation_index"] >= 0).all()

    def test_uses_leave_one_out_restaurant_count(self, sample_df: pd.DataFrame) -> None:
        result = add_saturation_index(sample_df)
        expected = 5 / (50000 / 10000 + 0.1)
        assert result.loc[0, "saturation_index"] == pytest.approx(expected)


class TestAddGenreSaturation:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_saturation(sample_df)
        assert "genre_saturation" in result.columns

    def test_uses_row_level_restaurant_count_and_population(self, sample_df: pd.DataFrame) -> None:
        result = add_genre_saturation(sample_df)
        expected = 10 / (50000 / 10000 + 0.1)
        assert result.loc[0, "genre_saturation"] == pytest.approx(expected)


class TestAddNeighborPopulation:
    def test_adds_column(self, sample_df: pd.DataFrame) -> None:
        result = add_neighbor_population(sample_df)
        assert "neighbor_avg_population" in result.columns

    def test_neighbor_population_values_nonnegative(self, sample_df: pd.DataFrame) -> None:
        result = add_neighbor_population(sample_df)
        assert (result["neighbor_avg_population"] >= 0).all()

    def test_uses_neighbor_mesh_population_average(self) -> None:
        df = pd.DataFrame(
            {
                "jis_mesh3": ["53393586", "53393587", "53393596"],
                "unified_genre": ["cafe", "ramen", "sushi"],
                "restaurant_count": [2, 3, 4],
                "population": [1000, 2000, 3000],
            }
        )
        result = add_neighbor_population(df)
        target = result.loc[result["jis_mesh3"] == "53393586", "neighbor_avg_population"].iloc[0]
        assert target == pytest.approx((2000 + 3000) / 8)


class TestAddAllFeatures:
    def test_adds_all_columns(self, sample_df: pd.DataFrame) -> None:
        result = add_all_features(sample_df)
        for col in (
            "genre_diversity",
            "genre_hhi",
            "other_genre_count",
            "genre_share",
            "neighbor_avg_restaurants",
            "neighbor_avg_population",
            "saturation_index",
            "genre_saturation",
        ):
            assert col in result.columns, f"{col} missing"

    def test_row_count_preserved(self, sample_df: pd.DataFrame) -> None:
        result = add_all_features(sample_df)
        assert len(result) == len(sample_df)
