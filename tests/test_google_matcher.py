from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess.google_matcher import match_google_to_hotpepper


class TestMatchGoogleToHotpepper:
    def test_matches_nearest_place_within_50m(self) -> None:
        hp_df = pd.DataFrame(
            {
                "id": ["hp1", "hp2"],
                "lat": [35.0, 35.0],
                "lng": [139.0, 139.001],
            }
        )
        gp_df = pd.DataFrame(
            {
                "place_id": ["gp1", "gp2"],
                "lat": [35.0002, 35.0],
                "lng": [139.0002, 139.0017],
                "rating": [4.3, 3.9],
                "review_count": [120, 45],
            }
        )

        result = match_google_to_hotpepper(hp_df, gp_df, max_distance_km=0.05)

        assert result.loc[0, "google_rating"] == pytest.approx(4.3)
        assert result.loc[0, "google_review_count"] == pytest.approx(120)
        assert result.loc[0, "google_match_distance"] < 0.05
        assert pd.isna(result.loc[1, "google_rating"])
        assert pd.isna(result.loc[1, "google_review_count"])
        assert pd.isna(result.loc[1, "google_match_distance"])

    def test_allows_one_to_many_matching(self) -> None:
        hp_df = pd.DataFrame(
            {
                "id": ["hp1", "hp2"],
                "lat": [35.0, 35.00018],
                "lng": [139.0, 139.00018],
            }
        )
        gp_df = pd.DataFrame(
            {
                "place_id": ["gp1"],
                "lat": [35.0001],
                "lng": [139.0001],
                "rating": [4.6],
                "review_count": [88],
            }
        )

        result = match_google_to_hotpepper(hp_df, gp_df, max_distance_km=0.05)

        assert result["google_rating"].notna().sum() == 2
        assert result.loc[0, "google_rating"] == pytest.approx(4.6)
        assert result.loc[1, "google_rating"] == pytest.approx(4.6)

    def test_returns_expected_columns_for_empty_inputs(self) -> None:
        hp_df = pd.DataFrame(columns=["id", "lat", "lng"])
        gp_df = pd.DataFrame(columns=["place_id", "lat", "lng", "rating", "review_count"])

        result = match_google_to_hotpepper(hp_df, gp_df)

        assert list(result.columns) == [
            "id",
            "lat",
            "lng",
            "google_rating",
            "google_review_count",
            "google_match_distance",
        ]
        assert result.empty
