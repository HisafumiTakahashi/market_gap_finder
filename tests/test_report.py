from __future__ import annotations

import pandas as pd

from src.visualize.report import generate_explanation


class TestGenerateExplanation:
    def test_high_population_pattern(self) -> None:
        text = generate_explanation(pd.Series({"population": 60000, "restaurant_count": 2, "unified_genre": "cafe"}))
        assert isinstance(text, str)
        assert "。" in text
        assert text.endswith("。")

    def test_low_population_pattern(self) -> None:
        text = generate_explanation(pd.Series({"population": 10000, "restaurant_count": 2, "unified_genre": "ramen"}))
        assert isinstance(text, str)
        assert text.endswith("。")

    def test_near_station_pattern(self) -> None:
        text = generate_explanation(
            pd.Series(
                {
                    "population": 30000,
                    "restaurant_count": 2,
                    "unified_genre": "izakaya",
                    "nearest_station_name": "新宿",
                    "nearest_station_distance": 0.2,
                }
            )
        )
        assert isinstance(text, str)
        assert text.endswith("。")

    def test_ml_gap_pattern(self) -> None:
        text = generate_explanation(
            pd.Series(
                {
                    "population": 30000,
                    "restaurant_count": 2,
                    "unified_genre": "sushi",
                    "market_gap": 1.2,
                }
            )
        )
        assert isinstance(text, str)
        assert text.endswith("。")
