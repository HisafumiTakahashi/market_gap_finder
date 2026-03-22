from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.visualize.report import generate_explanation, generate_report


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

    def test_empty_data_returns_fallback(self) -> None:
        text = generate_explanation(
            pd.Series(
                {
                    "population": 0,
                    "restaurant_count": 0,
                    "unified_genre": "",
                    "genre_diversity": 0,
                    "saturation_index": 0,
                    "nearest_station_distance": 0,
                    "nearest_station_name": "",
                    "market_gap": 0,
                }
            )
        )
        assert text == "追加確認が必要です。"

    def test_high_diversity_message(self) -> None:
        text = generate_explanation(pd.Series({"genre_diversity": 9}))
        assert "多様性" in text

    def test_high_saturation_message(self) -> None:
        text = generate_explanation(pd.Series({"saturation_index": 25}))
        assert "飽和度が高く" in text


class TestGenerateReport:
    def test_generate_report_creates_html(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tag = "area_tag"
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()
        pd.DataFrame(
            {
                "jis_mesh": ["5339453711", "5339458811"],
                "unified_genre": ["cafe", "ramen"],
                "restaurant_count": [5, 3],
                "population": [5000, 3000],
                "lat": [35.68, 35.69],
                "lng": [139.76, 139.77],
                "opportunity_score": [0.8, 0.6],
                "saturation_index": [5.0, 3.0],
                "genre_diversity": [6, 4],
                "nearest_station_distance": [0.3, 0.5],
                "nearest_station_name": ["新宿", "渋谷"],
                "reason": ["テスト理由1", "テスト理由2"],
            }
        ).to_csv(processed_dir / f"{tag}_integrated.csv", index=False)
        monkeypatch.setattr("src.visualize.report.settings.PROCESSED_DATA_DIR", processed_dir)
        monkeypatch.setattr("src.visualize.report.settings.PROJECT_ROOT", tmp_path)

        output_path = generate_report(tag)

        assert output_path.exists()
        html = output_path.read_text(encoding="utf-8")
        assert "<html" in html.lower()
        assert "area_tag" in html
        assert "Market Gap Report" in html
