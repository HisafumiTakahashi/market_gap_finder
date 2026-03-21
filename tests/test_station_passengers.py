from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.analyze.features import add_nearest_station_passengers
from src.collect import station_passengers


def _build_geojson_zip(features: list[dict]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("S12_test.geojson", json.dumps({"type": "FeatureCollection", "features": features}))
    return buffer.getvalue()


class DummyResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


class TestDownloadStationPassengers:
    def test_download_station_passengers_parses_geojson(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        payload = _build_geojson_zip(
            [
                {
                    "type": "Feature",
                    "properties": {
                        "S12_001": "東京",
                        "S12_002": "JR東日本",
                        "S12_003": "山手線",
                        "S12_006": "dup",
                        "S12_009": 1000,
                        "S12_013": 1200,
                        "S12_053": 1500,
                    },
                    "geometry": {"type": "Point", "coordinates": [139.7671, 35.6812]},
                },
                {
                    "type": "Feature",
                    "properties": {
                        "S12_001": "東京",
                        "S12_002": "JR東日本",
                        "S12_003": "京浜東北線",
                        "S12_053": 9999,
                    },
                    "geometry": {"type": "Point", "coordinates": [139.7671001, 35.6812001]},
                },
                {
                    "type": "Feature",
                    "properties": {
                        "S12_001": "新宿",
                        "S12_002": "JR東日本",
                        "S12_003": "中央線",
                        "S12_009": "bad",
                        "S12_049": "2000",
                    },
                    "geometry": {"type": "Point", "coordinates": [139.7006, 35.6896]},
                },
            ]
        )

        def fake_get(url: str, timeout: int) -> DummyResponse:
            assert "S12-22_GML.zip" in url
            assert timeout == 120
            return DummyResponse(payload)

        monkeypatch.setattr(station_passengers.requests, "get", fake_get)

        result = station_passengers.download_station_passengers(2022)

        assert list(result.columns) == ["station_name", "operator", "line_name", "lat", "lng", "passengers"]
        assert len(result) == 2
        assert result.loc[0, "station_name"] == "東京"
        assert result.loc[0, "passengers"] == pytest.approx(1500.0)
        assert result.loc[1, "station_name"] == "新宿"
        assert result.loc[1, "passengers"] == pytest.approx(2000.0)


class TestPassengerCache:
    def test_save_and_load_passenger_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        df = pd.DataFrame(
            {
                "station_name": ["東京"],
                "operator": ["JR東日本"],
                "line_name": ["山手線"],
                "lat": [35.6812],
                "lng": [139.7671],
                "passengers": [1500.0],
            }
        )
        monkeypatch.setattr(station_passengers.settings, "EXTERNAL_DATA_DIR", tmp_path)

        station_passengers.save_passenger_cache(df, "tokyo")
        loaded = station_passengers.load_passenger_cache("tokyo")

        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, df)


class TestAddNearestStationPassengers:
    def test_matches_by_nearest_station_name_first(self) -> None:
        df = pd.DataFrame(
            {
                "lat": [35.68, 35.69],
                "lng": [139.76, 139.70],
                "nearest_station_name": ["東京", "新宿"],
            }
        )
        passenger_df = pd.DataFrame(
            {
                "station_name": ["東京", "新宿"],
                "lat": [1.0, 2.0],
                "lng": [3.0, 4.0],
                "passengers": [1500.0, 2500.0],
            }
        )

        result = add_nearest_station_passengers(df, passenger_df)

        assert result["nearest_station_passengers"].tolist() == [1500.0, 2500.0]

    def test_falls_back_to_coordinate_match(self) -> None:
        df = pd.DataFrame({"lat": [35.6812], "lng": [139.7671]})
        passenger_df = pd.DataFrame(
            {
                "station_name": ["東京", "新宿"],
                "lat": [35.6812, 35.6896],
                "lng": [139.7671, 139.7006],
                "passengers": [1500.0, 2500.0],
            }
        )

        result = add_nearest_station_passengers(df, passenger_df)

        assert result.loc[0, "nearest_station_passengers"] == pytest.approx(1500.0)

    def test_empty_passenger_df_returns_zero(self) -> None:
        df = pd.DataFrame({"lat": [35.6812], "lng": [139.7671]})

        result = add_nearest_station_passengers(df, pd.DataFrame())

        assert result.loc[0, "nearest_station_passengers"] == 0.0
