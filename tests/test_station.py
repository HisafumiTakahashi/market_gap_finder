from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from src.collect import station


@pytest.fixture()
def station_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_name": ["Tokyo", "Shinjuku"],
            "lat": [35.681236, 35.690921],
            "lng": [139.767125, 139.700258],
        }
    )


class TestHaversineKm:
    def test_tokyo_to_shinjuku(self) -> None:
        distance = station.haversine_km(35.681236, 139.767125, 35.690921, 139.700258)
        assert distance == pytest.approx(6.4, abs=0.3)

    def test_same_point_is_zero(self) -> None:
        assert station.haversine_km(35.681236, 139.767125, 35.681236, 139.767125) == pytest.approx(0.0)


class TestFetchLines:
    def test_fetch_lines_normal_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = Mock()
        response.json.return_value = {"response": {"line": ["Yamanote", "Chuo"]}}
        response.raise_for_status.return_value = None
        monkeypatch.setattr(station.requests, "get", Mock(return_value=response))
        assert station.fetch_lines("東京都") == ["Yamanote", "Chuo"]

    def test_fetch_lines_single_string_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = Mock()
        response.json.return_value = {"response": {"line": "Yamanote"}}
        response.raise_for_status.return_value = None
        monkeypatch.setattr(station.requests, "get", Mock(return_value=response))
        assert station.fetch_lines("東京都") == ["Yamanote"]


class TestFetchStations:
    def test_fetch_stations_normal_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = Mock()
        response.json.return_value = {
            "response": {
                "station": [
                    {"name": "Tokyo", "x": "139.767125", "y": "35.681236"},
                    {"name": "Shinjuku", "x": "139.700258", "y": "35.690921"},
                ]
            }
        }
        response.raise_for_status.return_value = None
        monkeypatch.setattr(station.requests, "get", Mock(return_value=response))
        stations = station.fetch_stations("Yamanote")
        assert len(stations) == 2
        assert stations[0]["name"] == "Tokyo"

    def test_fetch_stations_single_dict_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = Mock()
        response.json.return_value = {"response": {"station": {"name": "Tokyo", "x": "139.767125", "y": "35.681236"}}}
        response.raise_for_status.return_value = None
        monkeypatch.setattr(station.requests, "get", Mock(return_value=response))
        assert station.fetch_stations("Yamanote") == [{"name": "Tokyo", "x": "139.767125", "y": "35.681236"}]


class TestStationCache:
    def test_save_and_load_station_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        station_df: pd.DataFrame,
    ) -> None:
        monkeypatch.setattr(station.settings, "EXTERNAL_DATA_DIR", tmp_path)
        station.save_station_cache(station_df, "tokyo")
        loaded = station.load_station_cache("tokyo")
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, station_df)
