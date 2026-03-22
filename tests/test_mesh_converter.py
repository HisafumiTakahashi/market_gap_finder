from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess.mesh_converter import (
    assign_jis_mesh_quarter,
    lat_lon_to_mesh_quarter,
    mesh_quarter_to_lat_lon,
    mesh_quarter_to_mesh1,
    mesh_quarter_to_mesh3,
)


class TestQuarterMeshConversion:
    def test_lat_lon_to_mesh_quarter_known_code(self) -> None:
        assert lat_lon_to_mesh_quarter(35.7085, 139.6970) == "5339455501"
        assert lat_lon_to_mesh_quarter(35.7128, 139.7035) == "5339455610"

    def test_mesh_quarter_to_lat_lon_returns_center(self) -> None:
        lat, lon = mesh_quarter_to_lat_lon("5339455501")
        assert lat == pytest.approx(35.71041666666667)
        assert lon == pytest.approx(139.696875)

    def test_mesh_quarter_extractors(self) -> None:
        assert mesh_quarter_to_mesh3("5339358611") == "53393586"
        assert mesh_quarter_to_mesh1("5339358611") == "5339"

    def test_round_trip_stays_within_same_quarter_mesh(self) -> None:
        points = [
            (35.7085, 139.6970),
            (35.7128, 139.7035),
            (43.0621, 141.3544),
        ]
        for lat, lon in points:
            mesh = lat_lon_to_mesh_quarter(lat, lon)
            center_lat, center_lon = mesh_quarter_to_lat_lon(mesh)
            assert lat_lon_to_mesh_quarter(center_lat, center_lon) == mesh
            assert abs(center_lat - lat) <= (1 / 240)
            assert abs(center_lon - lon) <= (1 / 160)


class TestAssignJisMeshQuarter:
    def test_assigns_jis_mesh_column(self) -> None:
        df = pd.DataFrame({"lat": [35.7085, 35.7128], "lng": [139.6970, 139.7035]})
        result = assign_jis_mesh_quarter(df)
        assert result["jis_mesh"].tolist() == ["5339455501", "5339455610"]

    def test_invalid_values_become_na(self) -> None:
        df = pd.DataFrame({"lat": [35.7085, None], "lng": [139.6970, "x"]})
        result = assign_jis_mesh_quarter(df)
        assert result.loc[0, "jis_mesh"] == "5339455501"
        assert pd.isna(result.loc[1, "jis_mesh"])
