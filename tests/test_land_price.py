from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from src.collect import land_price


@pytest.fixture()
def land_price_df() -> pd.DataFrame:
    return pd.DataFrame({"lat": [35.68], "lng": [139.76], "price_per_sqm": [1200000.0]})


class TestParseLandPriceGml:
    def test_parse_minimal_gml(self) -> None:
        gml = b"""
        <root xmlns:gml="http://www.opengis.net/gml/3.2">
          <PublicLandPrice>
            <gml:pos>35.6800 139.7600</gml:pos>
            <L01_006>1,200,000</L01_006>
          </PublicLandPrice>
        </root>
        """
        result = land_price._parse_land_price_gml(gml)
        assert len(result) == 1
        assert result.loc[0, "lat"] == pytest.approx(35.68)
        assert result.loc[0, "lng"] == pytest.approx(139.76)
        assert result.loc[0, "price_per_sqm"] == pytest.approx(1200000.0)


class TestParseLandPriceGeojson:
    def test_parse_minimal_geojson(self) -> None:
        geojson = b"""
        {
          "type": "FeatureCollection",
          "features": [
            {
              "type": "Feature",
              "properties": {"L01_008": "1200000"},
              "geometry": {"type": "Point", "coordinates": [139.76, 35.68]}
            }
          ]
        }
        """

        result = land_price._parse_land_price_geojson(geojson)

        assert len(result) == 1
        assert result.loc[0, "lat"] == pytest.approx(35.68)
        assert result.loc[0, "lng"] == pytest.approx(139.76)
        assert result.loc[0, "price_per_sqm"] == pytest.approx(1200000.0)

    def test_parse_invalid_json_returns_empty(self) -> None:
        result = land_price._parse_land_price_geojson(b"\x80\x81\x82")

        assert result.empty
        assert list(result.columns) == ["lat", "lng", "price_per_sqm"]


class TestExtractPriceRecord:
    def test_extract_price_record_normal(self) -> None:
        elem = ET.fromstring(
            """
            <PublicLandPrice xmlns:gml="http://www.opengis.net/gml/3.2">
              <gml:pos>35.6800 139.7600</gml:pos>
              <price>950000</price>
            </PublicLandPrice>
            """
        )
        assert land_price._extract_price_record(elem) == {
            "lat": 35.68,
            "lng": 139.76,
            "price_per_sqm": 950000.0,
        }

    def test_extract_price_record_missing_data(self) -> None:
        elem = ET.fromstring("<PublicLandPrice><price>950000</price></PublicLandPrice>")
        assert land_price._extract_price_record(elem) is None


class TestLandPriceCache:
    def test_save_and_load_land_price_cache(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        land_price_df: pd.DataFrame,
    ) -> None:
        monkeypatch.setattr(land_price.settings, "EXTERNAL_DATA_DIR", tmp_path)
        land_price.save_land_price_cache(land_price_df, "tokyo")
        loaded = land_price.load_land_price_cache("tokyo")
        assert loaded is not None
        pd.testing.assert_frame_equal(loaded, land_price_df)

    def test_load_missing_cache_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(land_price.settings, "EXTERNAL_DATA_DIR", tmp_path)

        assert land_price.load_land_price_cache("missing") is None


class TestPrefectureCodes:
    def test_prefecture_codes(self) -> None:
        assert land_price.PREFECTURE_CODES["東京都"] == "13"
        assert land_price.PREFECTURE_CODES["大阪府"] == "27"
        assert land_price.PREFECTURE_CODES["愛知県"] == "23"
