"""e-Stat 収集処理の基本ユニットテスト。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collect import estat


def test_as_list_none_returns_empty_list() -> None:
    """None を空リストに変換することを確認する。"""
    assert estat._as_list(None) == []


def test_as_list_scalar_wraps_in_list() -> None:
    """単一値を 1 要素のリストに変換することを確認する。"""
    assert estat._as_list("value") == ["value"]


def test_as_list_list_returns_as_is() -> None:
    """リストをそのまま返すことを確認する。"""
    value = ["a", "b"]
    assert estat._as_list(value) == value


def test_get_stats_list_returns_empty_dataframe_when_api_key_missing(monkeypatch) -> None:
    """API キー未設定時に統計一覧が空の DataFrame で返ることを確認する。"""
    monkeypatch.setattr(estat, "ESTAT_API_KEY", "")

    result = estat.get_stats_list("人口")

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["stats_id", "stat_name", "title", "survey_date"]


def test_get_stats_data_returns_empty_dataframe_when_api_key_missing(monkeypatch) -> None:
    """API キー未設定時に統計データが空の DataFrame で返ることを確認する。"""
    monkeypatch.setattr(estat, "ESTAT_API_KEY", "")

    result = estat.get_stats_data("dummy_stats_id")

    assert isinstance(result, pd.DataFrame)
    assert result.empty


def test_fetch_mesh_population_returns_empty_dataframe_for_empty_input() -> None:
    """メッシュコードが空リストなら空の DataFrame を返すことを確認する。"""
    result = estat.fetch_mesh_population([])

    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == ["mesh_code", "population"]
