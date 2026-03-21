"""
e-Stat API クライアント

政府統計の総合窓口 e-Stat API を利用して、
統計表一覧および統計データを取得する。

Reference
---------
https://www.e-stat.go.jp/api/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from config import settings
from config.settings import ESTAT_API_BASE_URL, ESTAT_API_KEY

logger = logging.getLogger(__name__)


def _as_list(value: Any) -> list[Any]:
    """値をリストとして扱える形に正規化する。"""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _extract_stats_list_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """getStatsList レスポンスの実データ部分を取り出す。"""
    return payload.get("GET_STATS_LIST", {}).get("DATALIST_INF", {})


def _extract_stats_data_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """getStatsData レスポンスの実データ部分を取り出す。"""
    return payload.get("GET_STATS_DATA", {}).get("STATISTICAL_DATA", {})


def _build_class_maps(class_inf: dict[str, Any]) -> dict[str, dict[str, str]]:
    """CLASS_INF からコードと名称の対応表を構築する。"""
    class_maps: dict[str, dict[str, str]] = {}

    for class_obj in _as_list(class_inf.get("CLASS_OBJ")):
        if not isinstance(class_obj, dict):
            continue

        class_id = class_obj.get("@id")
        if not isinstance(class_id, str) or not class_id:
            continue

        code_map: dict[str, str] = {}
        for class_item in _as_list(class_obj.get("CLASS")):
            if not isinstance(class_item, dict):
                continue

            code = class_item.get("@code")
            name = class_item.get("@name")
            if isinstance(code, str) and isinstance(name, str):
                code_map[code] = name

        if code_map:
            class_maps[class_id] = code_map

    return class_maps


def _apply_class_maps(df: pd.DataFrame, class_maps: dict[str, dict[str, str]]) -> pd.DataFrame:
    """CLASS_INF のコード名称マッピングを DataFrame に適用する。"""
    if df.empty:
        return df

    mapped = df.copy()
    for class_id, code_map in class_maps.items():
        candidate_columns = [class_id, f"@{class_id}"]
        for column in candidate_columns:
            if column not in mapped.columns:
                continue

            mapped[column] = mapped[column].map(
                lambda x, cm=code_map: cm.get(str(x), x) if pd.notna(x) else x
            )

            if column.startswith("@"):
                plain_name = column[1:]
                if plain_name not in mapped.columns:
                    mapped = mapped.rename(columns={column: plain_name})
            break

    return mapped


def get_stats_list(search_word: str, limit: int = 10) -> pd.DataFrame:
    """統計表一覧を検索する。

    Parameters
    ----------
    search_word : str
        統計表の検索キーワード
    limit : int, optional
        取得件数の上限, デフォルト 10

    Returns
    -------
    pd.DataFrame
        統計表一覧。返却列は `stats_id`, `stat_name`, `title`, `survey_date`
    """
    columns = ["stats_id", "stat_name", "title", "survey_date"]
    if not ESTAT_API_KEY:
        logger.error("ESTAT_API_KEY が設定されていません")
        return pd.DataFrame(columns=columns)

    params = {
        "appId": ESTAT_API_KEY,
        "searchWord": search_word,
        "limit": max(1, int(limit)),
    }

    try:
        response = requests.get(
            f"{ESTAT_API_BASE_URL}/json/getStatsList",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.error("e-Stat getStatsList リクエスト失敗: %s", e)
        return pd.DataFrame(columns=columns)

    table_infos = _as_list(_extract_stats_list_payload(payload).get("TABLE_INF"))
    rows: list[dict[str, Any]] = []
    for table_info in table_infos:
        if not isinstance(table_info, dict):
            continue

        stat_name = table_info.get("STAT_NAME")
        title = table_info.get("TITLE")
        survey_date = table_info.get("SURVEY_DATE")

        rows.append(
            {
                "stats_id": table_info.get("@id"),
                "stat_name": stat_name.get("$") if isinstance(stat_name, dict) else stat_name,
                "title": title.get("$") if isinstance(title, dict) else title,
                "survey_date": survey_date,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def get_stats_data(stats_data_id: str, cd_area: str | None = None, limit: int = 100000) -> pd.DataFrame:
    """統計データを取得する。

    Parameters
    ----------
    stats_data_id : str
        取得対象の統計データ ID
    cd_area : str | None, optional
        地域コード。指定時は対象地域で絞り込む
    limit : int, optional
        取得件数の上限, デフォルト 100000

    Returns
    -------
    pd.DataFrame
        CLASS_INF を適用して整形した統計データ
    """
    if not ESTAT_API_KEY:
        logger.error("ESTAT_API_KEY が設定されていません")
        return pd.DataFrame()

    params: dict[str, Any] = {
        "appId": ESTAT_API_KEY,
        "statsDataId": stats_data_id,
        "limit": max(1, int(limit)),
    }
    if cd_area:
        params["cdArea"] = cd_area

    try:
        response = requests.get(
            f"{ESTAT_API_BASE_URL}/json/getStatsData",
            params=params,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.error("e-Stat getStatsData リクエスト失敗: %s", e)
        return pd.DataFrame()

    stats_data = _extract_stats_data_payload(payload)
    class_maps = _build_class_maps(stats_data.get("CLASS_INF", {}))

    data_inf = stats_data.get("DATA_INF", {})
    values = _as_list(data_inf.get("VALUE"))
    rows = [value for value in values if isinstance(value, dict)]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "$" in df.columns and "value" not in df.columns:
        df = df.rename(columns={"$": "value"})

    df = _apply_class_maps(df, class_maps)
    return df


def _find_mesh_stats_data_id(mesh1_code: str, survey_year: str = "2020") -> str | None:
    """1次メッシュコードに対応する3次メッシュ人口統計のstatsDataIdを検索する。

    Parameters
    ----------
    mesh1_code : str
        4桁の1次メッシュコード (例: "5339")
    survey_year : str
        調査年 (例: "2020")

    Returns
    -------
    str | None
        見つかった statsDataId。見つからない場合は None
    """
    if not ESTAT_API_KEY:
        return None

    params = {
        "appId": ESTAT_API_KEY,
        "statsCode": "00200521",
        "searchKind": "2",
        "searchWord": f"人口 世帯 1次メッシュ M{mesh1_code}",
        "surveyYears": survey_year,
        "limit": 10,
    }

    try:
        response = requests.get(
            f"{ESTAT_API_BASE_URL}/json/getStatsList",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as e:
        logger.error("メッシュ統計表の検索に失敗: mesh1=%s, error=%s", mesh1_code, e)
        return None

    datalist = payload.get("GET_STATS_LIST", {}).get("DATALIST_INF", {})
    tables = _as_list(datalist.get("TABLE_INF"))

    if not tables:
        logger.warning("メッシュ統計表が見つかりません: mesh1=%s, year=%s", mesh1_code, survey_year)
        return None

    # 最少レコード数のテーブル = 3次メッシュ（最も粗い解像度）
    best: dict[str, Any] | None = None
    best_count = float("inf")
    for t in tables:
        if not isinstance(t, dict):
            continue
        count_str = t.get("OVERALL_TOTAL_NUMBER", "")
        try:
            count = int(count_str)
        except (ValueError, TypeError):
            count = float("inf")
        if count < best_count:
            best_count = count
            best = t

    if best is None:
        return None

    stats_id = best.get("@id")
    logger.info(
        "メッシュ統計表を特定: mesh1=%s => statsDataId=%s (records=%s)",
        mesh1_code, stats_id, best_count,
    )
    return stats_id


def fetch_mesh_population(
    mesh1_codes: list[str],
    stats_data_id: str | None = None,
    survey_year: str = "2020",
) -> pd.DataFrame:
    """1次メッシュ単位の人口データを取得する。

    各1次メッシュについて e-Stat の getStatsList で正しい statsDataId を
    動的に検索し、3次メッシュ解像度の人口総数データを取得する。

    Parameters
    ----------
    mesh1_codes : list[str]
        取得対象の 1 次メッシュコード一覧
    stats_data_id : str | None, optional
        指定時はすべてのメッシュにこの ID を使用する (後方互換)
    survey_year : str, optional
        調査年, デフォルト "2020"

    Returns
    -------
    pd.DataFrame
        `mesh_code`, `population` を含む人口データ
    """
    frames: list[pd.DataFrame] = []

    for mesh_code in mesh1_codes:
        sid = stats_data_id or _find_mesh_stats_data_id(mesh_code, survey_year)
        if not sid:
            logger.error("statsDataId が見つかりません: mesh1=%s", mesh_code)
            continue

        df = get_stats_data(stats_data_id=sid)
        if df.empty:
            logger.error("メッシュ人口の取得に失敗しました: mesh_code=%s, statsDataId=%s", mesh_code, sid)
            continue

        renamed = df.copy()
        rename_map: dict[str, str] = {}
        if "area" in renamed.columns:
            rename_map["area"] = "mesh_code"
        elif "@area" in renamed.columns:
            rename_map["@area"] = "mesh_code"

        if "value" in renamed.columns:
            rename_map["value"] = "population"
        elif "$" in renamed.columns:
            rename_map["$"] = "population"

        renamed = renamed.rename(columns=rename_map)
        if "mesh_code" not in renamed.columns:
            renamed["mesh_code"] = mesh_code

        # cat01=0010 (人口総数) のみ抽出
        if "cat01" in renamed.columns:
            cat01_col = renamed["cat01"].astype(str)
            total_mask = cat01_col.isin(["0010"]) | cat01_col.str.contains("人口（総数）|人口総数", na=False)
            if total_mask.any():
                renamed = renamed.loc[total_mask].copy()

        # 秘匿データ除外 (cat02=1: 無し のみ)
        if "cat02" in renamed.columns:
            cat02_col = renamed["cat02"].astype(str)
            valid_mask = cat02_col.isin(["1", "無し"])
            if valid_mask.any():
                renamed = renamed.loc[valid_mask].copy()

        frames.append(renamed)

    if not frames:
        return pd.DataFrame(columns=["mesh_code", "population"])

    return pd.concat(frames, ignore_index=True)


def save_raw(df: pd.DataFrame, filename: str) -> None:
    """Raw データとして CSV 保存する。

    Parameters
    ----------
    df : pd.DataFrame
        保存対象
    filename : str
        保存ファイル名。data/raw/ 配下に保存する
    """
    settings.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    target = Path(filename)
    if target.suffix.lower() != ".csv":
        target = target.with_suffix(".csv")

    output_path = settings.RAW_DATA_DIR / target.name
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info("e-Stat raw データ保存: %s", output_path)
