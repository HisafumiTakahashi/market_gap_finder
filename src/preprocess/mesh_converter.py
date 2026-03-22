"""JIS標準メッシュ変換ユーティリティ。"""

from __future__ import annotations

import math

import pandas as pd

_HALF_MAP = {(0, 0): 1, (0, 1): 2, (1, 0): 3, (1, 1): 4}
_CODE_TO_IDX = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}


def lat_lon_to_mesh3(lat: float, lon: float) -> str:
    """緯度経度を JIS 標準 3 次メッシュコードへ変換する。"""
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("lat and lon must be finite numbers")

    p = int(lat * 1.5)
    u = int(lon - 100)
    q = int((lat - p / 1.5) * 12)
    r = int((lon - u - 100) * 8)
    s = int((lat - p / 1.5 - q / 12) * 120)
    t = int((lon - u - 100 - r / 8) * 80)
    return f"{p:02d}{u:02d}{q}{r}{s}{t}"


def mesh3_to_lat_lon(mesh_code: str) -> tuple[float, float]:
    """JIS 標準 3 次メッシュコードから中心緯度経度を返す。"""
    code = str(mesh_code).strip()
    if len(code) != 8 or not code.isdigit():
        raise ValueError("mesh_code must be an 8-digit string")

    p = int(code[0:2])
    u = int(code[2:4])
    q = int(code[4])
    r = int(code[5])
    s = int(code[6])
    t = int(code[7])

    lat_sw = p / 1.5 + q / 12 + s / 120
    lon_sw = u + 100 + r / 8 + t / 80

    lat = lat_sw + 1 / 240
    lon = lon_sw + 1 / 160
    return (lat, lon)


def mesh3_to_mesh1(mesh3_code: str) -> str:
    """3次メッシュコードから1次メッシュコードを取り出す。"""
    code = str(mesh3_code).strip()
    if len(code) < 4 or not code[:4].isdigit():
        raise ValueError("mesh3_code must start with a 4-digit mesh1 code")
    return code[:4]


def lat_lon_to_mesh_quarter(lat: float, lon: float) -> str:
    """緯度経度を JIS 4分の1メッシュコードへ変換する。"""
    if not math.isfinite(lat) or not math.isfinite(lon):
        raise ValueError("lat and lon must be finite numbers")

    mesh3 = lat_lon_to_mesh3(lat, lon)

    p = int(mesh3[0:2])
    u = int(mesh3[2:4])
    q = int(mesh3[4])
    r = int(mesh3[5])
    s = int(mesh3[6])
    t = int(mesh3[7])

    lat_sw = p / 1.5 + q / 12 + s / 120
    lon_sw = u + 100 + r / 8 + t / 80

    mesh3_lat_h = 1 / 120
    mesh3_lon_w = 1 / 80

    lat_rel = lat - lat_sw
    lon_rel = lon - lon_sw

    half_lat = 1 if lat_rel >= mesh3_lat_h / 2 else 0
    half_lon = 1 if lon_rel >= mesh3_lon_w / 2 else 0
    half_code = _HALF_MAP[(half_lat, half_lon)]

    half_lat_sw = lat_sw + half_lat * (mesh3_lat_h / 2)
    half_lon_sw = lon_sw + half_lon * (mesh3_lon_w / 2)
    lat_in_half = lat - half_lat_sw
    lon_in_half = lon - half_lon_sw

    quarter_lat = 1 if lat_in_half >= mesh3_lat_h / 4 else 0
    quarter_lon = 1 if lon_in_half >= mesh3_lon_w / 4 else 0
    quarter_code = _HALF_MAP[(quarter_lat, quarter_lon)]

    return f"{mesh3}{half_code}{quarter_code}"


def mesh_quarter_to_lat_lon(mesh_code: str) -> tuple[float, float]:
    """JIS 4分の1メッシュコードから中心緯度経度を返す。"""
    code = str(mesh_code).strip()
    if len(code) != 10 or not code.isdigit():
        raise ValueError("mesh_code must be a 10-digit string")

    mesh3 = code[:8]
    half_code = int(code[8])
    quarter_code = int(code[9])
    if half_code not in _CODE_TO_IDX or quarter_code not in _CODE_TO_IDX:
        raise ValueError("quarter mesh suffix digits must be in 1-4")

    p = int(mesh3[0:2])
    u = int(mesh3[2:4])
    q = int(mesh3[4])
    r = int(mesh3[5])
    s = int(mesh3[6])
    t = int(mesh3[7])

    lat_sw = p / 1.5 + q / 12 + s / 120
    lon_sw = u + 100 + r / 8 + t / 80
    mesh3_lat_h = 1 / 120
    mesh3_lon_w = 1 / 80
    h_lat, h_lon = _CODE_TO_IDX[half_code]
    q_lat, q_lon = _CODE_TO_IDX[quarter_code]

    lat = lat_sw + h_lat * (mesh3_lat_h / 2) + q_lat * (mesh3_lat_h / 4) + mesh3_lat_h / 8
    lon = lon_sw + h_lon * (mesh3_lon_w / 2) + q_lon * (mesh3_lon_w / 4) + mesh3_lon_w / 8
    return (lat, lon)


def mesh_quarter_to_mesh3(mesh_quarter: str) -> str:
    """10桁メッシュコードから8桁3次メッシュコードを取り出す。"""
    code = str(mesh_quarter).strip()
    if len(code) < 8 or not code[:8].isdigit():
        raise ValueError("mesh_quarter must be at least 8 digits")
    return code[:8]


def mesh_quarter_to_mesh1(mesh_quarter: str) -> str:
    """10桁メッシュコードから4桁1次メッシュコードを取り出す。"""
    code = str(mesh_quarter).strip()
    if len(code) < 4 or not code[:4].isdigit():
        raise ValueError("mesh_quarter must start with a 4-digit mesh1 code")
    return code[:4]


def assign_jis_mesh(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lng_col: str = "lng",
) -> pd.DataFrame:
    """DataFrame に JIS 3 次メッシュ列を追加する。"""
    if lat_col not in df.columns or lng_col not in df.columns:
        raise KeyError(f"{lat_col} and {lng_col} columns are required")

    assigned = df.copy()
    latitudes = pd.to_numeric(assigned[lat_col], errors="coerce")
    longitudes = pd.to_numeric(assigned[lng_col], errors="coerce")
    valid_mask = latitudes.notna() & longitudes.notna()

    assigned["jis_mesh3"] = pd.Series(pd.NA, index=assigned.index, dtype="string")
    assigned.loc[valid_mask, "jis_mesh3"] = [
        lat_lon_to_mesh3(lat, lon)
        for lat, lon in zip(latitudes.loc[valid_mask], longitudes.loc[valid_mask])
    ]
    return assigned


def assign_jis_mesh_quarter(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lng_col: str = "lng",
) -> pd.DataFrame:
    """DataFrame に JIS 4分の1メッシュ列を追加する。"""
    if lat_col not in df.columns or lng_col not in df.columns:
        raise KeyError(f"{lat_col} and {lng_col} columns are required")

    assigned = df.copy()
    latitudes = pd.to_numeric(assigned[lat_col], errors="coerce")
    longitudes = pd.to_numeric(assigned[lng_col], errors="coerce")
    valid_mask = latitudes.notna() & longitudes.notna()

    assigned["jis_mesh"] = pd.Series(pd.NA, index=assigned.index, dtype="string")
    assigned.loc[valid_mask, "jis_mesh"] = [
        lat_lon_to_mesh_quarter(lat, lon)
        for lat, lon in zip(latitudes.loc[valid_mask], longitudes.loc[valid_mask])
    ]
    return assigned
