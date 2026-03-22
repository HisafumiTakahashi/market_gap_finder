"""Feature engineering utilities for market opportunity analysis."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.analyze.constants import _COMPETITOR_OFFSET, _POP_UNIT
from src.analyze.utils import mesh_col as _mesh_col
from src.preprocess.mesh_converter import lat_lon_to_mesh_quarter

logger = logging.getLogger(__name__)


def _shift_mesh3(mesh3: str, lat_steps: int, lon_steps: int) -> str | None:
    """Shift an 8-digit mesh3 code by third-mesh steps."""
    code = str(mesh3)
    if len(code) != 8 or not code.isdigit():
        return None

    p = int(code[0:2])
    u = int(code[2:4])
    q = int(code[4])
    r = int(code[5])
    s = int(code[6])
    t = int(code[7])

    ns = s + lat_steps
    nt = t + lon_steps
    nq, nr = q, r
    np_, nu = p, u

    while ns < 0:
        ns += 10
        nq -= 1
    while ns > 9:
        ns -= 10
        nq += 1

    while nt < 0:
        nt += 10
        nr -= 1
    while nt > 9:
        nt -= 10
        nr += 1

    while nq < 0:
        nq += 8
        np_ -= 1
    while nq > 7:
        nq -= 8
        np_ += 1

    while nr < 0:
        nr += 8
        nu -= 1
    while nr > 7:
        nr -= 8
        nu += 1

    if np_ < 0 or nu < 0:
        return None
    return f"{np_:02d}{nu:02d}{nq}{nr}{ns}{nt}"


def _neighbor_mesh_codes_mesh3(mesh3: str) -> list[str]:
    """Return the 8 neighboring third-level mesh codes."""
    mesh3 = str(mesh3)
    if len(mesh3) != 8 or not mesh3.isdigit():
        return []

    neighbors: list[str] = []
    for ds in (-1, 0, 1):
        for dt in (-1, 0, 1):
            if ds == 0 and dt == 0:
                continue
            shifted = _shift_mesh3(mesh3, ds, dt)
            if shifted is not None:
                neighbors.append(shifted)
    return neighbors


def _neighbor_mesh_codes_quarter(mesh: str) -> list[str]:
    """Return the 8 neighboring quarter-mesh codes."""
    code = str(mesh)
    if len(code) != 10 or not code.isdigit():
        return []

    parent = code[:8]
    half_code = int(code[8])
    quarter_code = int(code[9])
    code_to_idx = {1: (0, 0), 2: (0, 1), 3: (1, 0), 4: (1, 1)}
    if half_code not in code_to_idx or quarter_code not in code_to_idx:
        return []

    idx_to_code = {value: key for key, value in code_to_idx.items()}
    h_lat, h_lon = code_to_idx[half_code]
    q_lat, q_lon = code_to_idx[quarter_code]
    abs_lat = h_lat * 2 + q_lat
    abs_lon = h_lon * 2 + q_lon
    neighbors: list[str] = []

    for dlat in (-1, 0, 1):
        for dlon in (-1, 0, 1):
            if dlat == 0 and dlon == 0:
                continue

            new_abs_lat = abs_lat + dlat
            new_abs_lon = abs_lon + dlon
            parent_lat_shift = 0
            parent_lon_shift = 0

            if new_abs_lat < 0:
                new_abs_lat += 4
                parent_lat_shift -= 1
            elif new_abs_lat > 3:
                new_abs_lat -= 4
                parent_lat_shift += 1

            if new_abs_lon < 0:
                new_abs_lon += 4
                parent_lon_shift -= 1
            elif new_abs_lon > 3:
                new_abs_lon -= 4
                parent_lon_shift += 1

            shifted_parent = _shift_mesh3(parent, parent_lat_shift, parent_lon_shift)
            if shifted_parent is None:
                continue

            new_h_lat = new_abs_lat // 2
            new_q_lat = new_abs_lat % 2
            new_h_lon = new_abs_lon // 2
            new_q_lon = new_abs_lon % 2

            new_half = idx_to_code[(new_h_lat, new_h_lon)]
            new_quarter = idx_to_code[(new_q_lat, new_q_lon)]
            neighbors.append(f"{shifted_parent}{new_half}{new_quarter}")

    return neighbors


def _neighbor_mesh_codes(mesh: str) -> list[str]:
    """Return the 8 neighboring mesh codes for mesh3 or quarter meshes."""
    mesh = str(mesh)
    if len(mesh) == 10 and mesh.isdigit():
        return _neighbor_mesh_codes_quarter(mesh)
    if len(mesh) == 8 and mesh.isdigit():
        return _neighbor_mesh_codes_mesh3(mesh)
    return []


def add_genre_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Add the number of unique genres within each mesh."""
    mesh_col = _mesh_col(df)
    if df.empty or mesh_col not in df.columns:
        df = df.copy()
        df["genre_diversity"] = 0
        return df

    mesh_genre_count = df.groupby(mesh_col)["unified_genre"].nunique().rename("genre_diversity")
    return df.merge(mesh_genre_count, on=mesh_col, how="left")


def add_genre_hhi(df: pd.DataFrame) -> pd.DataFrame:
    """Add Herfindahl-Hirschman Index by mesh."""
    mesh_col = _mesh_col(df)
    if df.empty or mesh_col not in df.columns:
        df = df.copy()
        df["genre_hhi"] = 0.0
        return df

    out = df.copy()
    rc = pd.to_numeric(out["restaurant_count"], errors="coerce").fillna(0)
    mesh_total = rc.groupby(out[mesh_col]).transform("sum")
    # replace 0 with 1 BEFORE division to avoid inf (0 total -> share=0)
    share = rc / mesh_total.replace(0, 1)
    share_sq = share**2
    mesh_hhi_sum = share_sq.groupby(out[mesh_col]).transform("sum")
    other_total = mesh_total - rc
    out["genre_hhi"] = mesh_hhi_sum
    out.loc[other_total == 0, "genre_hhi"] = 0.0
    return out


def add_other_genre_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add the total restaurant count of all other genres within the mesh."""
    mesh_col = _mesh_col(df)
    if df.empty or mesh_col not in df.columns:
        df = df.copy()
        df["other_genre_count"] = 0
        return df

    mesh_total = df.groupby(mesh_col)["restaurant_count"].transform("sum")
    df = df.copy()
    df["other_genre_count"] = mesh_total - df["restaurant_count"]
    return df


def add_genre_share(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-level genre share within each mesh."""
    mesh_col = _mesh_col(df)
    if df.empty or mesh_col not in df.columns:
        out = df.copy()
        out["genre_share"] = 0.0
        return out

    mesh_total = df.groupby(mesh_col)["restaurant_count"].transform("sum")
    out = df.copy()
    out["genre_share"] = out["restaurant_count"] / mesh_total.replace(0, 1)
    return out


def add_neighbor_competition(df: pd.DataFrame) -> pd.DataFrame:
    """Add average restaurant count in neighboring meshes."""
    mesh_col = _mesh_col(df)
    if df.empty or mesh_col not in df.columns:
        df = df.copy()
        df["neighbor_avg_restaurants"] = 0.0
        return df

    mesh_total = df.groupby(mesh_col)["restaurant_count"].sum().to_dict()

    def _calc_neighbor_avg(mesh_code: str) -> float:
        neighbors = _neighbor_mesh_codes(mesh_code)
        if not neighbors:
            return 0.0
        counts = [mesh_total.get(n, 0) for n in neighbors]
        return sum(counts) / len(counts)

    neighbor_map = {mesh: _calc_neighbor_avg(mesh) for mesh in df[mesh_col].dropna().unique()}
    out = df.copy()
    out["neighbor_avg_restaurants"] = out[mesh_col].map(neighbor_map).fillna(0.0)
    return out


def add_neighbor_population(df: pd.DataFrame) -> pd.DataFrame:
    """Add average population in neighboring meshes."""
    mesh_col = _mesh_col(df)
    if df.empty or mesh_col not in df.columns:
        out = df.copy()
        out["neighbor_avg_population"] = 0.0
        return out

    mesh_population = df.groupby(mesh_col)["population"].first().fillna(0).to_dict()

    def _calc_neighbor_avg(mesh_code: str) -> float:
        neighbors = _neighbor_mesh_codes(mesh_code)
        if not neighbors:
            return 0.0
        populations = [mesh_population.get(n, 0) for n in neighbors]
        return float(sum(populations) / len(populations))

    neighbor_map = {mesh: _calc_neighbor_avg(mesh) for mesh in df[mesh_col].dropna().unique()}
    out = df.copy()
    out["neighbor_avg_population"] = out[mesh_col].map(neighbor_map).fillna(0.0)
    return out


def add_demographic_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Add population and household ratio features with safe zero fallbacks."""
    out = df.copy()
    metric_defaults = {
        "population": 0.0,
        "pop_working": 0.0,
        "pop_adult": 0.0,
        "pop_elderly": 0.0,
        "households": 0.0,
        "single_households": 0.0,
        "young_single": 0.0,
    }
    for column, default in metric_defaults.items():
        if column not in out.columns:
            out[column] = default
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(default)

    out["working_ratio"] = out["pop_working"] / (out["population"] + 1.0)
    out["elderly_ratio"] = out["pop_elderly"] / (out["population"] + 1.0)
    out["single_ratio"] = out["single_households"] / (out["households"] + 1.0)
    out["young_single_ratio"] = out["young_single"] / (out["households"] + 1.0)
    return out


def add_neighbor_avg_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add average neighboring opportunity score when available."""
    mesh_col = _mesh_col(df)
    out = df.copy()
    if out.empty or mesh_col not in out.columns:
        out["neighbor_avg_score"] = 0.0
        return out
    if "opportunity_score" not in out.columns:
        out["neighbor_avg_score"] = 0.0
        return out

    mesh_score = pd.to_numeric(out.groupby(mesh_col)["opportunity_score"].first(), errors="coerce").fillna(0.0).to_dict()

    def _calc_neighbor_avg(mesh_code: str) -> float:
        neighbors = _neighbor_mesh_codes(mesh_code)
        if not neighbors:
            return 0.0
        scores = [mesh_score.get(n, 0.0) for n in neighbors]
        return float(sum(scores) / len(scores))

    neighbor_map = {mesh: _calc_neighbor_avg(mesh) for mesh in out[mesh_col].dropna().unique()}
    out["neighbor_avg_score"] = out[mesh_col].map(neighbor_map).fillna(0.0)
    return out


def add_saturation_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add restaurants-per-population saturation index by mesh (LOO)."""
    mesh_col = _mesh_col(df)
    if df.empty or mesh_col not in df.columns:
        df = df.copy()
        df["saturation_index"] = 0.0
        return df

    out = df.copy()
    mesh_total = out.groupby(mesh_col)["restaurant_count"].transform("sum")
    other_total = mesh_total - out["restaurant_count"]
    pop = (
        pd.to_numeric(out.groupby(mesh_col)["population"].transform("first"), errors="coerce")
        .fillna(0)
        .clip(lower=0)
    )
    out["saturation_index"] = other_total / (pop / _POP_UNIT + _COMPETITOR_OFFSET)
    return out


def add_genre_saturation(df: pd.DataFrame) -> pd.DataFrame:
    """Add genre-level population-normalized saturation."""
    if df.empty or "population" not in df.columns:
        out = df.copy()
        out["genre_saturation"] = 0.0
        return out

    out = df.copy()
    population = pd.to_numeric(out["population"], errors="coerce").fillna(0).clip(lower=0)
    restaurant_count = pd.to_numeric(out["restaurant_count"], errors="coerce").fillna(0)
    out["genre_saturation"] = restaurant_count / (population / _POP_UNIT + _COMPETITOR_OFFSET)
    return out


def add_nearest_station(df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    """Add nearest station distance and name for each row."""
    from scipy.spatial import cKDTree

    from src.collect.station import haversine_km

    if df.empty or station_df is None or station_df.empty:
        out = df.copy()
        out["nearest_station_distance"] = 0.0
        out["nearest_station_name"] = ""
        return out

    out = df.copy()
    lats = pd.to_numeric(out["lat"], errors="coerce").fillna(0).values
    lngs = pd.to_numeric(out["lng"], errors="coerce").fillna(0).values
    valid = (lats != 0) & (lngs != 0)

    station_lats = pd.to_numeric(station_df["lat"], errors="coerce").values
    station_lngs = pd.to_numeric(station_df["lng"], errors="coerce").values
    station_names = station_df["station_name"].values
    station_valid = ~(pd.isna(station_lats) | pd.isna(station_lngs))
    if not station_valid.any():
        out["nearest_station_distance"] = 0.0
        out["nearest_station_name"] = ""
        return out

    station_lats = station_lats[station_valid]
    station_lngs = station_lngs[station_valid]
    station_names = station_names[station_valid]

    tree = cKDTree(np.column_stack([station_lats, station_lngs]))

    distances = np.zeros(len(out))
    names = [""] * len(out)

    if valid.any():
        query_points = np.column_stack([lats[valid], lngs[valid]])
        _, indices = tree.query(query_points, k=1)
        for idx, orig_idx in zip(np.atleast_1d(indices), np.where(valid)[0]):
            station_idx = int(idx)
            distances[orig_idx] = haversine_km(
                lats[orig_idx], lngs[orig_idx], station_lats[station_idx], station_lngs[station_idx]
            )
            names[orig_idx] = str(station_names[station_idx])

    out["nearest_station_distance"] = distances
    out["nearest_station_name"] = names
    return out


def add_nearest_station_passengers(df: pd.DataFrame, passenger_df: pd.DataFrame) -> pd.DataFrame:
    """Add nearest station passenger volume."""
    from scipy.spatial import cKDTree

    out = df.copy()
    if out.empty or passenger_df is None or passenger_df.empty:
        out["nearest_station_passengers"] = 0.0
        return out

    passenger_work = passenger_df.copy()
    passenger_work["passengers"] = pd.to_numeric(passenger_work.get("passengers"), errors="coerce").fillna(0.0)

    if "nearest_station_name" in out.columns and "station_name" in passenger_work.columns:
        station_passenger_map = passenger_work.groupby("station_name")["passengers"].max().to_dict()
        matched = out["nearest_station_name"].map(station_passenger_map)
        if matched.notna().any():
            out["nearest_station_passengers"] = matched.fillna(0.0)
            return out

    lats = pd.to_numeric(out.get("lat"), errors="coerce").fillna(0).values
    lngs = pd.to_numeric(out.get("lng"), errors="coerce").fillna(0).values
    valid = (lats != 0) & (lngs != 0)

    passenger_lats = pd.to_numeric(passenger_work["lat"], errors="coerce").values
    passenger_lngs = pd.to_numeric(passenger_work["lng"], errors="coerce").values
    passenger_counts = passenger_work["passengers"].values
    station_valid = ~(pd.isna(passenger_lats) | pd.isna(passenger_lngs))
    if not station_valid.any():
        out["nearest_station_passengers"] = 0.0
        return out

    passenger_lats = passenger_lats[station_valid]
    passenger_lngs = passenger_lngs[station_valid]
    passenger_counts = passenger_counts[station_valid]

    tree = cKDTree(np.column_stack([passenger_lats, passenger_lngs]))
    values = np.zeros(len(out))

    if valid.any():
        query_points = np.column_stack([lats[valid], lngs[valid]])
        _, indices = tree.query(query_points, k=1)
        for idx, orig_idx in zip(np.atleast_1d(indices), np.where(valid)[0]):
            values[orig_idx] = float(passenger_counts[int(idx)])

    out["nearest_station_passengers"] = values
    return out


def add_land_price(df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Map land price data to mesh level and merge into the analysis frame."""
    mesh_col = _mesh_col(df)
    if df.empty or price_df is None or price_df.empty or mesh_col not in df.columns:
        out = df.copy()
        out["land_price"] = 0.0
        return out

    price_work = price_df.copy()
    price_work["lat"] = pd.to_numeric(price_work["lat"], errors="coerce")
    price_work["lng"] = pd.to_numeric(price_work["lng"], errors="coerce")
    price_work = price_work.dropna(subset=["lat", "lng"])
    price_work["jis_mesh"] = [
        lat_lon_to_mesh_quarter(lat, lng) for lat, lng in zip(price_work["lat"], price_work["lng"])
    ]
    mesh_price = price_work.groupby("jis_mesh")["price_per_sqm"].mean().rename("land_price")

    result = df.merge(mesh_price, left_on=mesh_col, right_index=True, how="left")
    result["land_price"] = result["land_price"].fillna(0.0)
    return result


def add_all_features(
    df: pd.DataFrame,
    station_df: pd.DataFrame | None = None,
    passenger_df: pd.DataFrame | None = None,
    price_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    logger.info("Adding features to %d rows", len(df))
    out = df
    out = add_genre_diversity(out)
    out = add_genre_hhi(out)
    out = add_other_genre_count(out)
    out = add_genre_share(out)
    out = add_neighbor_competition(out)
    out = add_neighbor_population(out)
    out = add_demographic_ratios(out)
    out = add_saturation_index(out)
    out = add_genre_saturation(out)
    if station_df is not None and not station_df.empty:
        out = add_nearest_station(out, station_df)
    if passenger_df is not None:
        out = add_nearest_station_passengers(out, passenger_df)
    if price_df is not None and not price_df.empty:
        out = add_land_price(out, price_df)
    logger.info("Feature engineering completed: %d -> %d columns", len(df.columns), len(out.columns))
    return out
