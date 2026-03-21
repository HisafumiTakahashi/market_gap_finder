"""Feature engineering utilities for market opportunity analysis."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _neighbor_mesh_codes(mesh3: str) -> list[str]:
    """Return the 8 neighboring third-level mesh codes, wrapping across boundaries."""
    mesh3 = str(mesh3)
    if len(mesh3) != 8 or not mesh3.isdigit():
        return []

    p = int(mesh3[0:2])
    u = int(mesh3[2:4])
    q = int(mesh3[4])
    r = int(mesh3[5])
    s = int(mesh3[6])
    t = int(mesh3[7])
    neighbors = []

    for ds in (-1, 0, 1):
        for dt in (-1, 0, 1):
            if ds == 0 and dt == 0:
                continue

            ns, nt = s + ds, t + dt
            nq, nr = q, r
            np_, nu = p, u

            if ns < 0:
                ns = 9
                nq -= 1
            elif ns > 9:
                ns = 0
                nq += 1

            if nt < 0:
                nt = 9
                nr -= 1
            elif nt > 9:
                nt = 0
                nr += 1

            if nq < 0:
                nq = 7
                np_ -= 1
            elif nq > 7:
                nq = 0
                np_ += 1

            if nr < 0:
                nr = 7
                nu -= 1
            elif nr > 7:
                nr = 0
                nu += 1

            neighbors.append(f"{np_:02d}{nu:02d}{nq}{nr}{ns}{nt}")

    return neighbors


def add_genre_diversity(df: pd.DataFrame) -> pd.DataFrame:
    """Add the number of unique genres within each mesh."""
    if df.empty or "jis_mesh3" not in df.columns:
        df = df.copy()
        df["genre_diversity"] = 0
        return df

    mesh_genre_count = df.groupby("jis_mesh3")["unified_genre"].nunique().rename("genre_diversity")
    return df.merge(mesh_genre_count, on="jis_mesh3", how="left")


def add_genre_hhi(df: pd.DataFrame) -> pd.DataFrame:
    """Add Herfindahl-Hirschman Index by mesh."""
    if df.empty or "jis_mesh3" not in df.columns:
        df = df.copy()
        df["genre_hhi"] = 0.0
        return df

    mesh_total = df.groupby("jis_mesh3")["restaurant_count"].transform("sum")
    share = df["restaurant_count"] / mesh_total.replace(0, 1)
    df = df.copy()
    df["_share_sq"] = share**2
    hhi = df.groupby("jis_mesh3")["_share_sq"].sum().rename("genre_hhi")
    return df.drop(columns=["_share_sq"]).merge(hhi, on="jis_mesh3", how="left")


def add_other_genre_count(df: pd.DataFrame) -> pd.DataFrame:
    """Add the total restaurant count of all other genres within the mesh."""
    if df.empty or "jis_mesh3" not in df.columns:
        df = df.copy()
        df["other_genre_count"] = 0
        return df

    mesh_total = df.groupby("jis_mesh3")["restaurant_count"].transform("sum")
    df = df.copy()
    df["other_genre_count"] = mesh_total - df["restaurant_count"]
    return df


def add_neighbor_competition(df: pd.DataFrame) -> pd.DataFrame:
    """Add average restaurant count in neighboring meshes."""
    if df.empty or "jis_mesh3" not in df.columns:
        df = df.copy()
        df["neighbor_avg_restaurants"] = 0.0
        return df

    mesh_total = df.groupby("jis_mesh3")["restaurant_count"].sum().to_dict()

    def _calc_neighbor_avg(mesh3: str) -> float:
        neighbors = _neighbor_mesh_codes(mesh3)
        if not neighbors:
            return 0.0
        counts = [mesh_total.get(n, 0) for n in neighbors]
        return sum(counts) / len(counts)

    neighbor_map = {mesh: _calc_neighbor_avg(mesh) for mesh in df["jis_mesh3"].dropna().unique()}
    out = df.copy()
    out["neighbor_avg_restaurants"] = out["jis_mesh3"].map(neighbor_map).fillna(0.0)
    return out


def add_saturation_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add restaurants-per-population saturation index by mesh."""
    if df.empty or "jis_mesh3" not in df.columns:
        df = df.copy()
        df["saturation_index"] = 0.0
        return df

    mesh_agg = df.groupby("jis_mesh3").agg(
        total_restaurants=("restaurant_count", "sum"),
        population=("population", "first"),
    )
    pop = mesh_agg["population"].fillna(0).clip(lower=0)
    mesh_agg["saturation_index"] = mesh_agg["total_restaurants"] / (pop / 10000 + 0.1)
    return df.merge(mesh_agg["saturation_index"], on="jis_mesh3", how="left")


def add_nearest_station(df: pd.DataFrame, station_df: pd.DataFrame) -> pd.DataFrame:
    """Add nearest station distance and name for each row."""
    from src.collect.station import haversine_km

    if df.empty or station_df is None or station_df.empty:
        out = df.copy()
        out["nearest_station_distance"] = 0.0
        out["nearest_station_name"] = ""
        return out

    station_lats = station_df["lat"].values
    station_lngs = station_df["lng"].values
    station_names = station_df["station_name"].values

    distances: list[float] = []
    names: list[str] = []
    for _, row in df.iterrows():
        lat = float(row.get("lat", 0))
        lng = float(row.get("lng", 0))
        if lat == 0 or lng == 0:
            distances.append(0.0)
            names.append("")
            continue

        min_dist = float("inf")
        min_name = ""
        for s_lat, s_lng, s_name in zip(station_lats, station_lngs, station_names):
            distance = haversine_km(lat, lng, float(s_lat), float(s_lng))
            if distance < min_dist:
                min_dist = distance
                min_name = str(s_name)

        distances.append(min_dist)
        names.append(min_name)

    out = df.copy()
    out["nearest_station_distance"] = distances
    out["nearest_station_name"] = names
    return out


def add_land_price(df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
    """Map land price data to mesh level and merge into the analysis frame."""
    if df.empty or price_df is None or price_df.empty or "jis_mesh3" not in df.columns:
        out = df.copy()
        out["land_price"] = 0.0
        return out

    price_work = price_df.copy()
    price_work["lat"] = pd.to_numeric(price_work["lat"], errors="coerce")
    price_work["lng"] = pd.to_numeric(price_work["lng"], errors="coerce")
    price_work = price_work.dropna(subset=["lat", "lng"])

    from src.preprocess.mesh_converter import lat_lon_to_mesh3

    price_work["jis_mesh3"] = [
        lat_lon_to_mesh3(lat, lng) for lat, lng in zip(price_work["lat"], price_work["lng"])
    ]
    mesh_price = price_work.groupby("jis_mesh3")["price_per_sqm"].mean().rename("land_price")

    result = df.merge(mesh_price, on="jis_mesh3", how="left")
    result["land_price"] = result["land_price"].fillna(0.0)
    return result


def add_all_features(
    df: pd.DataFrame,
    station_df: pd.DataFrame | None = None,
    price_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    logger.info("Adding features to %d rows", len(df))
    out = df
    out = add_genre_diversity(out)
    out = add_genre_hhi(out)
    out = add_other_genre_count(out)
    out = add_neighbor_competition(out)
    out = add_saturation_index(out)
    if station_df is not None and not station_df.empty:
        out = add_nearest_station(out, station_df)
    if price_df is not None and not price_df.empty:
        out = add_land_price(out, price_df)
    logger.info("Feature engineering completed: %d -> %d columns", len(df.columns), len(out.columns))
    return out
