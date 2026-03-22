from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

LAT_KM_PER_DEGREE = 111.0
LNG_KM_PER_DEGREE = 91.0


def _to_projected_km(lat: pd.Series, lng: pd.Series) -> np.ndarray:
    return np.column_stack(
        [
            pd.to_numeric(lat, errors="coerce").to_numpy(dtype=float) * LAT_KM_PER_DEGREE,
            pd.to_numeric(lng, errors="coerce").to_numpy(dtype=float) * LNG_KM_PER_DEGREE,
        ]
    )


def match_google_to_hotpepper(
    hp_df: pd.DataFrame,
    gp_df: pd.DataFrame,
    max_distance_km: float = 0.1,
) -> pd.DataFrame:
    """Match each HotPepper row to the nearest Google Places row within range.

    One Google Places row may be matched to multiple HotPepper rows (1-to-many).
    """
    out = hp_df.copy()
    out["google_rating"] = np.nan
    out["google_review_count"] = np.nan
    out["google_match_distance"] = np.nan

    if out.empty or gp_df.empty:
        return out

    hp_coords = _to_projected_km(out.get("lat"), out.get("lng"))
    gp_work = gp_df.copy()
    gp_work["rating"] = pd.to_numeric(gp_work.get("rating"), errors="coerce")
    gp_work["review_count"] = pd.to_numeric(gp_work.get("review_count"), errors="coerce")
    gp_coords = _to_projected_km(gp_work.get("lat"), gp_work.get("lng"))

    valid_hp = np.isfinite(hp_coords).all(axis=1)
    valid_gp = np.isfinite(gp_coords).all(axis=1)
    if not valid_hp.any() or not valid_gp.any():
        return out

    hp_valid_idx = np.flatnonzero(valid_hp)
    gp_valid_idx = np.flatnonzero(valid_gp)
    hp_points = hp_coords[valid_hp]
    gp_points = gp_coords[valid_gp]

    tree = cKDTree(gp_points)
    distances, nearest_gp_pos = tree.query(hp_points, k=1)

    for i, (dist, gp_pos) in enumerate(zip(distances, nearest_gp_pos)):
        if dist > max_distance_km:
            continue
        hp_idx = hp_valid_idx[i]
        gp_idx = gp_valid_idx[gp_pos]
        hp_label = out.index[hp_idx]
        out.at[hp_label, "google_rating"] = gp_work.iloc[gp_idx]["rating"]
        out.at[hp_label, "google_review_count"] = gp_work.iloc[gp_idx]["review_count"]
        out.at[hp_label, "google_match_distance"] = float(dist)

    return out
