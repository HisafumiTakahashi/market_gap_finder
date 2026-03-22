from __future__ import annotations

import pandas as pd


def mesh_col(df: pd.DataFrame) -> str:
    """DataFrameからメッシュカラム名を検出する。"""
    for col in ("jis_mesh", "jis_mesh3", "mesh_code"):
        if col in df.columns:
            return col
    return "jis_mesh"
