"""Dashboard and map helpers for visualizing market gaps."""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import pandas as pd
from dash import Dash, Input, Output, dcc, html
from folium.plugins import HeatMap

from config import settings
from src.analyze.features import add_all_features
from src.analyze.utils import mesh_col as _mesh_col
from src.analyze.scoring import (
    compute_opportunity_score,
    compute_opportunity_score_v2,
    compute_opportunity_score_v3b,
    compute_opportunity_score_v4,
    generate_reason,
)

logger = logging.getLogger(__name__)

DEFAULT_CENTER_LAT = 35.6812
DEFAULT_CENTER_LNG = 139.7671
DEFAULT_ZOOM_START = 12

_AREA_LABELS: dict[str, str] = {
    "tokyo": "東京",
    "osaka": "大阪",
    "nagoya": "名古屋",
    "fukuoka": "福岡",
    "sapporo": "札幌",
}

_GENRE_LABELS: dict[str, str] = {
    "izakaya": "居酒屋",
    "italian": "イタリアン",
    "chinese": "中華",
    "yakiniku": "焼肉",
    "cafe": "カフェ",
    "ramen": "ラーメン",
    "washoku": "和食",
    "curry": "カレー",
    "other": "その他",
}


def _genre_label(genre: str) -> str:
    return _GENRE_LABELS.get(genre, genre)


def _coerce_map_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    work["lat"] = pd.to_numeric(work.get("lat"), errors="coerce")
    work["lng"] = pd.to_numeric(work.get("lng"), errors="coerce")
    return work.dropna(subset=["lat", "lng"])


def _resolve_map_center(
    df: pd.DataFrame,
    fallback_lat: float = DEFAULT_CENTER_LAT,
    fallback_lng: float = DEFAULT_CENTER_LNG,
) -> tuple[float, float]:
    if df.empty:
        return fallback_lat, fallback_lng
    return float(df["lat"].mean()), float(df["lng"].mean())


def create_score_heatmap(
    df: pd.DataFrame,
    center_lat: float = DEFAULT_CENTER_LAT,
    center_lng: float = DEFAULT_CENTER_LNG,
    zoom_start: int = DEFAULT_ZOOM_START,
) -> folium.Map:
    map_obj = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_start, tiles="CartoDB positron")
    plot_df = _coerce_map_frame(df)
    if plot_df.empty:
        return map_obj
    weights = pd.to_numeric(plot_df["opportunity_score"], errors="coerce").fillna(1.0) if "opportunity_score" in plot_df.columns else pd.Series(1.0, index=plot_df.index, dtype=float)
    heat_data = [[float(row.lat), float(row.lng), float(weights.loc[idx])] for idx, row in plot_df.iterrows()]
    if heat_data:
        HeatMap(heat_data, radius=18, blur=14, min_opacity=0.25).add_to(map_obj)
    return map_obj


def create_population_heatmap(
    df: pd.DataFrame,
    center_lat: float = DEFAULT_CENTER_LAT,
    center_lng: float = DEFAULT_CENTER_LNG,
    zoom_start: int = DEFAULT_ZOOM_START,
) -> folium.Map:
    map_obj = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_start, tiles="CartoDB positron")
    plot_df = _coerce_map_frame(df)
    if plot_df.empty or "population" not in plot_df.columns:
        return map_obj
    population = pd.to_numeric(plot_df["population"], errors="coerce").fillna(0.0)
    mesh_col = _mesh_col(plot_df)
    deduped = plot_df.assign(_pop=population).drop_duplicates(subset=[mesh_col] if mesh_col in plot_df.columns else ["lat", "lng"])
    heat_data = [[float(row.lat), float(row.lng), float(row._pop)] for row in deduped.itertuples() if row._pop > 0]
    if heat_data:
        HeatMap(
            heat_data,
            radius=20,
            blur=16,
            min_opacity=0.3,
            gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
        ).add_to(map_obj)
    return map_obj


def create_ml_gap_heatmap(
    df: pd.DataFrame,
    center_lat: float = DEFAULT_CENTER_LAT,
    center_lng: float = DEFAULT_CENTER_LNG,
    zoom_start: int = DEFAULT_ZOOM_START,
) -> folium.Map:
    map_obj = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_start, tiles="CartoDB positron")
    plot_df = _coerce_map_frame(df)
    if plot_df.empty or "market_gap" not in plot_df.columns:
        return map_obj
    gap = pd.to_numeric(plot_df["market_gap"], errors="coerce").fillna(0.0)
    positive = plot_df.assign(_gap=gap)
    positive = positive[positive["_gap"] > 0]
    heat_data = [[float(row.lat), float(row.lng), float(row._gap)] for row in positive.itertuples()]
    if heat_data:
        HeatMap(heat_data, radius=18, blur=14, min_opacity=0.25).add_to(map_obj)
    return map_obj


def create_marker_map(df: pd.DataFrame, top_n: int = 20) -> folium.Map:
    plot_df = _coerce_map_frame(df)
    if "opportunity_score" not in plot_df.columns:
        plot_df = plot_df.copy()
        plot_df["opportunity_score"] = 0.0

    if plot_df.empty:
        center_lat, center_lng = DEFAULT_CENTER_LAT, DEFAULT_CENTER_LNG
    else:
        plot_df = plot_df.sort_values("opportunity_score", ascending=False).head(max(1, int(top_n)))
        center_lat, center_lng = _resolve_map_center(plot_df)

    map_obj = folium.Map(location=[center_lat, center_lng], zoom_start=DEFAULT_ZOOM_START, tiles="CartoDB positron")
    if plot_df.empty:
        return map_obj

    for rank, (_, row) in enumerate(plot_df.iterrows(), 1):
        unified_genre = str(row.get("unified_genre", "") or "")
        genre_ja = _genre_label(unified_genre)
        station_name = str(row.get("nearest_station_name", "") or "")
        location_label = f"{station_name}駅付近" if station_name else ""
        display_name = f"#{rank} {genre_ja}"
        if location_label:
            display_name += f"（{location_label}）"

        score = float(pd.to_numeric(pd.Series([row.get("opportunity_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        restaurant_count = int(pd.to_numeric(pd.Series([row.get("restaurant_count", 0)]), errors="coerce").fillna(0).iloc[0])
        population = int(pd.to_numeric(pd.Series([row.get("population", 0)]), errors="coerce").fillna(0).iloc[0])

        popup_lines = [
            f"<b>{display_name}</b>",
            f"スコア: {score:.3f}",
            f"人口: {population:,}",
            f"競合店舗数: {restaurant_count}",
        ]
        if "predicted_count" in plot_df.columns and pd.notna(row.get("predicted_count")):
            predicted_count = pd.to_numeric(pd.Series([row.get("predicted_count", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
            popup_lines.append(f"予測店舗数: {predicted_count:.1f}")
        if "gap_count" in plot_df.columns and pd.notna(row.get("gap_count")):
            gap_count = pd.to_numeric(pd.Series([row.get("gap_count", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
            popup_lines.append(f"ギャップ: {gap_count:.1f}")

        for col, label, fmt in (
            ("genre_diversity", "ジャンル多様性", "d"),
            ("genre_hhi", "ジャンルHHI", ".3f"),
            ("neighbor_avg_restaurants", "近隣平均店舗数", ".1f"),
            ("saturation_index", "飽和度", ".2f"),
        ):
            if col in plot_df.columns and pd.notna(row.get(col)):
                val = pd.to_numeric(pd.Series([row[col]]), errors="coerce").fillna(0).iloc[0]
                popup_lines.append(f"{label}: {val:{fmt}}")
        if "reason" in plot_df.columns and pd.notna(row.get("reason")):
            popup_lines.append(f"理由: {row['reason']}")

        lat_val = float(row["lat"])
        lng_val = float(row["lng"])
        gmap_url = f"https://www.google.com/maps?q={lat_val},{lng_val}"
        popup_lines.append(f'<a href="{gmap_url}" target="_blank">Google Mapで開く</a>')

        folium.Marker(
            location=[lat_val, lng_val],
            popup=folium.Popup("<br>".join(popup_lines), max_width=320),
            tooltip=display_name,
        ).add_to(map_obj)

    return map_obj


def save_map(map_obj: folium.Map, filename: str) -> None:
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_path))
    logger.info("Saved map HTML to %s", output_path)


def _detect_available_tags() -> list[str]:
    """data/processed/ 内の *_integrated.csv からタグを自動検出。"""
    processed_dir = settings.PROCESSED_DATA_DIR
    tags = []
    for csv_path in sorted(processed_dir.glob("*_integrated.csv")):
        tag = csv_path.stem.replace("_integrated", "")
        if tag != "combined":
            tags.append(tag)
    return tags


def _refresh_reason(df: pd.DataFrame) -> pd.DataFrame:
    """Re-generate the reason column using the latest generate_reason logic."""
    if df.empty:
        return df
    df = df.copy()
    df["reason"] = df.apply(generate_reason, axis=1)
    return df


def _load_and_score(tag: str = "result", version: str = "v4") -> pd.DataFrame:
    csv_path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            if version == "v1":
                return compute_opportunity_score(df)
            if version == "v2":
                return compute_opportunity_score_v2(df)
            if version == "v3":
                return compute_opportunity_score_v3b(add_all_features(df))
            # v4
            df = add_all_features(df)
            ml_gap_path = settings.PROCESSED_DATA_DIR / f"{tag}_ml_gap.csv"
            if ml_gap_path.exists():
                ml_df = pd.read_csv(ml_gap_path)
                ml_mesh_col = _mesh_col(ml_df)
                merge_cols = [col for col in (ml_mesh_col, "unified_genre", "market_gap", "predicted_count", "gap_count") if col in ml_df.columns]
                df_mesh_col = _mesh_col(df)
                if {ml_mesh_col, "unified_genre", "market_gap"}.issubset(merge_cols) and df_mesh_col in df.columns:
                    df = df.merge(ml_df[merge_cols], left_on=[df_mesh_col, "unified_genre"], right_on=[ml_mesh_col, "unified_genre"], how="left")
                    return compute_opportunity_score_v4(df, ml_gap=df["market_gap"])
            return compute_opportunity_score_v4(df)
        except Exception:
            logger.exception("Failed to load integrated data: %s", csv_path)

    csv_path = settings.PROCESSED_DATA_DIR / f"{tag}_aggregated.csv"
    try:
        if not csv_path.exists():
            return pd.DataFrame()
        aggregated_df = pd.read_csv(csv_path)
        if version == "v1":
            return compute_opportunity_score(aggregated_df)
        if version == "v2":
            return compute_opportunity_score_v2(aggregated_df)
        if version == "v3":
            return compute_opportunity_score_v3b(add_all_features(aggregated_df))
        return compute_opportunity_score_v4(add_all_features(aggregated_df))
    except Exception:
        logger.exception("Failed to load or score aggregated data: %s", csv_path)
        return pd.DataFrame()


def build_dash_app(tag: str = "result", available_tags: list[str] | None = None) -> object:
    available_tags = available_tags or _detect_available_tags()
    selected_tag = tag if tag in available_tags else (available_tags[0] if available_tags else tag)
    data_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def _get_data(area_tag: str, version: str) -> pd.DataFrame:
        key = (area_tag, version)
        if key not in data_cache:
            data_cache[key] = _refresh_reason(_load_and_score(tag=area_tag, version=version))
        return data_cache[key]

    init_data = _get_data(selected_tag, "v4")
    genre_options = [{"label": "全ジャンル", "value": "__all__"}]
    if "unified_genre" in init_data.columns:
        genre_values = init_data["unified_genre"].dropna().astype(str).sort_values().unique().tolist()
        genre_options.extend({"label": _genre_label(g), "value": g} for g in genre_values)

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Market Gap Finder ダッシュボード"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("エリア"),
                            dcc.Dropdown(
                                id="area-selector",
                                options=[{"label": _AREA_LABELS.get(area_tag, area_tag), "value": area_tag} for area_tag in available_tags],
                                value=selected_tag,
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label("スコアバージョン"),
                            dcc.Dropdown(
                                id="score-version",
                                options=[
                                    {"label": "v4 (ML+v3bアンサンブル)", "value": "v4"},
                                    {"label": "v3b (特徴量強化)", "value": "v3"},
                                    {"label": "v2 (人口ベース)", "value": "v2"},
                                    {"label": "v1 (基本ベース)", "value": "v1"},
                                ],
                                value="v4",
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "180px"},
                    ),
                    html.Div(
                        [
                            html.Label("ジャンル"),
                            dcc.Dropdown(id="genre-filter", options=genre_options, value="__all__", clearable=False),
                        ],
                        style={"flex": "1", "minWidth": "220px"},
                    ),
                    html.Div(
                        [
                            html.Label("地図モード"),
                            dcc.Dropdown(
                                id="map-mode",
                                options=[
                                    {"label": "スコアヒートマップ", "value": "heatmap"},
                                    {"label": "MLギャップヒートマップ", "value": "ml_gap"},
                                    {"label": "人口ヒートマップ", "value": "population"},
                                    {"label": "マーカー", "value": "markers"},
                                ],
                                value="heatmap",
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "200px"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Label("表示件数"),
                    dcc.Slider(id="top-n", min=5, max=50, step=5, value=20, marks={5: "5", 20: "20", 35: "35", 50: "50"}),
                ],
                style={"marginTop": "20px"},
            ),
            html.Iframe(id="map-frame", style={"width": "100%", "height": "75vh", "border": "none", "marginTop": "20px"}),
        ],
        style={"padding": "16px"},
    )

    @app.callback(
        Output("genre-filter", "options"),
        Output("genre-filter", "value"),
        Input("area-selector", "value"),
    )
    def _update_genre_options(area_tag: str) -> tuple[list[dict], str]:
        df = _get_data(area_tag or tag, "v4")
        options = [{"label": "全ジャンル", "value": "__all__"}]
        if "unified_genre" in df.columns:
            for g in sorted(df["unified_genre"].dropna().astype(str).unique()):
                options.append({"label": _genre_label(g), "value": g})
        return options, "__all__"

    @app.callback(
        Output("map-frame", "srcDoc"),
        Input("area-selector", "value"),
        Input("score-version", "value"),
        Input("genre-filter", "value"),
        Input("top-n", "value"),
        Input("map-mode", "value"),
    )
    def _update_map(area_tag: str, score_version: str, selected_genre: str, top_n: int, map_mode: str) -> str:
        filtered_df = _get_data(area_tag or tag, score_version or "v2").copy()
        if selected_genre and selected_genre != "__all__" and "unified_genre" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["unified_genre"].astype(str) == selected_genre]

        center_lat, center_lng = _resolve_map_center(_coerce_map_frame(filtered_df))
        if map_mode == "markers":
            map_obj = create_marker_map(filtered_df, top_n=top_n or 20)
        elif map_mode == "population":
            map_obj = create_population_heatmap(filtered_df, center_lat=center_lat, center_lng=center_lng, zoom_start=DEFAULT_ZOOM_START)
        elif map_mode == "ml_gap":
            map_obj = create_ml_gap_heatmap(filtered_df, center_lat=center_lat, center_lng=center_lng, zoom_start=DEFAULT_ZOOM_START)
        else:
            map_obj = create_score_heatmap(filtered_df, center_lat=center_lat, center_lng=center_lng, zoom_start=DEFAULT_ZOOM_START)
        return map_obj.get_root().render()

    return app


def run_dashboard(tag: str = "result", host: str = "127.0.0.1", port: int = 8050) -> None:
    app = build_dash_app(tag=tag)
    app.run(host=host, port=port, debug=False)
