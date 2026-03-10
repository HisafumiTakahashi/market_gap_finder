"""市場ギャップ可視化用の地図と Dash アプリを提供するモジュール。"""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import pandas as pd
from dash import Dash, Input, Output, dcc, html
from folium.plugins import HeatMap

from config import settings
from src.analyze.scoring import compute_opportunity_score

logger = logging.getLogger(__name__)

DEFAULT_CENTER_LAT = 35.6812
DEFAULT_CENTER_LNG = 139.7671
DEFAULT_ZOOM_START = 12


def _coerce_map_frame(df: pd.DataFrame) -> pd.DataFrame:
    """地図描画用に緯度経度列を数値化し、欠損行を除外する。

    Args:
        df: `lat` と `lng` を含む可能性がある DataFrame。

    Returns:
        緯度経度を数値化し、描画不能な行を除いた DataFrame のコピー。
    """
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
    """DataFrame の平均座標から地図中心を決定する。

    Args:
        df: 描画対象の DataFrame。
        fallback_lat: データが空の場合に使う緯度。
        fallback_lng: データが空の場合に使う経度。

    Returns:
        地図中心として使う緯度・経度タプル。
    """
    if df.empty:
        return fallback_lat, fallback_lng

    return float(df["lat"].mean()), float(df["lng"].mean())


def create_score_heatmap(
    df: pd.DataFrame,
    center_lat: float = DEFAULT_CENTER_LAT,
    center_lng: float = DEFAULT_CENTER_LNG,
    zoom_start: int = DEFAULT_ZOOM_START,
) -> folium.Map:
    """機会スコアを重みとしたヒートマップを生成する。

    `opportunity_score` 列が存在しない場合はすべての地点を同一重みで描画する。

    Args:
        df: 描画対象 DataFrame。
        center_lat: 初期中心緯度。
        center_lng: 初期中心経度。
        zoom_start: 初期ズームレベル。

    Returns:
        生成した `folium.Map` オブジェクト。
    """
    map_obj = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    plot_df = _coerce_map_frame(df)
    if plot_df.empty:
        return map_obj

    if "opportunity_score" in plot_df.columns:
        weights = pd.to_numeric(
            plot_df["opportunity_score"], errors="coerce"
        ).fillna(1.0)
    else:
        weights = pd.Series(1.0, index=plot_df.index, dtype=float)

    heat_data = [
        [float(row.lat), float(row.lng), float(weights.loc[idx])]
        for idx, row in plot_df.iterrows()
    ]
    if heat_data:
        HeatMap(heat_data, radius=18, blur=14, min_opacity=0.25).add_to(map_obj)

    return map_obj


def create_marker_map(df: pd.DataFrame, top_n: int = 20) -> folium.Map:
    """上位候補にマーカーを配置した地図を生成する。

    機会スコアの高い順に最大 `top_n` 件を抽出し、ポップアップに主要指標を表示する。

    Args:
        df: 描画対象 DataFrame。
        top_n: マーカー表示する上位件数。

    Returns:
        生成した `folium.Map` オブジェクト。
    """
    plot_df = _coerce_map_frame(df)
    if "opportunity_score" not in plot_df.columns:
        plot_df = plot_df.copy()
        plot_df["opportunity_score"] = 0.0

    if plot_df.empty:
        center_lat, center_lng = DEFAULT_CENTER_LAT, DEFAULT_CENTER_LNG
    else:
        plot_df = plot_df.sort_values("opportunity_score", ascending=False).head(
            max(1, int(top_n))
        )
        center_lat, center_lng = _resolve_map_center(plot_df)

    map_obj = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=DEFAULT_ZOOM_START,
        tiles="CartoDB positron",
    )

    if plot_df.empty:
        return map_obj

    for _, row in plot_df.iterrows():
        mesh_code = row.get("mesh_code", "")
        unified_genre = row.get("unified_genre", "")
        score = float(
            pd.to_numeric(pd.Series([row.get("opportunity_score", 0.0)]), errors="coerce")
            .fillna(0.0)
            .iloc[0]
        )
        competitor_density = int(
            pd.to_numeric(
                pd.Series(
                    [row.get("competitor_density", row.get("restaurant_count", 0))]
                ),
                errors="coerce",
            )
            .fillna(0)
            .iloc[0]
        )
        demand_score = float(
            pd.to_numeric(pd.Series([row.get("demand_score", 0.0)]), errors="coerce")
            .fillna(0.0)
            .iloc[0]
        )

        popup_lines = [
            f"<b>{mesh_code} &times; {unified_genre}</b>",
            f"Score: {score:.3f}",
            f"Competitors: {competitor_density}",
            f"Demand: {demand_score:.3f}",
        ]
        if "reason" in plot_df.columns and pd.notna(row.get("reason")):
            popup_lines.append(f"Reason: {row['reason']}")

        folium.Marker(
            location=[float(row["lat"]), float(row["lng"])],
            popup=folium.Popup("<br>".join(popup_lines), max_width=320),
            tooltip=f"{mesh_code} x {unified_genre}",
        ).add_to(map_obj)

    return map_obj


def save_map(map_obj: folium.Map, filename: str) -> None:
    """`folium.Map` を HTML ファイルとして保存する。

    Args:
        map_obj: 保存対象の地図オブジェクト。
        filename: 出力先 HTML ファイル名またはパス。
    """
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(output_path))
    logger.info("Saved map HTML to %s", output_path)


def _load_and_score(tag: str = "result", top_n: int = 20) -> pd.DataFrame:
    """集計済み CSV を読み込み、可視化用のスコア列を付与する。

    Args:
        tag: 対象 CSV のタグ。
        top_n: 呼び出し互換性のため受け取るが、この関数内では使用しない。

    Returns:
        スコアリング済み DataFrame。読み込みや計算に失敗した場合は空 DataFrame。
    """
    del top_n

    csv_path = settings.PROCESSED_DATA_DIR / f"{tag}_aggregated.csv"
    try:
        if not csv_path.exists():
            return pd.DataFrame()

        aggregated_df = pd.read_csv(csv_path)
        return compute_opportunity_score(aggregated_df)
    except Exception:
        logger.exception("Failed to load or score aggregated data: %s", csv_path)
        return pd.DataFrame()


def build_dash_app(tag: str = "result") -> object:
    """市場ギャップ可視化用の Dash アプリを構築する。

    Args:
        tag: 読み込む集計済み CSV のタグ。

    Returns:
        レイアウトとコールバックを設定済みの Dash アプリケーション。
    """
    data = _load_and_score(tag=tag)
    genre_options = [{"label": "All", "value": "__all__"}]
    if "unified_genre" in data.columns:
        genre_values = (
            data["unified_genre"].dropna().astype(str).sort_values().unique().tolist()
        )
        genre_options.extend(
            {"label": genre_name, "value": genre_name} for genre_name in genre_values
        )

    app = Dash(__name__)
    app.layout = html.Div(
        [
            html.H2("Market Gap Finder Dashboard"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Genre"),
                            dcc.Dropdown(
                                id="genre-filter",
                                options=genre_options,
                                value="__all__",
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "220px"},
                    ),
                    html.Div(
                        [
                            html.Label("Map Mode"),
                            dcc.Dropdown(
                                id="map-mode",
                                options=[
                                    {"label": "Heatmap", "value": "heatmap"},
                                    {"label": "Markers", "value": "markers"},
                                ],
                                value="heatmap",
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "180px"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    html.Label("Top N"),
                    dcc.Slider(
                        id="top-n",
                        min=5,
                        max=50,
                        step=5,
                        value=20,
                        marks={5: "5", 20: "20", 35: "35", 50: "50"},
                    ),
                ],
                style={"marginTop": "20px"},
            ),
            html.Iframe(
                id="map-frame",
                style={"width": "100%", "height": "75vh", "border": "none", "marginTop": "20px"},
            ),
        ],
        style={"padding": "16px"},
    )

    @app.callback(
        Output("map-frame", "srcDoc"),
        Input("genre-filter", "value"),
        Input("top-n", "value"),
        Input("map-mode", "value"),
    )
    def _update_map(selected_genre: str, top_n: int, map_mode: str) -> str:
        """UI の選択状態に応じて描画用地図 HTML を更新する。"""
        filtered_df = data.copy()
        if (
            selected_genre
            and selected_genre != "__all__"
            and "unified_genre" in filtered_df.columns
        ):
            filtered_df = filtered_df[
                filtered_df["unified_genre"].astype(str) == selected_genre
            ]

        if map_mode == "markers":
            map_obj = create_marker_map(filtered_df, top_n=top_n or 20)
        else:
            center_lat, center_lng = _resolve_map_center(_coerce_map_frame(filtered_df))
            map_obj = create_score_heatmap(
                filtered_df,
                center_lat=center_lat,
                center_lng=center_lng,
                zoom_start=DEFAULT_ZOOM_START,
            )

        return map_obj.get_root().render()

    return app


def run_dashboard(host: str = "127.0.0.1", port: int = 8050) -> None:
    """Dash ダッシュボードを起動する。

    Args:
        host: バインドするホスト名または IP アドレス。
        port: リッスンするポート番号。
    """
    app = build_dash_app()
    if hasattr(app, "run_server"):
        app.run_server(host=host, port=port, debug=False)
    else:
        app.run(host=host, port=port, debug=False)
