"""
ダッシュボードモジュール

Folium による地図可視化と Plotly Dash による
インタラクティブダッシュボードを提供する。

機能
----
- ヒートマップ: メッシュ単位の機会スコアを地図上に可視化
- ランキングテーブル: 出店候補の詳細一覧
- フィルタ: ジャンル・エリアで絞り込み
"""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import pandas as pd
from folium.plugins import HeatMap

from config import settings

logger = logging.getLogger(__name__)


def create_score_heatmap(
    df: pd.DataFrame,
    center_lat: float = 35.6812,
    center_lng: float = 139.7671,
    zoom_start: int = 12,
) -> folium.Map:
    """機会スコアのヒートマップを作成する。

    Parameters
    ----------
    df : pd.DataFrame
        opportunity_score 付きメッシュデータ
    center_lat : float, optional
        地図中心の緯度, デフォルト 東京駅
    center_lng : float, optional
        地図中心の経度
    zoom_start : int, optional
        初期ズームレベル

    Returns
    -------
    folium.Map
        ヒートマップ付き Folium 地図オブジェクト
    """
    map_obj = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=zoom_start,
        tiles="CartoDB positron",
    )

    if df.empty or not {"lat", "lng"}.issubset(df.columns):
        return map_obj

    work = df.copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lng"] = pd.to_numeric(work["lng"], errors="coerce")
    work = work.dropna(subset=["lat", "lng"])

    if work.empty:
        return map_obj

    if "opportunity_score" in work.columns:
        score = pd.to_numeric(work["opportunity_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
        s_min, s_max = float(score.min()), float(score.max())
        weight = (score - s_min) / (s_max - s_min) if s_max > s_min else pd.Series(1.0, index=score.index)
    else:
        weight = pd.Series(1.0, index=work.index)

    heat_data = [
        [float(row.lat), float(row.lng), float(weight.loc[idx])]
        for idx, row in work.iterrows()
    ]
    HeatMap(heat_data, radius=18, blur=14, min_opacity=0.25).add_to(map_obj)

    return map_obj


def create_marker_map(
    df: pd.DataFrame,
    top_n: int = 20,
) -> folium.Map:
    """上位出店候補をマーカーで表示する地図を作成する。

    Parameters
    ----------
    df : pd.DataFrame
        ランキング済みデータ
    top_n : int, optional
        表示件数

    Returns
    -------
    folium.Map
        マーカー付き地図
    """
    if df.empty or not {"lat", "lng"}.issubset(df.columns):
        return folium.Map(location=[35.6812, 139.7671], zoom_start=12, tiles="CartoDB positron")

    work = df.copy()
    work["lat"] = pd.to_numeric(work["lat"], errors="coerce")
    work["lng"] = pd.to_numeric(work["lng"], errors="coerce")
    work = work.dropna(subset=["lat", "lng"])

    if "opportunity_score" not in work.columns:
        work["opportunity_score"] = 0.0

    work = work.sort_values("opportunity_score", ascending=False).head(max(1, int(top_n)))

    center_lat = float(work["lat"].mean())
    center_lng = float(work["lng"].mean())
    map_obj = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles="CartoDB positron",
    )

    for _, row in work.iterrows():
        score = float(pd.to_numeric(pd.Series([row["opportunity_score"]]), errors="coerce").fillna(0.0).iloc[0])
        color = "#1b9e77" if score >= 0 else "#d95f02"
        name = row.get("name", "")
        genre = row.get("unified_genre", row.get("genre", ""))
        popup = folium.Popup(
            f"<b>{name}</b><br>ジャンル: {genre}<br>スコア: {score:.3f}",
            max_width=260,
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lng"])],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=popup,
        ).add_to(map_obj)

    return map_obj


def save_map(map_obj: folium.Map, filename: str) -> None:
    """Folium 地図を HTML ファイルとして保存する。

    Parameters
    ----------
    map_obj : folium.Map
        保存対象の地図
    filename : str
        保存ファイル名
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.save(str(path))
    logger.info("地図 HTML 保存: %s", path)


def _load_latest_processed() -> pd.DataFrame:
    """最新の加工済みデータを読み込む。"""
    if not settings.PROCESSED_DATA_DIR.exists():
        return pd.DataFrame()
    files = sorted(
        settings.PROCESSED_DATA_DIR.glob("*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return pd.DataFrame()
    return pd.read_csv(files[0])


def build_dash_app() -> object:
    """Plotly Dash アプリケーションを構築する。

    Returns
    -------
    dash.Dash
        設定済みの Dash アプリインスタンス
    """
    from dash import Dash, Input, Output, dcc, html

    data = _load_latest_processed()
    app = Dash(__name__)

    app.layout = html.Div([
        html.H2("Market Gap Finder Dashboard"),
        dcc.Dropdown(
            id="map-mode",
            options=[
                {"label": "ヒートマップ", "value": "heatmap"},
                {"label": "上位マーカー", "value": "markers"},
            ],
            value="heatmap",
            clearable=False,
            style={"maxWidth": "280px"},
        ),
        dcc.Slider(
            id="top-n",
            min=5, max=100, step=5, value=20,
            marks={5: "5", 20: "20", 50: "50", 100: "100"},
        ),
        html.Iframe(
            id="map-frame",
            style={"width": "100%", "height": "75vh", "border": "none"},
        ),
    ], style={"padding": "16px"})

    @app.callback(
        Output("map-frame", "srcDoc"),
        Input("map-mode", "value"),
        Input("top-n", "value"),
    )
    def _render_map(map_mode: str, top_n: int) -> str:
        if map_mode == "markers":
            map_obj = create_marker_map(data, top_n=top_n or 20)
        else:
            map_obj = create_score_heatmap(data)
        return map_obj.get_root().render()

    return app


def run_dashboard(host: str = "127.0.0.1", port: int = 8050) -> None:
    """ダッシュボードサーバーを起動する。

    Parameters
    ----------
    host : str, optional
        バインドホスト
    port : int, optional
        ポート番号
    """
    app = build_dash_app()
    if hasattr(app, "run_server"):
        app.run_server(host=host, port=port, debug=False)
    else:
        app.run(host=host, port=port, debug=False)
