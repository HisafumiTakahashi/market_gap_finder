"""HTML report generation utilities."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
from jinja2 import Template

from config import settings

logger = logging.getLogger(__name__)


def _mesh_col(df: pd.DataFrame) -> str:
    if "jis_mesh" in df.columns:
        return "jis_mesh"
    if "jis_mesh3" in df.columns:
        return "jis_mesh3"
    return "mesh_code"


def generate_explanation(row: pd.Series) -> str:
    """Generate a short Japanese explanation for a candidate row."""
    parts: list[str] = []
    pop = int(row.get("population", 0) or 0)
    rest = int(row.get("restaurant_count", 0) or 0)
    genre = str(row.get("unified_genre", "") or "")
    diversity = int(row.get("genre_diversity", 0) or 0)
    saturation = float(row.get("saturation_index", 0) or 0)
    station_dist = float(row.get("nearest_station_distance", 0) or 0)
    station_name = str(row.get("nearest_station_name", "") or "")
    gap = float(row.get("market_gap", 0) or 0)

    if (
        pop <= 0
        and rest <= 0
        and not genre
        and diversity <= 0
        and saturation <= 0
        and station_dist <= 0
        and not station_name
        and gap <= 0
    ):
        return "追加確認が必要です。"

    if pop >= 50000:
        parts.append(f"人口{pop:,}人の高需要エリアです")
    elif pop >= 30000:
        parts.append(f"人口{pop:,}人の中需要帯エリアです")
    elif pop > 0:
        parts.append(f"人口{pop:,}人の商圏です")

    if genre:
        if rest <= 1:
            parts.append(f"{genre}の競合がかなり少ない状況です")
        elif rest <= 3:
            parts.append(f"{genre}の競合が比較的少ない状況です")
        else:
            parts.append(f"{genre}は既に{rest}店舗あります")

    if diversity >= 8:
        parts.append("ジャンル多様性が高く回遊性があります")
    elif 0 < diversity <= 3:
        parts.append("ジャンル偏りがあり差別化余地があります")

    if 0 < saturation < 5:
        parts.append("飽和度が低く新規出店余地があります")
    elif saturation > 20:
        parts.append("飽和度が高く競争が激しい可能性があります")

    if station_name and station_dist > 0:
        if station_dist <= 0.3:
            parts.append(f"{station_name}駅から約{station_dist * 1000:.0f}mで駅近です")
        elif station_dist <= 0.8:
            parts.append(f"{station_name}駅から約{station_dist * 1000:.0f}mで徒歩圏です")
        else:
            parts.append(f"{station_name}駅から約{station_dist:.1f}kmです")

    if gap > 1.0:
        parts.append(f"ML推定で大きな需要超過が示唆されています（gap={gap:.2f}）")
    elif gap > 0.5:
        parts.append(f"ML推定で需要超過が示唆されています（gap={gap:.2f}）")

    if not parts:
        return "追加確認が必要です。"
    return "。".join(parts) + "。"


def _build_map(candidates: pd.DataFrame) -> str:
    """Build a simple folium map HTML from candidate rows."""
    if candidates.empty:
        return "<p>候補データがありません。</p>"

    center_lat = candidates["lat"].mean()
    center_lng = candidates["lng"].mean()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles="CartoDB positron")

    for rank, (_, row) in enumerate(candidates.iterrows(), 1):
        score = float(row.get("opportunity_score", 0))
        color = "red" if score >= 0.8 else "orange" if score >= 0.5 else "blue"
        popup_html = (
            f"<b>#{rank} {row.get('jis_mesh', row.get('jis_mesh3', ''))} / {row.get('unified_genre', '')}</b><br>"
            f"Score: {score:.3f}<br>"
            f"人口: {int(row.get('population', 0)):,}<br>"
            f"競合: {int(row.get('restaurant_count', 0))}店舗<br>"
            f"駅距離: {float(row.get('nearest_station_distance', 0)):.2f}km"
        )
        folium.Marker(
            location=[float(row["lat"]), float(row["lng"])],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"#{rank} {row.get('unified_genre', '')}",
            icon=folium.Icon(color=color, icon="info-sign"),
        ).add_to(m)

    return m.get_root().render()


_REPORT_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Gap Report - {{ area_tag }}</title>
</head>
<body>
<h1>Market Gap Report</h1>
<p>エリア: {{ area_tag }} / 生成日時: {{ generated_at }} / 候補件数: {{ total_candidates }}</p>
<iframe srcdoc="{{ map_html | e }}" style="width:100%;height:500px;border:none;"></iframe>
</body>
</html>"""
)


def generate_report(tag: str, top_n: int = 20, ml_r2: str = "-") -> Path:
    """Generate an HTML report for one area."""
    integrated_path = settings.PROCESSED_DATA_DIR / f"{tag}_integrated.csv"
    ml_gap_path = settings.PROCESSED_DATA_DIR / f"{tag}_ml_gap.csv"

    if not integrated_path.exists():
        raise FileNotFoundError(f"{integrated_path} not found")

    df = pd.read_csv(integrated_path)
    ml_df = pd.read_csv(ml_gap_path) if ml_gap_path.exists() else None

    v3_top = df.nlargest(top_n, "opportunity_score").copy()
    if ml_df is not None and "market_gap" in ml_df.columns:
        ml_mesh_col = _mesh_col(ml_df)
        df_mesh_col = _mesh_col(v3_top)
        gap_map = ml_df.set_index([ml_mesh_col, "unified_genre"])["market_gap"].to_dict()
        pred_map = ml_df.set_index([ml_mesh_col, "unified_genre"])["predicted_count"].to_dict()
        v3_top["market_gap"] = v3_top.apply(lambda r: gap_map.get((r[df_mesh_col], r["unified_genre"]), 0), axis=1)
        v3_top["predicted_count"] = v3_top.apply(
            lambda r: pred_map.get((r[df_mesh_col], r["unified_genre"]), 0), axis=1
        )

    html = _REPORT_TEMPLATE.render(
        area_tag=tag,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        total_candidates=len(df),
        top_n=top_n,
        avg_population=f"{int(df['population'].mean()):,}" if "population" in df.columns else "-",
        ml_r2=ml_r2,
        map_html=_build_map(v3_top),
        v3_candidates=[],
        ml_candidates=[],
    )

    output_dir = settings.PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{tag}_report.html"
    output_path.write_text(html, encoding="utf-8")
    logger.info("Generated report: %s", output_path)
    return output_path
