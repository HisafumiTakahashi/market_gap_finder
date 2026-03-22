"""HTML report generation utilities."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
from jinja2 import Template

from config import settings
from src.analyze.utils import mesh_col as _mesh_col

logger = logging.getLogger(__name__)

REPORT_GENRES = [
    "izakaya",
    "italian",
    "chinese",
    "yakiniku",
    "cafe",
    "ramen",
    "washoku",
    "curry",
    "other",
]

FALLBACK_EXPLANATION = "十分な説明データがありません。"


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
        return FALLBACK_EXPLANATION

    if pop >= 50000:
        parts.append(f"人口{pop:,}人の大きな商圏です")
    elif pop >= 30000:
        parts.append(f"人口{pop:,}人の中規模商圏です")
    elif pop > 0:
        parts.append(f"人口{pop:,}人を抱えるエリアです")

    if genre:
        if rest <= 1:
            parts.append(f"{genre}の競合がほぼ見当たりません")
        elif rest <= 3:
            parts.append(f"{genre}の競合はまだ少なめです")
        else:
            parts.append(f"{genre}は既に{rest}店舗あります")

    if diversity >= 8:
        parts.append("ジャンル多様性が高く、回遊性が期待できます")
    elif 0 < diversity <= 3:
        parts.append("ジャンル構成が絞られており、差別化余地があります")

    if 0 < saturation < 5:
        parts.append("飽和度が低く、新規出店余地があります")
    elif saturation > 20:
        parts.append("飽和度が高く、競争はやや激しめです")

    if station_name and station_dist > 0:
        if station_dist <= 0.3:
            parts.append(f"{station_name}から約{station_dist * 1000:.0f}mで駅近です")
        elif station_dist <= 0.8:
            parts.append(f"{station_name}から約{station_dist * 1000:.0f}mで徒歩圏です")
        else:
            parts.append(f"{station_name}から約{station_dist:.1f}kmです")

    if gap > 1.0:
        parts.append(f"ML予測では需給ギャップが大きめです (gap={gap:.2f})")
    elif gap > 0.5:
        parts.append(f"ML予測では需給ギャップが見込まれます (gap={gap:.2f})")

    if not parts:
        return FALLBACK_EXPLANATION
    return "。".join(parts) + "。"


def _score_color(score: float) -> str:
    if score >= 0.8:
        return "#d32f2f"
    if score >= 0.6:
        return "#e65100"
    if score >= 0.4:
        return "#ffa000"
    return "#1565c0"


def _build_map(candidates: pd.DataFrame) -> str:
    """Build a folium map HTML from candidate rows."""
    if candidates.empty:
        return "<p>候補データがありません。</p>"

    plot_df = candidates.copy()
    plot_df["lat"] = pd.to_numeric(plot_df.get("lat"), errors="coerce")
    plot_df["lng"] = pd.to_numeric(plot_df.get("lng"), errors="coerce")
    plot_df = plot_df.dropna(subset=["lat", "lng"])
    if plot_df.empty:
        return "<p>候補データがありません。</p>"

    center_lat = float(plot_df["lat"].mean())
    center_lng = float(plot_df["lng"].mean())
    map_obj = folium.Map(location=[center_lat, center_lng], zoom_start=12, tiles="CartoDB positron")

    groups: dict[str, folium.FeatureGroup] = {}
    genres = plot_df["unified_genre"].fillna("other").astype(str)
    for genre in sorted(genres.unique()):
        group = folium.FeatureGroup(name=genre, show=True)
        group.add_to(map_obj)
        groups[genre] = group

    mesh_col = _mesh_col(plot_df)
    for rank, (_, row) in enumerate(plot_df.iterrows(), 1):
        score = float(row.get("opportunity_score", 0) or 0)
        gap = float(row.get("market_gap", 0) or 0)
        genre = str(row.get("unified_genre", "other") or "other")
        radius = max(5.0, min(20.0, 5.0 + gap * 5.0))
        popup_html = (
            f"<b>#{rank} {row.get(mesh_col, '')}</b><br>"
            f"Genre: {genre}<br>"
            f"Score: {score:.3f}<br>"
            f"Population: {int(row.get('population', 0) or 0):,}<br>"
            f"Competitors: {int(row.get('restaurant_count', 0) or 0)}<br>"
            f"Station Distance: {float(row.get('nearest_station_distance', 0) or 0):.2f}km<br>"
            f"Market Gap: {gap:.2f}"
        )
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lng"])],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"#{rank} {genre}",
            radius=radius,
            color=_score_color(score),
            weight=1,
            fill=True,
            fill_color=_score_color(score),
            fill_opacity=0.85,
        ).add_to(groups[genre])

    folium.LayerControl(collapsed=False).add_to(map_obj)
    return map_obj.get_root().render()


_REPORT_TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Market Gap Report - {{ area_tag }}</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "Hiragino Sans", "Meiryo", "Noto Sans JP", sans-serif;
         background: #f5f7fa; color: #333; line-height: 1.6; padding: 24px; }
  h1 { font-size: 1.6rem; margin-bottom: 8px; color: #1a1a2e; }
  h2 { font-size: 1.2rem; margin: 24px 0 12px; color: #16213e; border-left: 4px solid #0f3460; padding-left: 10px; }
  .meta { font-size: 0.85rem; color: #666; margin-bottom: 20px; }
  .stats { display: flex; flex-wrap: wrap; gap: 14px; margin-bottom: 24px; }
  .stat-card { background: #fff; border-radius: 8px; padding: 16px 20px; min-width: 160px;
               flex: 1; box-shadow: 0 1px 4px rgba(0,0,0,0.08); text-align: center; }
  .stat-card .label { font-size: 0.78rem; color: #888; margin-bottom: 4px; }
  .stat-card .value { font-size: 1.5rem; font-weight: 700; color: #0f3460; }
  .map-wrap { margin-bottom: 24px; }
  .map-wrap iframe { width: 100%; height: 520px; border: none; border-radius: 8px;
                     box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .table-wrap { overflow-x: auto; margin-bottom: 32px; }
  .table-controls { margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  .table-controls label { font-size: 0.85rem; color: #555; }
  .table-controls select { padding: 8px 10px; border: 1px solid #cfd8e3; border-radius: 6px; background: #fff; }
  table { width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.08); font-size: 0.85rem; }
  thead { background: #0f3460; color: #fff; }
  th, td { padding: 10px 12px; text-align: left; white-space: nowrap; }
  th { font-weight: 600; }
  tbody tr:nth-child(even) { background: #f9fbfd; }
  tbody tr:hover { background: #eef2f7; }
  td.explanation { white-space: normal; max-width: 360px; font-size: 0.8rem; color: #555; }
  .score-high { color: #d32f2f; font-weight: 700; }
  .score-mid  { color: #e65100; font-weight: 600; }
  .score-low  { color: #1565c0; }
  footer { text-align: center; font-size: 0.75rem; color: #aaa; margin-top: 16px; }
</style>
</head>
<body>

<h1>Market Gap Report</h1>
<p class="meta">エリア: <strong>{{ area_tag }}</strong> &nbsp;|&nbsp; 生成日時: {{ generated_at }} &nbsp;|&nbsp; ML R&sup2;: {{ ml_r2 }}</p>

<h2>サマリー</h2>
<div class="stats">
  <div class="stat-card">
    <div class="label">候補件数</div>
    <div class="value">{{ total_candidates }}</div>
  </div>
  <div class="stat-card">
    <div class="label">平均人口</div>
    <div class="value">{{ avg_population }}</div>
  </div>
  <div class="stat-card">
    <div class="label">平均スコア</div>
    <div class="value">{{ avg_score }}</div>
  </div>
  <div class="stat-card">
    <div class="label">平均 Market Gap</div>
    <div class="value">{{ avg_market_gap }}</div>
  </div>
</div>

<h2>出店候補マップ Top {{ top_n }}</h2>
<div class="map-wrap">
  <iframe srcdoc="{{ map_html | e }}"></iframe>
</div>

<h2>出店候補ランキング Top {{ top_n }}</h2>
<div class="table-wrap">
<div class="table-controls">
  <label for="genre-filter">ジャンル</label>
  <select id="genre-filter">
    <option value="">全ジャンル</option>
    {% for genre in genres %}
    <option value="{{ genre }}">{{ genre }}</option>
    {% endfor %}
  </select>
</div>
<table>
<thead>
<tr>
  <th>順位</th><th>メッシュコード</th><th>ジャンル</th><th>スコア</th>
  <th>人口</th><th>競合数</th><th>Market Gap</th><th>最寄り駅</th><th>説明</th>
</tr>
</thead>
<tbody>
{% for c in candidates %}
<tr data-genre="{{ c.genre }}">
  <td>{{ c.rank }}</td>
  <td>{{ c.mesh_code }}</td>
  <td>{{ c.genre }}</td>
  <td class="{{ c.score_class }}">{{ c.score }}</td>
  <td>{{ c.population }}</td>
  <td>{{ c.restaurant_count }}</td>
  <td>{{ c.market_gap }}</td>
  <td>{{ c.station }}</td>
  <td class="explanation">{{ c.explanation }}</td>
</tr>
{% endfor %}
</tbody>
</table>
</div>

<script>
  const genreFilter = document.getElementById("genre-filter");
  const tableRows = Array.from(document.querySelectorAll("tbody tr[data-genre]"));
  if (genreFilter) {
    genreFilter.addEventListener("change", function () {
      const selectedGenre = genreFilter.value;
      tableRows.forEach(function (row) {
        const matches = !selectedGenre || row.dataset.genre === selectedGenre;
        row.style.display = matches ? "" : "none";
      });
    });
  }
</script>

<footer>Market Gap Finder &copy; {{ year }}</footer>
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

    mesh_col = _mesh_col(v3_top)
    candidates = []
    for rank, (_, row) in enumerate(v3_top.iterrows(), 1):
        score_val = float(row.get("opportunity_score", 0) or 0)
        score_class = "score-high" if score_val >= 0.8 else "score-mid" if score_val >= 0.5 else "score-low"
        gap_val = float(row.get("market_gap", 0) or 0)
        station_name = str(row.get("nearest_station_name", "") or "")
        station_dist = float(row.get("nearest_station_distance", 0) or 0)
        station_str = f"{station_name} ({station_dist * 1000:.0f}m)" if station_name else "-"
        candidates.append(
            {
                "rank": rank,
                "mesh_code": row.get(mesh_col, ""),
                "genre": row.get("unified_genre", ""),
                "score": f"{score_val:.3f}",
                "score_class": score_class,
                "population": f"{int(row.get('population', 0) or 0):,}",
                "restaurant_count": int(row.get("restaurant_count", 0) or 0),
                "market_gap": f"{gap_val:.2f}" if gap_val else "-",
                "station": station_str,
                "explanation": generate_explanation(row),
            }
        )

    avg_pop = f"{int(df['population'].mean()):,}" if "population" in df.columns else "-"
    avg_score = f"{df['opportunity_score'].mean():.3f}" if "opportunity_score" in df.columns else "-"
    has_gap = "market_gap" in v3_top.columns and v3_top["market_gap"].notna().any()
    avg_gap = f"{v3_top['market_gap'].mean():.2f}" if has_gap else "-"

    html = _REPORT_TEMPLATE.render(
        area_tag=tag,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        total_candidates=len(df),
        top_n=top_n,
        avg_population=avg_pop,
        avg_score=avg_score,
        avg_market_gap=avg_gap,
        ml_r2=ml_r2,
        map_html=_build_map(v3_top),
        genres=REPORT_GENRES,
        candidates=candidates,
        year=datetime.now().year,
    )

    output_dir = settings.PROJECT_ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{tag}_report.html"
    output_path.write_text(html, encoding="utf-8")
    logger.info("Generated report: %s", output_path)
    return output_path
