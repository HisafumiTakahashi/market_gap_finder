"""
市場空白地帯発見システム - 設定ファイル

APIキー、検索パラメータ、ファイルパスなど
プロジェクト全体で使用する定数・設定値を一元管理する。
"""

import os
from pathlib import Path

# ──────────────────────────────────────
# プロジェクトルート
# ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ──────────────────────────────────────
# データディレクトリ
# ──────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ──────────────────────────────────────
# API キー（環境変数から取得）
# ──────────────────────────────────────
HOTPEPPER_API_KEY: str = os.getenv("HOTPEPPER_API_KEY", "a43c9f65735eed2c")
GOOGLE_PLACES_API_KEY: str = os.getenv("GOOGLE_PLACES_API_KEY", "")
ESTAT_API_KEY: str = os.getenv("ESTAT_API_KEY", "1de94b2d66d3db18a316a3afd04c9a85f363cba3")

# ──────────────────────────────────────
# ホットペッパー API 設定
# ──────────────────────────────────────
HOTPEPPER_BASE_URL: str = "https://webservice.recruit.co.jp/hotpepper/gourmet/v1/"
HOTPEPPER_MAX_RESULTS: int = 100  # 1リクエストあたりの最大取得件数

# ──────────────────────────────────────
# Google Places API 設定
# ──────────────────────────────────────
GOOGLE_PLACES_BASE_URL: str = "https://maps.googleapis.com/maps/api/place"
GOOGLE_PLACES_RADIUS_M: int = 1000  # 検索半径（メートル）

# ──────────────────────────────────────
# e-Stat API 設定
# ──────────────────────────────────────
ESTAT_API_BASE_URL: str = "https://api.e-stat.go.jp/rest/3.0/app"

# ──────────────────────────────────────
# 分析パラメータ
# ──────────────────────────────────────
# 対象ジャンル一覧
TARGET_GENRES: list[str] = [
    "居酒屋",
    "イタリアン",
    "中華",
    "焼肉",
    "カフェ",
    "ラーメン",
    "和食",
    "フレンチ",
    "韓国料理",
    "カレー",
]

# メッシュグリッド設定（緯度経度の刻み幅）
MESH_LAT_STEP: float = 0.005   # 約 500 m
MESH_LON_STEP: float = 0.00625  # 約 500 m

# 機会スコアの重み（各因子の線形乗数として適用）
# opportunity_score_raw = w_demand * demand_score / (competitor_density + 0.01)
#                       * w_population * population_density_norm
#                       * w_land_price * (1 / (land_price_norm + ε))
WEIGHT_DEMAND: float = 1.0       # 需要スコアの重み
WEIGHT_COMPETITOR: float = 1.0   # 競合密度の重み（除算側に乗算）
WEIGHT_POPULATION: float = 1.0   # 人口密度の重み
WEIGHT_LAND_PRICE: float = 1.0   # 地価逆数の重み
 
# ──────────────────────────────────────
# 外部データディレクトリ
# ──────────────────────────────────────
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
