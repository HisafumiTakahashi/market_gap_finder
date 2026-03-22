"""
市場空白地帯発見システム - 設定ファイル

APIキー、検索パラメータ、ファイルパスなど
プロジェクト全体で使用する定数・設定値を一元管理する。
APIキーは .env ファイルで管理（.gitignore対象）。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ──────────────────────────────────────
# プロジェクトルート
# ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# .env を読み込み
load_dotenv(PROJECT_ROOT / ".env")

# ──────────────────────────────────────
# データディレクトリ
# ──────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# ──────────────────────────────────────
# API キー（.env → 環境変数から取得）
# ──────────────────────────────────────
HOTPEPPER_API_KEY: str = os.getenv("HOTPEPPER_API_KEY", "")
GOOGLE_PLACES_API_KEY: str = os.getenv("GOOGLE_PLACES_API_KEY", "")  # 勝手にAPIアクセスしないこと
ESTAT_API_KEY: str = os.getenv("ESTAT_API_KEY", "")

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

# API 収集グリッド間隔（Hotpepper / Google Places の走査ステップ）
# ※ 分析用メッシュは JIS 標準地域メッシュ（mesh_converter.py）を使用
MESH_LAT_STEP: float = 0.005   # 約 500 m
MESH_LON_STEP: float = 0.00625  # 約 500 m

# 機会スコア v3 の重み（Optuna 100trial で最適化）
# 最適化目標: v3スコアとMLギャップ(予測-実測)のSpearman相関を最大化
# 結果: Spearman 0.084 → 0.172 (デフォルト1.0比 +105%)
# 競合密度の重みが最大 — 供給過多エリアの減点が最も重要
WEIGHT_DEMAND: float = 0.17       # 需要スコアの重み（人口ベース需要シグナル）
WEIGHT_COMPETITOR: float = 2.76   # 競合密度の重み（除算側に乗算、供給過多ペナルティ）
WEIGHT_POPULATION: float = 0.18   # 人口密度の重み（需要と高相関のため低く抑制）
WEIGHT_LAND_PRICE: float = 1.65   # 地価逆数の重み（低地価＝参入コスト低のボーナス）
 
# ──────────────────────────────────────
# 外部データディレクトリ
# ──────────────────────────────────────
# v3bスコアリング重み（Optuna 200trial, Spearman=0.586で最適化）
V3B_W_OTHER_GENRE: float = 2.71   # 他ジャンル店舗数 = 集客力シグナル
V3B_W_GENRE_GAP: float = 0.07     # ジャンルHHI（ほぼ不要）
V3B_W_RATING: float = 0.31        # Google評価
V3B_W_DIVERSITY: float = 0.0      # 未使用（Optunaで探索外）
V3B_W_STATION: float = 0.0        # 未使用（Optunaで探索外）
V3B_W_SATURATION: float = 0.15    # 飽和度ペナルティ（小）
V3B_W_GENRE_BASE: float = 1.49    # ジャンル別log空間ギャップ
V3B_W_NEIGHBOR: float = 0.05      # 近隣圧力（ほぼ不要）
V3B_W_LAND_PRICE: float = 0.0     # 未使用

EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"
