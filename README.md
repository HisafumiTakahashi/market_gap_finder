# 市場空白地帯発見システム (Market Gap Finder)

飲食チェーン本部向け新規出店意思決定支援システム。
「需要が高いにもかかわらず競合が少ないエリア × ジャンルの組み合わせ」を
データドリブンで特定し、出店候補をスコアリングしてレコメンドする。

## 概要

```
需要指標 (人口・昼間人口)  ──┐
                              ├─→  機会スコア  →  出店候補ランキング
供給指標 (店舗密度・飽和度)  ──┘
空間特徴量 (駅距離・地価等)  ──┘
```

### スコアリングモデル

本システムは 3 段階のルールベーススコアリングと ML ベースのギャップ予測を搭載している。

| バージョン | アルゴリズム | 特徴 |
|-----------|-------------|------|
| **v1** | 飲食店数ベース | `demand / (supply + ε)` の単純比率 |
| **v2** | 人口ベース需要 | 昼間人口→人口の優先順で需要計算、人口1万人あたり競合密度 |
| **v3** | 空間特徴量統合 | v2 + ジャンル多様性・HHI・隣接競合・飽和度・駅距離・地価 |
| **ML** | LightGBM | エリア特性から市場容量を予測し、実数との差分を参入機会として評価（R²=0.74〜0.84） |

### 対象エリア（実証済み）

- 東京（tokyo）
- 大阪（osaka）
- 名古屋（nagoya）

### 対象ジャンル

居酒屋 / イタリアン（フレンチ統合） / 中華 / 焼肉（韓国料理統合） / カフェ / ラーメン / 和食 / その他（8ジャンル。`curry` は `GENRE_MAPPING` に定義されているが HotPepper 実データに該当する店舗が存在しないため欠番）

## データソース

| データソース | 用途 | モジュール |
|-------------|------|-----------|
| **ホットペッパー API** | 飲食店データ（座標・ジャンル・店舗数） | `src/collect/hotpepper.py` |
| **e-Stat API** | 国勢調査メッシュ別人口（総人口・昼間人口） | `src/collect/estat.py` |
| **HeartRails Express API** | 鉄道駅座標（最寄り駅距離特徴量） | `src/collect/station.py` |
| **国土数値情報** | 地価公示データ（コスト圧特徴量） | `src/collect/land_price.py` |

> **Note:** Google Places API クライアント（`src/collect/google_places.py`）は実装済みだが、現在は未使用。将来的に評価・レビューデータの取得に活用予定。

## ディレクトリ構成

```
market_gap_finder/
├── config/
│   └── settings.py              # API設定・パラメータ一元管理
├── data/
│   ├── raw/                     # 生データ（HotPepper / e-Stat）
│   ├── processed/               # 統合・スコアリング済みCSV + MLギャップ
│   └── external/                # 外部データキャッシュ（駅・地価）
├── src/
│   ├── collect/
│   │   ├── hotpepper.py         # ホットペッパーAPIクライアント
│   │   ├── estat.py             # e-Stat APIクライアント（動的statsDataId検索）
│   │   ├── station.py           # 駅データ取得（HeartRails Express API）
│   │   ├── land_price.py        # 地価公示データ取得（国土数値情報）
│   │   ├── collector.py         # メッシュグリッド生成 + HotPepper統合収集
│   │   └── google_places.py     # Google Places APIクライアント（未使用）
│   ├── preprocess/
│   │   ├── cleaner.py           # データクレンジング・名寄せ・集約
│   │   └── mesh_converter.py    # JIS標準3次メッシュ（8桁）変換
│   ├── analyze/
│   │   ├── scoring.py           # v1 / v2 / v3 スコアリング
│   │   ├── features.py          # 空間特徴量（多様性・HHI・隣接競合・飽和度・駅距離・地価）
│   │   ├── ml_model.py          # LightGBM 市場容量予測モデル
│   │   └── backtest.py          # バックテスト（開店履歴データ必要）
│   ├── visualize/
│   │   ├── dashboard.py         # Dash ダッシュボード（v1/v2/v3切替 + 人口ヒートマップ）
│   │   └── report.py            # HTMLレポート自動生成（地図 + 特徴量ベース説明文）
│   └── utils/
│       └── geocode.py           # ジオコーディングユーティリティ
├── scripts/
│   ├── collect_area.py          # データ収集CLI
│   ├── integrate_estat.py       # e-Stat統合パイプライン（駅データ連携対応）
│   ├── fetch_external.py        # 外部データ（駅・地価）取得CLI
│   ├── validate_scores.py       # スコア検証・相関分析ツール
│   ├── train_model.py           # LightGBMモデル学習CLI
│   ├── generate_report.py       # HTMLレポート生成CLI
│   ├── preprocess_area.py       # 前処理CLI
│   ├── score_area.py            # スコアリングCLI
│   └── run_dashboard.py         # ダッシュボード起動CLI
├── models/
│   └── {tag}_lightgbm.txt       # 学習済みモデル（tokyo/osaka/nagoya）
├── reports/
│   └── {tag}_report.html        # 自動生成HTMLレポート
├── notebooks/
│   ├── 01_collection_eda.ipynb  # HotPepper収集EDA
│   ├── 02_estat_demand_data.ipynb # e-Stat需要データ取得・結合プロトタイプ
│   ├── 03_estat_eda.ipynb       # e-Stat EDA（人口分布・空間分布・データ品質）
│   └── eda.ipynb                # 基礎分析用
├── tests/
│   ├── test_scoring.py          # スコアリングテスト（30件）
│   ├── test_features.py         # 空間特徴量テスト（15件）
│   └── test_estat.py            # e-Statクライアントテスト（6件）
├── requirements.txt
└── README.md
```

## セットアップ

### 1. 仮想環境の作成

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

主要な依存パッケージ:

| カテゴリ | パッケージ |
|---------|-----------|
| データ処理 | pandas, numpy |
| HTTP | requests, httpx |
| 可視化 | folium, plotly, dash |
| ML | lightgbm, scikit-learn |
| テンプレート | jinja2 |
| 地理計算 | geopy |
| テスト | pytest, pytest-cov |

### 3. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、API キーを設定する。

```env
HOTPEPPER_API_KEY=your_hotpepper_api_key
ESTAT_API_KEY=your_estat_api_key
GOOGLE_PLACES_API_KEY=your_google_places_api_key  # 現在は未使用
```

> `config/settings.py` にホットペッパー / e-Stat のデフォルトキーが設定済みのため、開発時は `.env` なしでも動作する。

## 使用方法

すべてのスクリプトはプロジェクトルートから `PYTHONPATH=.` を付けて実行する。

### Step 1: データ収集

```bash
# 東京エリアのHotPepperデータを収集
PYTHONPATH=. python scripts/collect_area.py \
  --lat-min 35.6 --lat-max 35.8 \
  --lng-min 139.6 --lng-max 139.85 \
  --tag tokyo
```

### Step 2: 外部データ取得（駅・地価）

```bash
# 駅データ・地価データをキャッシュ取得
PYTHONPATH=. python scripts/fetch_external.py --tag tokyo --prefectures 東京都
```

### Step 3: e-Stat統合 + スコアリング

```bash
# e-Stat人口データの結合、空間特徴量計算、v1/v2/v3スコアリングを一括実行
PYTHONPATH=. python scripts/integrate_estat.py --tag tokyo
```

### Step 4: スコア検証

```bash
# 上位20件のスコア・相関分析を表示
PYTHONPATH=. python scripts/validate_scores.py --tag tokyo --top-n 20
```

### Step 5: MLモデル学習

```bash
# LightGBMで市場容量予測モデルを学習し、市場ギャップを算出
PYTHONPATH=. python scripts/train_model.py --tag tokyo --top-n 20
```

### Step 6: レポート生成

```bash
# 単一エリアのHTMLレポートを生成
PYTHONPATH=. python scripts/generate_report.py --tag tokyo --top-n 20

# 全エリア（tokyo/osaka/nagoya）のレポートを一括生成
PYTHONPATH=. python scripts/generate_report.py --all --top-n 20
```

### Step 7: ダッシュボード

```bash
# インタラクティブなDashダッシュボードを起動
PYTHONPATH=. python -c "from src.visualize.dashboard import run_dashboard; run_dashboard(tag='tokyo')"
```

ダッシュボード機能:
- **Score Version 切替**: v1 / v2 / v3 スコアをドロップダウンで切替
- **人口ヒートマップ**: メッシュ別人口の密度マップ
- **空間特徴量ツールチップ**: マーカーホバー時に特徴量詳細を表示

## パイプライン全体像

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐
│  HotPepper  │    │   e-Stat     │    │  HeartRails   │
│  API        │    │   API        │    │  Express API  │
└──────┬──────┘    └──────┬───────┘    └──────┬────────┘
       │                  │                   │
       ▼                  ▼                   ▼
┌──────────────────────────────────────────────────────┐
│  data/raw/          生データ保存                      │
│  {tag}_hotpepper.csv / estat / stations              │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  integrate_estat.py  統合パイプライン                  │
│  ・JIS 3次メッシュ変換                                │
│  ・人口データ結合                                     │
│  ・空間特徴量計算（features.py）                      │
│  ・v1 / v2 / v3 スコアリング（scoring.py）            │
└──────────────────────┬───────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────┐
│  data/processed/    統合済みデータ                     │
│  {tag}_estat_scored.csv                              │
└──────┬───────────────────────┬───────────────────────┘
       │                       │
       ▼                       ▼
┌──────────────┐    ┌──────────────────┐
│  train_model │    │  generate_report │
│  LightGBM    │    │  HTML + 地図     │
│  市場ギャップ │    │  自動説明文      │
└──────────────┘    └──────────────────┘
       │
       ▼
┌──────────────┐
│  dashboard   │
│  Dash Web UI │
└──────────────┘
```

## メッシュ設計

- **独自メッシュ**: `{lat_bin}_{lng_bin}` 形式（MESH_LAT_STEP=0.005 / MESH_LON_STEP=0.00625 ≒ 約500m四方）
- **JIS標準3次メッシュ**: 8桁コード（e-Stat人口データとの結合に使用）
- `mesh_converter.py` で独自メッシュ ↔ JIS 3次メッシュの変換を行う

## テスト

```bash
# 全テスト実行（51件）
pytest tests/ -v

# カバレッジ付き
pytest tests/ -v --cov=src --cov-report=term-missing
```

| テストファイル | 件数 | 対象 |
|--------------|------|------|
| `test_scoring.py` | 30 | v1/v2/v3 スコアリング、正規化、需要計算 |
| `test_features.py` | 15 | 空間特徴量（多様性・HHI・近傍・飽和度・駅距離・地価） |
| `test_estat.py` | 6 | e-Stat APIクライアント |

## 既知の課題

| 課題 | 詳細 | 優先度 |
|------|------|--------|
| `neighbor_avg_restaurants` 分散ゼロ | 全3エリアで相関N/A。近傍計算ロジックの2次メッシュ境界跨ぎ未対応 | 高 |
| 新規モジュールにテストなし | station / land_price / ml_model / report のテスト未実装 | 高 |
| 地価データ未取得 | 国土数値情報API障害。代替手段を要検討 | 中 |
| HotPepper rating/review が空 | API仕様上データなし。Google Places で代替可能 | 中 |
| `backtest.py` が未使用 | 開店履歴データがないと動作しない | 低 |

## ライセンス

Private - All Rights Reserved
