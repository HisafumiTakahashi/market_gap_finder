# 市場空白地帯発見システム (Market Gap Finder)

飲食チェーン本部向け新規出店意思決定支援システム。
「需要が高いにもかかわらず競合が少ないエリア × ジャンルの組み合わせ」を
データドリブンで特定し、出店候補をスコアリングしてレコメンドする。

## 概要

```
需要指標 (demand_index)  ──┐
                           ├─→  機会スコア  →  出店候補ランキング
供給指標 (supply_index)  ──┘
```

### スコア算出式

```
opportunity_score = w_demand × demand_index − w_supply × supply_index
```

| 指標 | 説明 | データソース |
|------|------|-------------|
| demand_index | 需要の大きさ（口コミ数・評価等） | Google Places API |
| supply_index | 競合の充足度（同ジャンル店舗密度） | ホットペッパー API / Google Places API |

## ディレクトリ構成

```
market_gap_finder/
├── config/
│   └── settings.py          # APIキー・パラメータ設定
├── data/
│   ├── raw/                 # 取得した生データ保存先
│   └── processed/           # 加工済みデータ保存先
├── src/
│   ├── collect/
│   │   ├── hotpepper.py     # ホットペッパーAPI取得
│   │   └── google_places.py # Google Places API取得
│   ├── preprocess/
│   │   └── cleaner.py       # クレンジング・名寄せ・集約
│   ├── analyze/
│   │   ├── scoring.py       # 機会スコア算出
│   │   └── backtest.py      # バックテスト
│   └── visualize/
│       └── dashboard.py     # Folium地図・Plotly Dash
├── notebooks/
│   └── eda.ipynb            # 基礎分析用
├── tests/
│   └── test_scoring.py      # スコアリングのユニットテスト
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

### 3. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、API キーを設定する。

```env
HOTPEPPER_API_KEY=your_hotpepper_api_key
GOOGLE_PLACES_API_KEY=your_google_places_api_key
```

## 使用方法

> **Note:** 現在はスケルトン段階です。実装は次のフェーズで行います。

### データ収集

```python
from src.collect import hotpepper, google_places
```

### 前処理

```python
from src.preprocess.cleaner import deduplicate, map_genre, assign_mesh_code
```

### スコアリング

```python
from src.analyze.scoring import compute_opportunity_score, rank_opportunities
```

### ダッシュボード

```python
from src.visualize.dashboard import run_dashboard
run_dashboard()
```

## テスト

```bash
pytest tests/ -v
```

## 今後の開発フェーズ

1. **Phase 1**: データ収集モジュール実装（API 連携）
2. **Phase 2**: 前処理パイプライン実装（名寄せ・メッシュ化）
3. **Phase 3**: スコアリングロジック実装・チューニング
4. **Phase 4**: ダッシュボード構築
5. **Phase 5**: バックテスト・精度検証

## ライセンス

Private - All Rights Reserved
