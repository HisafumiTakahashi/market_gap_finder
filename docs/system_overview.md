# 市場空白地帯発見システム (Market Gap Finder) システム概要

## 1. システムの目的とコンセプト
**「需要が高いにもかかわらず競合が少ないエリア（空白地帯）」を特定する**システムです。
「どのエリア（JIS標準メッシュ単位）」に「どのジャンル（居酒屋、カフェ等）」を出店するのが最も勝率が高いかを、**機会スコア（Opportunity Score）** として数値化し、ランキング形式でレコメンドします。

**対象エリア**: 東京・大阪・名古屋・福岡・札幌の5都市

## 2. スコアリングロジック（v1 → v2 → v3 → v3b → v4）

### v1: 飲食店数ベース（初期版）
- `demand / (supply + offset)` の単純比率
- 飲食店数のみに基づく基礎スコア

### v2: 人口ベース需要
- e-Stat国勢調査メッシュ人口を需要のベースとして導入
- 昼間人口 → 人口 → 飲食店数の優先順で需要を算出
- 人口1万人あたりの競合密度を供給指標として使用

### v3: 空間特徴量統合
- v2に加え、ジャンル多様性・HHI・飽和指数・近隣メッシュ平均・駅距離・地価を組み込んだ多変量スコア
- 分子（需要側）と分母（供給側）に各特徴量を配分する比率型

### v3b: 対数空間ジャンルギャップ（現行メイン）
- ML構造を模した加算型スコアリング
- `genre_log_gap`（ジャンル平均対数 − 当該メッシュ対数）で出店余地を定量化
- Optunaで200回試行の重み最適化（Spearman=0.586 vs ML）
- 主要シグナル: 他ジャンル店舗数（W=2.71）、ジャンルベース（W=1.49）、地価（W=1.65）
- ペナルティ: 飽和指数、近隣圧力

### v4: アンサンブル
- v3bランク（60%）+ MLギャップランク（40%）の加重アンサンブル
- MLモデルの予測が利用可能な場合のみ使用可能

## 3. 機械学習モデル（LightGBM）

### モデル概要
- **アルゴリズム**: LightGBM（GBDT）
- **目的変数**: `log1p(restaurant_count)`（対数変換した飲食店数）
- **特徴量**: 数値28個 + カテゴリ1個（unified_genre）
- **学習データ**: 9,665行（東京・大阪・名古屋・福岡の4エリア）
- **ハイパーパラメータ**: Optunaで自動チューニング

### 特徴量（28+1）
| カテゴリ | 特徴量 |
|---|---|
| 人口 | population, pop_working, pop_adult, pop_elderly, households, single_households, young_single |
| 人口比率 | working_ratio, elderly_ratio, single_ratio, young_single_ratio |
| 空間・競合 | genre_diversity, genre_hhi, other_genre_count, commercial_density_rank, neighbor_avg_restaurants, neighbor_avg_population, saturation_index |
| 駅 | nearest_station_distance, nearest_station_passengers, station_accessibility |
| 地価 | land_price |
| Google | google_avg_rating, reviews_per_shop, google_density |
| ジャンル | unified_genre（カテゴリカル: cafe, chinese, curry, italian, izakaya, other, ramen, washoku, yakiniku） |

### モデル性能
| 指標 | 値 |
|---|---|
| 5-Fold CV R² | 0.797 |
| 5-Fold CV RMSE | 0.309 |
| LOAO 東京 R² | 0.732 |
| LOAO 大阪 R² | 0.797 |
| LOAO 名古屋 R² | 0.810 |
| LOAO 福岡 R² | 0.816 |

### SHAP特徴量重要度（Top 5）
1. genre_encoded: 0.219
2. other_genre_count: 0.154
3. google_avg_rating: 0.098
4. neighbor_avg_restaurants: 0.078
5. saturation_index: 0.065

### マーケットギャップ算出
- `market_gap = predicted - actual`（予測値 − 実測値）
- 正の値 = 供給不足（出店機会あり）、負の値 = 供給過多

### 予測信頼度
- 5-Fold CVの5モデル間の予測ばらつきから95%信頼区間（CI）を算出
- CI下限 > 0 → **high**（統計的に有意な出店余地）
- gap > 0 かつ CI下限 ≤ 0 → **medium**（不確実）
- gap ≤ 0 → **low**（出店余地なし）
- 分布: high=50.1%, medium=11.9%, low=38.0%
- `filter_recommendations` 関数で信頼度ベースのフィルタリングが可能

## 4. データパイプラインと処理フロー

```
1. 収集 (Collect)
   HotPepper API → メッシュグリッド探索 → data/raw/{tag}_hotpepper.csv
   e-Stat API → 7カテゴリ人口データ（250m 4分の1メッシュ）
   HeartRails Express API → 駅データ
   国土数値情報 → 駅乗降客数・地価

2. 前処理 (Preprocess)
   ジャンル正規化（9統一ジャンル）
   JIS 3次/4分の1メッシュコード変換
   データクレンジング・重複除去

3. 統合 (Integrate)
   人口データ + 店舗データ → メッシュコードで結合
   空間特徴量エンジニアリング（28特徴量生成）
   v1〜v3bスコアリング

4. ML学習 (Train)
   LightGBM学習（Optuna + LOAO交差検証）
   マーケットギャップ算出 → v4スコア

5. 出力 (Output)
   data/processed/{tag}_integrated.csv（全スコア・特徴量付き）
   models/{tag}_lightgbm.txt（学習済みモデル）
   HTMLレポート・Dashダッシュボード
```

### 統合パイプラインCLI
```bash
python scripts/run_pipeline.py              # 4エリア一括実行
python scripts/add_area.py --tag sapporo    # 新エリア追加
```

## 5. システム構造

### 収集層 (`src/collect/`)
| モジュール | 役割 | データソース |
|---|---|---|
| `hotpepper.py` | 飲食店競合データ取得 | HotPepper API |
| `estat.py` | 人口統計データ取得（7カテゴリ） | e-Stat API |
| `station.py` | 鉄道駅データ取得 | HeartRails Express API |
| `station_passengers.py` | 駅乗降客数取得 | 国土数値情報 S12 |
| `land_price.py` | 公示地価取得 | 国土数値情報 |
| `collector.py` | メッシュグリッド生成＋統合収集 | - |
| `google_places.py` | Google Places（現在未使用） | Google Places API |

### 前処理層 (`src/preprocess/`)
| モジュール | 役割 |
|---|---|
| `cleaner.py` | ジャンル正規化・データクレンジング |
| `mesh_converter.py` | JIS 4分の1メッシュ変換（10桁, 250m）・近隣メッシュ算出 |

### 分析層 (`src/analyze/`)
| モジュール | 役割 |
|---|---|
| `scoring.py` | v1/v2/v3/v3b/v4スコアリング（408行） |
| `features.py` | 28空間特徴量エンジニアリング（cKDTree最適化） |
| `ml_model.py` | LightGBM学習・推論・SHAP・Optunaチューニング・予測信頼度・推薦フィルタ |
| `backtest.py` | バックテスト機能（過去データ必要） |

### 可視化層 (`src/visualize/`)
| モジュール | 役割 |
|---|---|
| `dashboard.py` | Dashダッシュボード（エリア選択・ジャンルフィルタ・信頼度フィルタ・ランキング表・メッシュグリッド・全エリア比較） |
| `report.py` | HTMLレポート生成（地図・スコア説明付き） |

### CLIスクリプト (`scripts/`)
| スクリプト | 役割 |
|---|---|
| `collect_area.py` | HotPepperデータ収集 |
| `fetch_external.py` | 外部データ（駅・地価・乗降客数）取得 |
| `integrate_estat.py` | e-Stat統合 + 特徴量生成 + スコアリング |
| `train_model.py` | MLモデル学習（単体/結合, チューニング, LOAO） |
| `tune_v3b_weights.py` | v3b重み最適化（Optuna） |
| `validate_scores.py` | スコア検証・相関分析 |
| `generate_report.py` | HTMLレポート生成 |
| `run_dashboard.py` | ダッシュボード起動 |
| `run_pipeline.py` | 全パイプライン一括実行（4エリア） |
| `add_area.py` | 新エリア追加（統合ワンショット） |

## 6. テスト
- **153テスト**が全パス
- 主要テストファイル: `test_scoring.py`(30), `test_ml_model.py`(20), `test_features.py`(15), `test_estat.py`(6), `test_mesh_converter.py`, `test_report.py`, `test_run_pipeline.py` 他

## 7. 既知の課題と今後の方向性

### 既知の課題
- **rc=1が全データの49.9%** を占める構造的課題 — 2段階モデル（分類→回帰）で解決の可能性あり
- **Google特徴量**がSHAP上位だが、札幌にはGoogleデータなし
- **ダッシュボード性能**: 読み込み時に毎回特徴量再計算が発生
- **メッシュグリッド描画**: 東京の1,233メッシュ描画が重い場合あり

### 今後の拡張方向
- rc=1の2段階モデル化（分類→回帰）
- Google Places以外の代替レビューデータ検討
- ダッシュボードの特徴量キャッシュ機構
- バックテスト（実際の出店成功/撤退データとの照合）
- 時系列トレンド（人口増減）の組み込み
- 推薦の事後検証の仕組み
