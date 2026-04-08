# スコアリング & ML ガイド

## 1. 全体像

このプロジェクトでは、LightGBM の ML Gap をrank正規化した `v5` をデフォルトスコアとして使います。`v4` は `v3b` 0.6 + ML gap 0.4 のアンサンブル版として残っています。

- `v3b`: 解釈しやすいルールベース
- `ML`: market gap を予測
- `v4`: `v3b` と ML gap のアンサンブル
- `v5`: ML Gap のrank正規化のみ。ダッシュボードのデフォルト

---

## 2. MLモデル

### 2.1 概要

- モデル: LightGBM
- 目的変数: `log1p(restaurant_count)`
- 学習データ: 東京のみ 4,520件
- 特徴量数: **28個**
- CV方式: 店舗数ビンで層化した 5-Fold `StratifiedKFold`

### 2.2 使用する特徴量

| カテゴリ | カラム | 生成元 |
|---|---|---|
| 人口・世帯 | `population`, `pop_working`, `pop_adult`, `pop_elderly`, `households`, `single_households`, `young_single` | `src/analyze/features.py` |
| 人口比率 | `working_ratio`, `elderly_ratio`, `single_ratio`, `young_single_ratio` | `src/analyze/features.py` |
| 競合構造 | `genre_diversity`, `genre_hhi`, `other_genre_count`, `commercial_density_rank`, `saturation_index` | `src/analyze/features.py` |
| 近隣 | `neighbor_avg_restaurants`, `neighbor_avg_population` | `src/analyze/features.py` |
| 駅 | `nearest_station_distance`, `nearest_station_passengers`, `station_accessibility` | `src/analyze/features.py` |
| 地価 | `land_price` | `src/analyze/features.py` |
| Google | `google_avg_rating`, `reviews_per_shop`, `google_density` | 統合処理 |
| ジャンル | `genre_encoded` | `src/analyze/ml_model.py` |
| 相互作用 | `price_x_saturation`, `pop_x_station_dist` | `src/analyze/ml_model.py` |

### 2.3 人口欠損の扱い

- 東京エリアの人口欠損率は **10.1%**
- fallback補完は使わず、欠損値は **ゼロ埋め**
- アブレーション検証では人口あり（RMSE=0.3266）と人口なし（RMSE=0.3271）がほぼ同等
- 解釈性と将来拡張のため、人口を含む28特徴量版を採用

### 2.4 特徴量数の変遷

- 旧構成: 4エリア結合 9,665件、13特徴量、人口特徴量なし
- 現構成: 東京のみ 4,520件、28特徴量、人口特徴量あり
- 交互作用特徴量: `price_x_saturation`, `pop_x_station_dist`

### 2.5 現在の性能

| 指標 | 値 |
|---|---:|
| CV R² | 0.7870 |
| CV RMSE | 0.3288 |

Optuna 100 trial でチューニング済みです。

| パラメータ | 値 |
|---|---:|
| `num_leaves` | 45 |
| `learning_rate` | 0.075 |
| `min_child_samples` | 5 |
| `feature_fraction` | 0.535 |
| `bagging_fraction` | 0.996 |
| `bagging_freq` | 3 |
| `lambda_l1` | 0.786 |
| `lambda_l2` | 0.796 |
| `num_boost_round` | 457 |

### 2.6 SHAP上位

1. `genre_encoded`: 0.196
2. `other_genre_count`: 0.109
3. `google_avg_rating`: 0.109
4. `neighbor_avg_restaurants`: 0.075
5. `saturation_index`: 0.063
6. `reviews_per_shop`: 0.056
7. `genre_hhi`: 0.048
8. `genre_diversity`: 0.046
9. `commercial_density_rank`: 0.038
10. `google_density`: 0.029

---

## 3. v3bスコアリング

v3b は引き続き人口系カラムを利用可能です。  
そのため、以下のような入力列は統合データに残してあります。

- `population`
- `restaurant_count`
- `other_genre_count`
- `google_avg_rating`
- `genre_hhi`
- `saturation_index`
- `neighbor_avg_restaurants`

v3b はルールベース版として維持し、v4のアンサンブル要素としても使います。

---

## 4. 2段階学習モード

`DEMAND_FEATURES` は現在以下の7項目です。

- `population`
- `pop_working`
- `households`
- `neighbor_avg_population`
- `nearest_station_passengers`
- `land_price`
- `nearest_station_distance`

---

## 5. 関連ファイル

| ファイル | 役割 |
|---|---|
| `src/analyze/scoring.py` | v1-v5 スコア計算 |
| `src/analyze/ml_model.py` | MLコア |
| `src/analyze/features.py` | 特徴量生成 |
| `scripts/train_model.py` | 学習CLI |
| `scripts/run_pipeline.py` | パイプライン |
