# Market Gap Finder システム概要

## 1. 概要

Market Gap Finder は、飲食店メッシュデータと外部データを統合し、出店余地を推定するシステムです。  
現在のダッシュボードでは、LightGBM の ML Gap をrank正規化した `v5` をデフォルトスコアとして使います。`v4` は `v3b` 0.6 + ML gap 0.4 のアンサンブル版として残っており、ダッシュボードで選択可能です。

対象エリア:
- 東京

---

## 2. スコアリングの流れ

- `v1`: 需要 / 供給の基本スコア
- `v2`: 人口ベース需要を追加
- `v3`: 競合・近隣・駅・地価を追加
- `v3b`: ML重要特徴量に寄せたルールベース改善版
- `v4`: `v3b` 0.6 + ML gap 0.4 のアンサンブル
- `v5`: ML Gap のrank正規化のみで算出するスコア。ダッシュボードのデフォルト

---

## 3. 機械学習モデル

### モデル概要

- アルゴリズム: LightGBM
- 目的変数: `log1p(restaurant_count)`
- 学習データ: 東京のみ 4,520件
- 特徴量数: **28**
- 人口系特徴量: 使用する

### 特徴量

| カテゴリ | 特徴量 |
|---|---|
| 人口・世帯 | `population`, `pop_working`, `pop_adult`, `pop_elderly`, `households`, `single_households`, `young_single` |
| 人口比率 | `working_ratio`, `elderly_ratio`, `single_ratio`, `young_single_ratio` |
| 競合構造 | `genre_diversity`, `genre_hhi`, `other_genre_count`, `commercial_density_rank`, `saturation_index` |
| 近隣 | `neighbor_avg_restaurants`, `neighbor_avg_population` |
| 駅 | `nearest_station_distance`, `nearest_station_passengers`, `station_accessibility` |
| 地価 | `land_price` |
| Google | `google_avg_rating`, `reviews_per_shop`, `google_density` |
| ジャンル | `genre_encoded` |
| 交互作用 | `price_x_saturation`, `pop_x_station_dist` |

### モデル性能

東京エリアのみを対象に、店舗数ビンで層化した 5-Fold `StratifiedKFold` で評価しています。

| 指標 | 値 |
|---|---:|
| 5-Fold CV R² | 0.7870 |
| 5-Fold CV RMSE | 0.3288 |

### SHAP特徴量重要度

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

### Feature Importance（gain）

1. `other_genre_count`: 22.9%
2. `genre_encoded`: 22.0%
3. `commercial_density_rank`: 10.0%
4. `saturation_index`: 9.4%
5. `google_avg_rating`: 8.5%

### 信頼度分布

対象は tokyo 4,520件です。

- high: 2,068件 (45.8%)
- medium: 557件 (12.3%)
- low: 1,895件 (41.9%)

---

## 4. パイプライン

```text
1. Collect
   HotPepper / Google Places / e-Stat / 駅 / 地価データを収集

2. Preprocess
   ジャンル正規化、メッシュ変換、クリーニング

3. Integrate
   全データを統合し、28特徴量を生成

4. Train
   28特徴量でLightGBMを学習し、market gapと信頼度を算出

5. Score
   v5をデフォルトスコアとして作成し、v4も選択肢として保持

6. Output
   CSV、モデル、レポート、ダッシュボードを出力
```

---

## 5. 人口データの位置づけ

- e-Stat人口データは引き続き収集・保持する
- **現行MLモデルでも使用する**
- 東京エリアの人口欠損率 10.1% はゼロ埋めで扱う
- `v2 / v3 / v3b` のスコアリングや将来拡張でも引き続き利用可能

---

## 6. 関連ファイル

| ファイル | 役割 |
|---|---|
| `src/analyze/ml_model.py` | LightGBM学習・推論・SHAP・信頼度 |
| `src/analyze/features.py` | 特徴量生成 |
| `src/analyze/scoring.py` | v1-v5 スコアリング |
| `scripts/train_model.py` | モデル学習CLI |
| `scripts/run_pipeline.py` | 全体パイプライン |
| `src/visualize/dashboard.py` | ダッシュボード |
