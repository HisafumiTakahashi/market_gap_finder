# MLモデル詳細

## 1. アーキテクチャ

- モデル: LightGBM
- 目的変数: `log1p(restaurant_count)`
- 学習データ: 東京のみ 4,520件
- 特徴量: **28個**
- 構成: **25数値（人口比率4を含む） + 1カテゴリ + 2交互作用**
- CV方式: 店舗数ビンで層化した 5-Fold `StratifiedKFold`
- 実装: `src/analyze/ml_model.py`

人口特徴量を含む28特徴量を使用します。東京エリアの人口欠損率は 10.1% で、欠損値はゼロ埋めします。

---

## 2. 特徴量一覧

### 数値特徴量（23個）

| カテゴリ | 特徴量 | 説明 | log変換 |
|---|---|---|---|
| 人口 | `population` | 総人口 | Yes |
| 人口 | `pop_working` | 生産年齢人口 | Yes |
| 人口 | `pop_adult` | 成人人口 | Yes |
| 人口 | `pop_elderly` | 高齢者人口 | Yes |
| 人口 | `households` | 世帯数 | Yes |
| 人口 | `single_households` | 単身世帯数 | Yes |
| 人口 | `young_single` | 若年単身者数 | Yes |
| 競合構造 | `genre_diversity` | ジャンル多様性 | |
| 競合構造 | `genre_hhi` | ジャンル集中度 | |
| 競合構造 | `other_genre_count` | 他ジャンル店舗数 | Yes |
| 競合構造 | `commercial_density_rank` | 商業密度ランク | |
| 近隣 | `neighbor_avg_restaurants` | 隣接8メッシュ平均店舗数 | Yes |
| 近隣 | `neighbor_avg_population` | 隣接8メッシュ平均人口 | Yes |
| 競合・人口 | `saturation_index` | 人口あたり競合飽和度 | |
| 駅 | `nearest_station_distance` | 最寄駅距離 | |
| 駅 | `nearest_station_passengers` | 最寄駅乗降客数 | |
| 駅 | `station_accessibility` | 駅アクセス性 | Yes |
| 地価 | `land_price` | 地価 | Yes |
| Google | `google_avg_rating` | Google平均評価 | |
| Google | `reviews_per_shop` | 1店舗あたりレビュー数 | |
| Google | `google_density` | Google店舗密度 | |

### カテゴリカル特徴量（1個）

- `genre_encoded`: `unified_genre`

### 交互作用特徴量（2個）

- `price_x_saturation`: 地価(log) × 飽和度
- `pop_x_station_dist`: 人口(log) × 駅距離

### 人口比率特徴量（4個）

- `working_ratio`
- `elderly_ratio`
- `single_ratio`
- `young_single_ratio`

---

## 3. 特徴量削減の経緯

アブレーション検証の結果、人口あり（RMSE=0.3266）と人口なし（RMSE=0.3271）はほぼ同等でした。解釈性と将来拡張のため、人口を含む全特徴量版を採用しています。

---

## 4. SHAP特徴量重要度（mean |SHAP|）

| 順位 | 特徴量 | mean \|SHAP\| | 解釈 |
|---|---|---:|---|
| 1 | `genre_encoded` | 0.196 | ジャンル固有の基準値 |
| 2 | `other_genre_count` | 0.109 | 商業集積度の強い代理変数 |
| 3 | `google_avg_rating` | 0.109 | エリア品質・集客力の代理変数 |
| 4 | `neighbor_avg_restaurants` | 0.075 | 近隣の商業集積 |
| 5 | `saturation_index` | 0.063 | 人口あたり競合飽和度 |
| 6 | `reviews_per_shop` | 0.056 | 利用量・人気の代理指標 |
| 7 | `genre_hhi` | 0.048 | ジャンル集中度 |
| 8 | `genre_diversity` | 0.046 | 業種の混在度 |
| 9 | `commercial_density_rank` | 0.038 | 商業地らしさ |
| 10 | `google_density` | 0.029 | Google観測ベースの密度 |

---

## 5. Feature Importance（gain）

1. `other_genre_count`: 22.9%
2. `genre_encoded`: 22.0%
3. `commercial_density_rank`: 10.0%
4. `saturation_index`: 9.4%
5. `google_avg_rating`: 8.5%

---

## 6. 交差検証結果

東京エリアのみを対象に、店舗数ビンで層化した 5-Fold `StratifiedKFold` で評価しています。

| 指標 | 値 |
|---|---:|
| CV R² | 0.7870 |
| CV RMSE | 0.3288 |

ハイパーパラメータは Optuna 100 trial でチューニング済みです。

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

---

## 7. Market Gapと信頼度

- `market_gap = predicted_log_count - actual_log_count`
- 対象: tokyo 4,520件
- `high`: 2,068件 (45.8%)
- `medium`: 557件 (12.3%)
- `low`: 1,895件 (41.9%)

詳細は `docs/prediction_reliability.md` を参照。

---

## 8. 2段階学習モード

`DEMAND_FEATURES` は以下の7項目です。

- `population`
- `pop_working`
- `households`
- `neighbor_avg_population`
- `nearest_station_passengers`
- `land_price`
- `nearest_station_distance`

人口特徴量を需要ベースラインに含めます。

---

## 9. 重要な知見

1. **人口特徴量は復活した。** 東京エリアの人口欠損率は 10.1% まで下がり、欠損はゼロ埋めで扱う。
2. **人口あり版と人口なし版の精度はほぼ同等。** RMSE は人口あり 0.3266、人口なし 0.3271 で、解釈性と将来拡張のため全特徴量版を採用した。
3. **ジャンル × 商業集積が支配的。** `genre_encoded` と `other_genre_count` がSHAP・gainの双方で上位。
4. **Google特徴量は引き続き有効。** `google_avg_rating`, `reviews_per_shop`, `google_density` が上位に入る。
5. **v5を新設した。** ML Gap のrank正規化のみでスコアリングし、ダッシュボードのデフォルトとして使う。
6. **v4は残存する。** `v3b` 0.6 + ML gap 0.4 のアンサンブルとして、ダッシュボードで選択可能。
