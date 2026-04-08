# Market Gap Finder モデル解説

## 1. MLモデルの役割

MLモデルは、各メッシュに対して「その場所に本来どれくらい飲食店が存在していてもよいか」を推定し、実測との差を `market_gap` として出力します。

- 正の `market_gap`: 出店余地あり
- 負の `market_gap`: 供給過多の可能性

---

## 2. 入力特徴量（28個）

| カテゴリ | 特徴量 |
|---|---|
| 人口・世帯 | `population`, `pop_working`, `pop_adult`, `pop_elderly`, `households`, `single_households`, `young_single` |
| 人口比率 | `working_ratio`, `elderly_ratio`, `single_ratio`, `young_single_ratio` |
| 競合構造 | `genre_diversity`, `genre_hhi`, `other_genre_count`, `commercial_density_rank`, `saturation_index` |
| 近隣 | `neighbor_avg_restaurants`, `neighbor_avg_population` |
| 駅 | `nearest_station_distance`, `nearest_station_passengers`, `station_accessibility` |
| 地価 | `land_price` |
| Google | `google_avg_rating`, `reviews_per_shop`, `google_density` |
| カテゴリ | `genre_encoded` |
| 相互作用 | `price_x_saturation`, `pop_x_station_dist` |

人口欠損率は東京エリアで 10.1% です。欠損値はゼロ埋めし、人口系・人口比率・近隣人口・飽和度もMLモデルで使用します。

---

## 3. 評価指標

東京エリアのみ 4,520件を対象に、店舗数ビンで層化した 5-Fold `StratifiedKFold` で評価しています。

| 指標 | 値 |
|---|---:|
| R² | 0.7870 |
| RMSE | 0.3288 |

---

## 4. 2段階学習モード

2段階学習モードでは、Ridge回帰で需要のベースラインを先に学習し、その残差をLightGBMで補正します。

現在の `DEMAND_FEATURES`:
- `population`
- `pop_working`
- `households`
- `neighbor_avg_population`
- `nearest_station_passengers`
- `land_price`
- `nearest_station_distance`

---

## 5. Market Gap の意味

```python
market_gap = predicted_log_count - actual_log_count
```

- 正: 実測より店が少ない可能性
- 負: 実測より店が多い可能性

---

## 6. 信頼度分布

信頼度は5-Fold CVのfold予測ばらつきから計算します。対象は tokyo 4,520件です。

| ラベル | 件数 | 比率 |
|---|---:|---:|
| high | 2,068 | 45.8% |
| medium | 557 | 12.3% |
| low | 1,895 | 41.9% |

---

## 7. SHAPの読み方

現在の重要度上位:

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

## 8. 人口データの扱い

- 東京エリアの人口欠損率は **10.1%**
- 欠損値は fallback 補完ではなく **ゼロ埋め**
- アブレーション検証では、人口あり（RMSE=0.3266）と人口なし（RMSE=0.3271）はほぼ同等
- 解釈性と将来拡張のため、人口を含む全特徴量版を採用

---

## 9. 関連ファイル

| ファイル | 役割 |
|---|---|
| `src/analyze/ml_model.py` | 学習・推論・SHAP・信頼度 |
| `src/analyze/features.py` | 特徴量生成 |
| `src/analyze/scoring.py` | v1-v5スコアリング |
| `docs/prediction_reliability.md` | 信頼度の詳細 |
