# Market Gap Finder モデル特徴量詳細

本ドキュメントは、現在の機械学習モデルで実際に使っている特徴量と、人口系データの扱いを整理したものです。

---

## 1. 現在のMLモデル特徴量

MLモデルは **28特徴量** を使用します。

### 人口・世帯
- `population` (`log1p` 変換)
- `pop_working` (`log1p` 変換)
- `pop_adult` (`log1p` 変換)
- `pop_elderly` (`log1p` 変換)
- `households` (`log1p` 変換)
- `single_households` (`log1p` 変換)
- `young_single` (`log1p` 変換)

### 人口比率
- `working_ratio`
- `elderly_ratio`
- `single_ratio`
- `young_single_ratio`

### 競合構造
- `genre_diversity`
- `genre_hhi`
- `other_genre_count` (`log1p` 変換)
- `commercial_density_rank`
- `saturation_index`

### 近隣
- `neighbor_avg_restaurants` (`log1p` 変換)
- `neighbor_avg_population` (`log1p` 変換)

### 駅
- `nearest_station_distance`
- `nearest_station_passengers`
- `station_accessibility` (`log1p` 変換)

### 地価
- `land_price` (`log1p` 変換)

### Google
- `google_avg_rating`
- `reviews_per_shop`
- `google_density`

### カテゴリカル特徴量
- `genre_encoded` (`unified_genre`)

### モデル内で動的に合成される特徴量
- `price_x_saturation`: 地価(log) × 飽和度
- `pop_x_station_dist`: 人口(log) × 駅距離

---

## 2. 人口データの取得元と利用方法

人口・世帯データは **e-Stat API** から取得して統合データに保持します。

### 現在の利用先
- **MLモデル**: 使用する
- **v2 / v3 / v3b のルールベーススコアリング**: 使用する
- **分析用データセット / 比較検証 / 将来拡張**: 使用する

### 欠損の扱い
- 東京エリアの人口欠損率は **10.1%**
- fallback補完は使わず、欠損値は **ゼロ埋め**
- アブレーション検証では人口あり（RMSE=0.3266）と人口なし（RMSE=0.3271）がほぼ同等
- 解釈性と将来拡張のため、人口を含む全特徴量版を採用

### 2段階学習モードの需要特徴量

`DEMAND_FEATURES` は現在以下の7項目です。

- `population`
- `pop_working`
- `households`
- `neighbor_avg_population`
- `nearest_station_passengers`
- `land_price`
- `nearest_station_distance`

---

## 3. 昼間人口の扱い

MLモデルでは人口系特徴量を使用します。昼間人口は `daytime_population` カラムが存在すれば、ルールベーススコアリング側で総人口より優先して需要計算に使える設計を維持しています。

### 現在の整理
- **MLモデル**: 人口・世帯・人口比率・近隣人口を使用する
- **スコアリング系ロジック**: `daytime_population` を将来利用可能な拡張ポイントとして維持
- **e-Stat / 経済センサス連携**: 今後の拡張候補
