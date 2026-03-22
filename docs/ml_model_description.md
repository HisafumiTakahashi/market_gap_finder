# MLモデル説明書

## 1. アーキテクチャ

- **モデル**: LightGBM 回帰（Gradient Boosting Decision Tree）
- **ターゲット**: `log1p(restaurant_count)` — 飲食店数の対数変換
- **学習データ**: 9,665行（tokyo / osaka / nagoya / fukuoka × 9ジャンル × 250mメッシュ）
- **特徴量**: 26個（23数値 + 1カテゴリ + 2交互作用）
- **実装**: `src/analyze/ml_model.py`

## 2. ハイパーパラメータ

| パラメータ | 値 | 備考 |
|---|---|---|
| boosting_type | gbdt | |
| num_leaves | 31 | |
| learning_rate | 0.05 | |
| feature_fraction | 0.8 | |
| bagging_fraction | 0.8 | |
| num_rounds | 300 | |
| early_stopping | 30 | |
| objective | regression | |
| metric | rmse | |

## 3. 特徴量一覧（重要度順）

### Top 3（合計66%）

| 特徴量 | 重要度 | SHAP | 説明 |
|---|---|---|---|
| genre_encoded | 25.4% | 0.218 | ジャンル種別（カテゴリカル）。izakayaは基本的に多い、curryは少ない等のジャンル固有の基準値を学習 |
| other_genre_count | 22.9% | 0.127 | 同メッシュ内の他ジャンル店舗数（log変換済み）。集客力・商業集積度の代理指標 |
| saturation_index | 18.1% | 0.098 | `他ジャンル店舗数 / (人口/1万 + 0.064)`。人口あたりの飽和度 |

### 中位（合計20%）

| 特徴量 | 重要度 | SHAP | 説明 |
|---|---|---|---|
| google_avg_rating | 7.2% | 0.099 | Google Places評価平均。エリアの品質シグナル |
| genre_hhi | 5.9% | 0.043 | ジャンル集中度（Herfindahl-Hirschman Index）。高HHI=特定ジャンルに偏り |
| neighbor_avg_restaurants | 5.5% | 0.075 | 隣接8メッシュの平均店舗数。周辺の商業密度 |
| reviews_per_shop | 3.9% | 0.050 | 店舗あたりGoogleレビュー数。需要の強さ |
| genre_diversity | 2.1% | 0.038 | メッシュ内のユニークジャンル数 |

### 下位（合計14%）

| 特徴量 | 重要度 | 説明 |
|---|---|---|
| google_density | 1.7% | Google Places一致率 |
| nearest_station_distance | 0.9% | 最寄り駅までの距離（km） |
| nearest_station_passengers | 0.7% | 最寄り駅の乗降客数 |
| price_x_saturation | 0.6% | 交互作用: 地価 × 飽和度 |
| pop_x_station_dist | 0.6% | 交互作用: 人口 × 駅距離 |
| neighbor_avg_population | 0.6% | 隣接8メッシュの平均人口 |
| working_ratio | 0.5% | 生産年齢人口比率 |
| elderly_ratio | 0.4% | 高齢者比率 |
| single_ratio | 0.4% | 単身世帯比率 |
| young_single_ratio | 0.4% | 若年単身比率 |
| pop_elderly | 0.4% | 65歳以上人口 |
| land_price | 0.4% | 地価（円/m²） |
| single_households | 0.4% | 単身世帯数 |
| young_single | 0.3% | 20-29歳単身世帯数 |
| households | 0.3% | 世帯数 |
| population | 0.2% | 総人口 |
| pop_adult | 0.2% | 20歳以上人口 |
| pop_working | 0.2% | 15-64歳人口 |

## 4. 交差検証結果

### 5-Fold GroupKFold（メッシュ単位グループ化）

| Fold | RMSE | R² |
|---|---|---|
| 1 | 0.317 | 0.791 |
| 2 | 0.313 | 0.798 |
| 3 | 0.307 | 0.792 |
| 4 | 0.306 | 0.790 |
| 5 | 0.308 | 0.802 |
| **平均** | **0.310** | **0.795** |

### Leave-One-Area-Out CV（エリア間汎化性能）

| テストエリア | RMSE | R² | 備考 |
|---|---|---|---|
| tokyo | 0.371 | 0.729 | 最低精度。データ数最大だが分散も最大 |
| osaka | 0.313 | 0.796 | |
| nagoya | 0.269 | 0.815 | 最高精度 |
| fukuoka | 0.260 | 0.810 | |
| **平均** | **0.303** | **0.787** | |

> sapporo は Google Places データなしのため LOAO から除外

## 5. Market Gap（出店機会スコア）

### 算出方法

```
predicted_log = MLモデル予測値（log空間）
actual_log = log1p(実際の飲食店数)

market_gap = predicted_log - actual_log     # log空間での差分
gap_count = expm1(predicted_log) - actual   # 実数空間での差分
```

### 解釈

| market_gap | 意味 | アクション |
|---|---|---|
| 大きい正値 | MLが「もっと店があるべき」と予測 | 出店機会あり |
| ≈ 0 | 予測と実態が一致 | 均衡状態 |
| 大きい負値 | 実態がMLの予測を上回る | 過剰供給の可能性 |

### Top 5 出店機会（2026-03-22時点）

| メッシュ | ジャンル | gap | 予測 | 実際 | 人口 | 駅距離 |
|---|---|---|---|---|---|---|
| 5339453633 | izakaya | +1.88 | 18.7 | 2 | 894 | 0.28km |
| 5339454611 | izakaya | +1.80 | 17.2 | 2 | 1,183 | 0.11km |
| 5235030942 | izakaya | +1.80 | 29.2 | 4 | 259 | 0.27km |
| 5339453634 | izakaya | +1.62 | 14.2 | 2 | 1,282 | 0.23km |
| 5339460043 | izakaya | +1.49 | 7.9 | 1 | 145 | 0.07km |

## 6. v3bスコアリング（ルールベース）

MLモデルの特徴量重要度を反映した加算型スコアリング。

### 算出式

```
raw_score = genre_log_gap × 1.49      # ジャンル別log空間での期待値差
          + other_genre_norm × 2.71   # 他ジャンル店舗数（正規化）
          + google_rating_norm × 0.31 # Google評価（正規化）
          + genre_hhi_norm × 0.07     # ジャンル集中度
          - saturation_norm × 0.15    # 飽和度ペナルティ
          - neighbor_pressure × 0.05  # 近隣圧力ペナルティ

opportunity_score = normalize(raw_score)  # 0-1正規化
```

> 重みはOptuna 200trialで最適化（Spearman=0.586）

### v4アンサンブル（最終スコア）

```
opportunity_score = 0.6 × v3b_rank + 0.4 × ml_gap_rank
```

ルールベース（v3b）の解釈性とMLの予測力を融合。

## 7. v3b vs ML 相関

| 指標 | 旧v3 | v3b |
|---|---|---|
| Spearman ρ | 0.417 | **0.586** |
| 改善率 | - | +41% |

### 残存する乖離の原因

MLのトップ候補（izakaya系）がv3bで1000-5000位にとどまるのは、MLがnon-linearな特徴量間の相互作用（genre × saturation × google_ratingの組み合わせ効果）を捉えているため。formula-basedの構造的限界であり、v4アンサンブルで補完される。

## 8. 重要な知見

1. **人口は飲食店数をほぼ説明しない**（重要度 < 1%）。店舗数を決めるのは「そのエリアの商業集積度」と「ジャンル特性」
2. **ジャンル × 競合構造が支配的**。genre_encoded + other_genre_count + saturation_indexで全体の66%
3. **tokyoの予測精度が最低**（R²=0.729）。データ数は最大だが、繁華街から住宅街まで分散が大きい
4. **2段階残差モデル（Ridge→LightGBM）は不採用**。CV R²が0.795→0.739に悪化
5. **外れ値フィルタも不採用**。Optuna検証で全閾値においてフィルタなしがR²最良。外れ値7件は繁華街の正当データ
