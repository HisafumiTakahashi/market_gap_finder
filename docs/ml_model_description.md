# MLモデル説明書

## 1. アーキテクチャ

- **モデル**: LightGBM 回帰（Gradient Boosting Decision Tree）
- **ターゲット**: `log1p(restaurant_count)` — 飲食店数の対数変換
- **学習データ**: 9,665行（tokyo / osaka / nagoya / fukuoka × 9ジャンル × 250mメッシュ）
- **特徴量**: 28個（25数値 + 1カテゴリ + 2交互作用）
- **実装**: `src/analyze/ml_model.py`

> sapporo は Google Places データなしのため学習データから除外

## 2. ハイパーパラメータ

| パラメータ | デフォルト値 | 備考 |
|---|---|---|
| boosting_type | gbdt | |
| num_leaves | 31 | Optuna探索範囲: 15-63 |
| learning_rate | 0.05 | Optuna探索範囲: 0.01-0.3 (log) |
| feature_fraction | 0.8 | Optuna探索範囲: 0.5-1.0 |
| bagging_fraction | 0.8 | Optuna探索範囲: 0.5-1.0 |
| num_rounds | 300 | Optuna探索範囲: 100-500 |
| early_stopping | 30 | |
| objective | regression | |
| metric | rmse | |

Optunaで `min_child_samples`, `lambda_l1`, `lambda_l2`, `bagging_freq` も探索対象。

## 3. 特徴量一覧

### 数値特徴量（25個）

| カテゴリ | 特徴量 | 説明 | log変換 |
|---|---|---|---|
| **人口** | population | 総人口 | |
| | pop_working | 15-64歳人口（生産年齢人口） | |
| | pop_adult | 20歳以上人口 | |
| | pop_elderly | 65歳以上人口 | |
| | households | 世帯数 | |
| | single_households | 単身世帯数 | |
| | young_single | 若年単身者数 | |
| **人口比率** | working_ratio | 生産年齢人口比率 | |
| | elderly_ratio | 高齢者比率 | |
| | single_ratio | 単身世帯比率 | |
| | young_single_ratio | 若年単身比率 | |
| **競合構造** | genre_diversity | メッシュ内ユニークジャンル数（Shannon） | |
| | genre_hhi | ジャンル集中度（Herfindahl-Hirschman Index） | |
| | other_genre_count | 同メッシュ内の他ジャンル店舗数 | Yes |
| | commercial_density_rank | 他ジャンル店舗数のパーセンタイルランク | |
| **近隣** | neighbor_avg_restaurants | 隣接8メッシュの平均店舗数 | Yes |
| | neighbor_avg_population | 隣接8メッシュの平均人口 | |
| **飽和度** | saturation_index | (メッシュ全店舗数−自ジャンル) / (人口/1万+0.064) | |
| **駅** | nearest_station_distance | 最寄り駅までの距離（km） | |
| | nearest_station_passengers | 最寄り駅の乗降客数 | |
| | station_accessibility | 乗降客数 / (駅距離+0.1) | Yes |
| **地価** | land_price | 地価公示（円/m²） | Yes |
| **Google** | google_avg_rating | Google Places評価平均 | |
| | reviews_per_shop | 店舗あたりGoogleレビュー数 | |
| | google_density | Google Places一致率 | |

### カテゴリカル特徴量（1個）

- `unified_genre`: 飲食ジャンル（`CategoricalDtype` + `GENRE_ORDER` で固定エンコーディング）
- GENRE_ORDER: cafe=0, chinese=1, curry=2, italian=3, izakaya=4, other=5, ramen=6, washoku=7, yakiniku=8

### 交互作用特徴量（2個）

- `price_x_saturation`: 地価(log) × 飽和度 — 出店コスト×競合のダブルハードル
- `pop_x_station_dist`: 人口 × 駅距離 — 駅遠×人口多＝ロードサイド需要

## 4. SHAP特徴量重要度（mean |SHAP|）

### Top 5（合計61%）

| 順位 | 特徴量 | mean |SHAP| | 解釈 |
|---|---|---|---|
| 1 | genre_encoded | 0.219 | ジャンル固有の基準値（izakayaは多い、curryは少ない等） |
| 2 | other_genre_count | 0.154 | 商業集積度の代理指標。店が多い場所はさらに多い |
| 3 | google_avg_rating | 0.098 | エリアの品質シグナル。高評価=集客力のあるエリア |
| 4 | neighbor_avg_restaurants | 0.078 | 周辺の商業密度。近隣の繁盛は波及効果を示す |
| 5 | saturation_index | 0.065 | 人口あたりの競合飽和度。高い=参入余地が少ない |

### 中位

| 順位 | 特徴量 | mean |SHAP| | 解釈 |
|---|---|---|---|
| 6 | reviews_per_shop | 0.048 | 需要の強さ。レビューが多い=利用者が多い |
| 7 | genre_hhi | 0.044 | ジャンル偏在度。特定ジャンルの独占度合い |
| 8 | genre_diversity | 0.030 | ジャンルの多様性。多様=成熟した飲食街 |
| 9 | google_density | 0.018 | Google登録率。高い=可視性のあるエリア |
| 10 | commercial_density_rank | 0.015 | 商業集積の相対ランク |

### 下位（各 < 0.01）

人口系（population, pop_working等）、人口比率系（working_ratio等）、駅系（nearest_station_distance等）、地価（land_price）、交互作用項（price_x_saturation, pop_x_station_dist）

## 5. 交差検証結果

### 5-Fold GroupKFold（メッシュ単位グループ化）

| 指標 | 値 |
|---|---|
| **平均 RMSE** | **0.309** |
| **平均 R²** | **0.797** |

### Leave-One-Area-Out CV（エリア間汎化性能）

| テストエリア | RMSE | R² | 備考 |
|---|---|---|---|
| tokyo | - | 0.732 | 最低精度。データ数最大だが繁華街〜住宅街の分散が大きい |
| osaka | - | 0.797 | |
| nagoya | - | 0.810 | |
| fukuoka | - | 0.816 | 最高精度 |

> sapporo は Google Places データなしのため LOAO から除外

## 6. Market Gap（出店機会スコア）

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

### 予測信頼度（Prediction Reliability）

5-Fold CVで生成される5つのモデル全てで全データを予測し、fold間のばらつき（gap_std）から95%信頼区間を算出する。信頼区間がゼロを跨ぐかどうかで3段階に分類:

| 信頼度 | 条件 | 意味 |
|---|---|---|
| high | CI下限 > 0 | 統計的に有意な出店余地（5モデルが一致） |
| medium | gap > 0 かつ CI下限 ≤ 0 | 出店余地の可能性はあるが統計的に不確実 |
| low | gap ≤ 0 | モデル上は出店余地なし |

分布（combined 9,665件）: high=4,843件(50.1%), medium=1,148件(11.9%), low=3,674件(38.0%)

推薦フィルタ `filter_recommendations` で信頼度・最小店舗数・最小ギャップによる絞り込みが可能。

> 詳細は `docs/prediction_reliability.md` を参照

## 7. v3bスコアリング（ルールベース）

MLモデルの特徴量重要度を反映した加算型スコアリング。

### 算出式

```
genre_log_gap = genre_avg_log - current_log  # 正値 = ジャンル平均より少ない = 機会

raw_score = genre_log_gap × 1.49      # ジャンル別log空間での期待値差
          + other_genre_norm × 2.71   # 他ジャンル店舗数（正規化）— 最強シグナル
          + google_rating_norm × 0.31 # Google評価（正規化）
          + genre_hhi_norm × 0.07     # ジャンル集中度（ほぼ不要）
          - saturation_norm × 0.15    # 飽和度ペナルティ
          - neighbor_pressure × 0.05  # 近隣圧力ペナルティ（ほぼ不要）

opportunity_score = normalize(raw_score)  # 0-1正規化
```

> 重みはOptuna 200trialで最適化（対ML gap Spearman=0.586）
> V3B_W_STATION, V3B_W_DIVERSITY, V3B_W_LAND_PRICE は 0.0（Optuna探索で不要と判定）

### v4アンサンブル（最終スコア）

```
opportunity_score = 0.6 × v3b_rank + 0.4 × ml_gap_rank
```

ルールベース（v3b）の解釈性とMLの予測力を融合。

## 8. v3b vs ML 相関

| 指標 | 旧v3 | v3b |
|---|---|---|
| Spearman ρ | 0.417 | **0.586** |
| 改善率 | - | +41% |

### 残存する乖離の原因

MLのトップ候補（izakaya系）がv3bで1000-5000位にとどまるのは、MLがnon-linearな特徴量間の相互作用（genre × saturation × google_ratingの組み合わせ効果）を捉えているため。formula-basedの構造的限界であり、v4アンサンブルで補完される。

## 9. 重要な知見

1. **人口は飲食店数をほぼ説明しない**（SHAP < 0.01）。店舗数を決めるのは「そのエリアの商業集積度」と「ジャンル特性」
2. **ジャンル × 競合構造が支配的**。genre_encoded + other_genre_count で全体SHAP の37%
3. **Google特徴量が高寄与**（rating + reviews + density で SHAP 16%）。ただし札幌には欠損
4. **tokyoの予測精度が最低**（R²=0.732）。データ数は最大だが、繁華街から住宅街まで分散が大きい
5. **2段階残差モデル（Ridge→LightGBM）は不採用**。CV R²が0.795→0.739に悪化
6. **外れ値フィルタも不採用**。Optuna検証で全閾値においてフィルタなしがR²最良。外れ値7件は繁華街の正当データ
7. **サンプル重み付け（weight=log1p(rc)）も不採用**。全体R² 0.797→0.781に悪化
8. **rc=0行追加も不採用**。全体R²は0.913に上昇するが、rc≥1の全セグメントでRMSE悪化（Google特徴量が0/非0分類に使われる）
9. **2段階モデル（分類→回帰）も不採用**。rc=1が49.9%を占める構造的課題に対し、Stage1（rc≤1 vs rc≥2分類）+ Stage2（rc≥2回帰）を検証。Optuna 30trialチューニング後でも分類F1=0.830（17%誤分類）、rc≥2回帰R²=0.758で、全データ単一モデル（R²=0.797）に及ばず。2モデル管理の複雑さに見合わないため不採用。代替として予測信頼度（CI跨ぎベース）によるフィルタリングで実質的に対処
