# 次のセッションでやること

## 現在のステータス

- **ブランチ**: `feature/add-docstrings`
- **パイプライン**: 東京・大阪・名古屋・福岡の4エリア統合済み（札幌はGoogle Placesデータなし）
- **Google Places**: 4エリア取得済み（tokyo:6,441 / osaka:5,151 / nagoya:3,582 / fukuoka:4,019）
- **店舗マッチング**: HotPepper個店にGoogle Places rating/review_countを座標ベース1対多マッチ（87-91%マッチ率）
- **MLモデル**: LightGBM 2段階残差モデル（4エリア結合、LOAO R2=0.838）
- **特徴量**: 16個（Google Places由来3個含む: google_avg_rating, reviews_per_shop, google_density）
- **テスト**: 96件全パス

---

## P0: 250mメッシュへの移行（最優先）

### 目的
より細かい立地分析とデータ量増加のため、分析単位を1km→250mメッシュに変更する。

### 対応内容
1. **e-Stat API**: `_find_mesh_stats_data_id` を修正し、4分の1メッシュ（250m、10桁）のstatsDataIdを取得する
   - 現在は「最少レコード数 = 3次メッシュ」を選択 → 「最多レコード数 = 4分の1メッシュ」に変更
2. **メッシュ変換**: `mesh_converter.py` に250mメッシュ（10桁）対応を追加
   - `lat_lon_to_mesh_quarter(lat, lng)` → 10桁メッシュコード
3. **データ再取得**: e-Stat人口データを250mメッシュで再取得（`--skip-fetch` なしで実行）
4. **統合パイプライン**: `integrate_estat.py` を250mメッシュ基準に修正
   - HotPepper/Google Placesの店舗を250mメッシュに割当
   - 特徴量計算（近傍メッシュ等）を250m単位に修正
5. **MLモデル再学習**: 250mメッシュデータで再学習・評価

### 効果
- データ量: 約16倍（tokyo: 927行 → 約15,000行）
- 人口と店舗が同じ粒度で対応
- より精細な市場ギャップ検出

### 注意
- Google Places APIへのアクセスは禁止（既取得データを再利用）
- APIキーはコメントアウト済み

---

## P1: 新規モジュールのテスト追加

### 対象
- `tests/test_station.py`: fetch_lines/fetch_stations のモック、haversine_km の精度検証
- `tests/test_land_price.py`: GMLパース、キャッシュ保存/読み込み
- `tests/test_report.py`: generate_explanation の出力検証、generate_report のHTML構造

---

## P2: neighbor_avg_restaurants の分散ゼロ問題を修正

### 現象
全エリアで `neighbor_avg_restaurants` のピアソン相関が N/A。
validate_scores.py の相関計算時に分散がゼロになっている。

### 対応
- `features.py` の `_neighbor_mesh_codes` を2次メッシュ境界跨ぎ対応に拡張
- 統合後の `neighbor_avg_restaurants` の分布を確認するデバッグスクリプトを追加

---

## P3: MLモデルの改善

### アンサンブル
- v3ルールベース と MLギャップ のランク加重平均で統合スコアを作成
- `scoring.py` に `compute_opportunity_score_v4(df, ml_gap)` を追加

### ハイパーパラメータチューニング
- Optuna で LightGBM のパラメータ探索（P0完了後に実施）

---

## P4: ダッシュボードにMLギャップ表示を追加

### 対応内容
- Map Mode ドロップダウンに「ML Gap Heatmap」を追加
- `market_gap` を重みとしたヒートマップ表示
- マーカーポップアップに `predicted_count`, `gap_count` を追加

---

## P5: エリア間比較分析ノートブック

### 内容
- `notebooks/04_cross_area_comparison.ipynb` を作成
- 人口分布、飽和度分布、駅距離分布のエリア間比較
- MLモデルの特徴量重要度のエリア間比較
- SHAP summary plot

---

## 既知の課題

| 課題 | 詳細 | 優先度 |
|------|------|--------|
| neighbor_avg_restaurants 分散ゼロ | 全エリアで相関N/A。近傍計算ロジックを要修正 | 高 |
| 新規モジュールにテストなし | station/land_price/report のテスト未実装 | 中 |
| backtest.py が未使用 | 開店履歴データがないと動作しない | 低 |
| sklearn `squared` 非推奨警告 | `root_mean_squared_error` に移行必要 | 低 |
| 札幌Google Placesデータ未取得 | API無料枠不足で未取得。翌月以降に対応可 | 低 |

---

## 環境変数（設定済み）
- `HOTPEPPER_API_KEY`: settings.py にデフォルト値設定済み
- `ESTAT_API_KEY`: settings.py にデフォルト値設定済み
- `GOOGLE_PLACES_API_KEY`: コメントアウト済み（勝手にAPIアクセスしないこと）

## プロジェクト構造（最新）

```
config/settings.py              # API設定・パラメータ一元管理
src/collect/
  hotpepper.py                  # HotPepper APIクライアント
  estat.py                      # e-Stat APIクライアント（動的statsDataId検索）
  collector.py                  # メッシュグリッド生成 + HotPepper統合収集
  station.py                    # 駅データ取得（HeartRails Express API）
  land_price.py                 # 地価公示データ取得（国土数値情報）
  station_passengers.py         # 駅別乗降客数データ（国土数値情報 S12）
  google_places.py              # Google Places APIクライアント
  google_collector.py           # Google Placesメッシュベース収集
src/preprocess/
  cleaner.py                    # データクレンジング
  mesh_converter.py             # JIS 3次メッシュ変換
  google_matcher.py             # HotPepper↔Google Places座標マッチング    ← NEW
src/analyze/
  features.py                   # 空間特徴量（多様性・HHI・隣接競合・飽和度・駅距離・地価）
  scoring.py                    # v1/v2/v3 スコアリング
  ml_model.py                   # LightGBM 2段階残差モデル（16特徴量）
  backtest.py                   # バックテスト（未使用）
src/visualize/
  dashboard.py                  # Dash ダッシュボード
  report.py                     # HTMLレポート自動生成
scripts/
  collect_area.py               # HotPepperデータ収集CLI
  collect_google.py             # Google Placesデータ収集CLI
  integrate_estat.py            # e-Stat統合パイプライン（店舗マッチング対応）
  validate_scores.py            # スコア検証・比較ツール
  fetch_external.py             # 外部データ（駅・地価・乗降客数）取得CLI
  train_model.py                # LightGBMモデル学習CLI（4エリア結合・LOAO対応）
  generate_report.py            # HTMLレポート生成CLI
models/
  combined_lightgbm.txt         # 学習済みモデル（4エリア結合）
data/
  raw/                          # 生データ（HotPepper/e-Stat/Google Places）
  processed/                    # 統合・スコアリング済みCSV + MLギャップ
  external/                     # 外部データキャッシュ（駅・地価・乗降客数）
tests/                          # 96 tests
  test_estat.py, test_features.py, test_scoring.py,
  test_ml_model.py, test_station_passengers.py, test_google_matcher.py
```
