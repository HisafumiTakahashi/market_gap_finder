# 次のセッションでやること

## 現在のステータス

- **ブランチ**: `feature/add-docstrings`
- **パイプライン**: 東京・大阪・名古屋の3エリア統合済み
- **スコアリング**: v1(店舗数) / v2(人口) / v3(空間特徴量+駅距離) 実装済み
- **MLモデル**: LightGBM 市場容量予測（R2=0.74〜0.84）
- **ダッシュボード**: v1/v2/v3切替 + 人口ヒートマップ + 空間特徴量ツールチップ
- **レポート**: 3エリアHTML自動生成済み（`reports/`）
- **テスト**: 51件全パス

### 即実行コマンド（キャッシュ利用）
```bash
# ダッシュボード起動
PYTHONPATH=. py -c "from src.visualize.dashboard import run_dashboard; run_dashboard(tag='tokyo')"

# スコア検証
PYTHONPATH=. py scripts/validate_scores.py --tag tokyo --top-n 20

# MLモデル学習
PYTHONPATH=. py scripts/train_model.py --tag tokyo --top-n 20

# レポート全エリア生成
PYTHONPATH=. py scripts/generate_report.py --all --top-n 20
```

---

## P1: 新規モジュールのテスト追加（信頼性向上）

### 目的
今回追加したモジュール（station.py, land_price.py, ml_model.py, report.py）にテストがない。
カバレッジを上げて回帰テストの安全網を作る。

### 対象
- `tests/test_station.py`: fetch_lines/fetch_stations のモック、haversine_km の精度検証
- `tests/test_land_price.py`: GMLパース、キャッシュ保存/読み込み
- `tests/test_ml_model.py`: prepare_features、train_cv（小データ）、compute_market_gap
- `tests/test_report.py`: generate_explanation の出力検証、generate_report のHTML構造

---

## P2: neighbor_avg_restaurants の分散ゼロ問題を修正

### 現象
全3エリアで `neighbor_avg_restaurants` のピアソン相関が N/A。
validate_scores.py の相関計算時に分散がゼロになっている。

### 原因候補
- `add_neighbor_competition` が隣接メッシュの店舗数を計算するが、
  統合パイプラインの呼び出し順で `jis_mesh3` 単位の合計が正しく取れていない可能性
- 3次メッシュの末尾 ±1 による近傍計算が2次メッシュ境界を跨げない

### 対応
- `features.py` の `_neighbor_mesh_codes` を2次メッシュ境界跨ぎ対応に拡張
- 統合後の `neighbor_avg_restaurants` の分布を確認するデバッグスクリプトを追加

---

## P3: 地価公示データの代替取得

### 現象
国土数値情報 API（`getGMLStList`）が全都道府県でダウンロードURL返却に失敗。
API仕様変更またはサービス停止の可能性。

### 代替案（優先順）
1. **国土数値情報の直接URL構築**: `https://nlftp.mlit.go.jp/ksj/gml/data/L01/L01-24/` のファイル一覧からZIPを特定してダウンロード
2. **SUUMO/不動産情報ライブラリのスクレイピング**: 地価公示ポイント座標+価格を取得
3. **手動ダウンロード対応**: `data/external/` に手動配置したCSVを読み込むフロー（`land_price.py` は対応済み）

### 効果
- `land_price` 特徴量が有効化されると v3 スコアの分母に反映（高地価=コスト圧）
- ML モデルの説明力向上が期待される（現在7特徴量→8特徴量）

---

## P4: MLモデルの改善

### クロスエリア汎化
- 東京で学習→大阪・名古屋で予測（地域外汎化性能の確認）
- 3エリア統合データでの学習 → 単一モデルの構築

### アンサンブル
- v3ルールベース と MLギャップ のランク加重平均で統合スコアを作成
- `scoring.py` に `compute_opportunity_score_v4(df, ml_gap)` を追加

### ハイパーパラメータチューニング
- Optuna で LightGBM のパラメータ探索
- 特に `num_leaves`, `learning_rate`, `min_child_samples` を最適化

### ラベル改善
- Google Places API から評価・レビュー数を取得（HotPepperにはデータなし）
- `config/settings.py` の `GOOGLE_PLACES_API_KEY` を有効化
- 既存の `src/collect/google_places.py` を活用

---

## P5: ダッシュボードにMLギャップ表示を追加

### 対応内容
- Map Mode ドロップダウンに「ML Gap Heatmap」を追加
- `market_gap` を重みとしたヒートマップ表示
- マーカーポップアップに `predicted_count`, `gap_count` を追加
- Score Version に `v4 (ML+v3アンサンブル)` を追加（P4完了後）

---

## P6: エリア間比較分析ノートブック

### 目的
3エリアの特徴量分布・スコア分布を横断比較し、都市構造の違いを可視化する。

### 内容
- `notebooks/04_cross_area_comparison.ipynb` を作成
- 人口分布、飽和度分布、駅距離分布のエリア間比較ヒストグラム
- v3 上位候補のジャンル構成比較
- MLモデルの特徴量重要度のエリア間比較
- SHAP summary plot（ビジュアル）

---

## 既知の課題

| 課題 | 詳細 | 優先度 |
|------|------|--------|
| `neighbor_avg_restaurants` 分散ゼロ | 全3エリアで相関N/A。近傍計算ロジックを要修正 | 高 |
| 地価データ未取得 | 国土数値情報API障害。代替手段を要検討 | 中 |
| 新規モジュールにテストなし | station/land_price/ml_model/report のテスト未実装 | 高 |
| HotPepper rating/review が空 | API仕様上データなし。Google Places で代替可能 | 中 |
| `backtest.py` が未使用 | 開店履歴データがないと動作しない | 低 |
| sklearn `squared` 非推奨警告 | `root_mean_squared_error` に移行必要 | 低 |
| `google_places.py` 未使用 | P4ラベル改善で活用予定 | 低 |

---

## 環境変数（設定済み）
- `HOTPEPPER_API_KEY`: settings.py にデフォルト値設定済み
- `ESTAT_API_KEY`: settings.py にデフォルト値設定済み
- `GOOGLE_PLACES_API_KEY`: 未設定（P4ラベル改善で必要）

## プロジェクト構造（最新）

```
config/settings.py              # API設定・パラメータ一元管理
src/collect/
  hotpepper.py                  # HotPepper APIクライアント
  estat.py                      # e-Stat APIクライアント（動的statsDataId検索）
  station.py                    # 駅データ取得（HeartRails Express API）   ← NEW
  land_price.py                 # 地価公示データ取得（国土数値情報）        ← NEW
  google_places.py              # Google Places（未使用）
src/preprocess/
  cleaner.py                    # データクレンジング
  mesh_converter.py             # JIS 3次メッシュ変換
src/analyze/
  features.py                   # 空間特徴量（多様性・HHI・隣接競合・飽和度・駅距離・地価）
  scoring.py                    # v1/v2/v3 スコアリング
  ml_model.py                   # LightGBM 市場容量予測モデル             ← NEW
  backtest.py                   # バックテスト（未使用）
src/visualize/
  dashboard.py                  # Dash ダッシュボード（v1/v2/v3対応）
  report.py                     # HTMLレポート自動生成                     ← NEW
scripts/
  collect_area.py               # データ収集CLI
  integrate_estat.py            # e-Stat統合パイプライン（駅データ連携対応）
  validate_scores.py            # スコア検証・比較ツール
  fetch_external.py             # 外部データ（駅・地価）取得CLI           ← NEW
  train_model.py                # LightGBMモデル学習CLI                    ← NEW
  generate_report.py            # HTMLレポート生成CLI                      ← NEW
models/
  {tag}_lightgbm.txt            # 学習済みモデル（tokyo/osaka/nagoya）
reports/
  {tag}_report.html             # 自動生成HTMLレポート
data/
  raw/                          # 生データ（HotPepper/e-Stat）
  processed/                    # 統合・スコアリング済みCSV + MLギャップ
  external/                     # 外部データキャッシュ（駅・地価）
tests/
  test_estat.py                 # 6 tests
  test_features.py              # 15 tests
  test_scoring.py               # 30 tests
notebooks/
  01_collection_eda.ipynb       # HotPepper EDA
  02_estat_demand_data.ipynb    # e-Stat需要データ
  03_estat_eda.ipynb            # e-Stat EDA
```
