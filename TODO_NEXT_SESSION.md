# TODO: 次セッション向けタスク

## 現状サマリ (2026-03-23)

### ブランチ・インフラ
- **ブランチ**: `feature/add-docstrings`（masterからの差分が大きい）
- **テスト**: 143 passed
- **GitHub認証**: `gh auth login` 済みだが `repo` スコープ不足 → PR作成に再認証が必要
- **未コミット変更**: なし（プッシュ済み: 4b8ecda）

### データパイプライン
- **5エリア対応**: tokyo / osaka / nagoya / fukuoka / sapporo
- **e-Stat**: ページネーション対応済み。全7人口・世帯カテゴリ（4分の1メッシュ 250m）取得可能
- **Google Places**: 4エリア取得済み（sapporo は Google データなし）
- **ジャンル**: izakaya / italian / chinese / yakiniku / cafe / ramen / washoku / curry / other

### MLモデル性能
| 指標 | 値 |
|------|-----|
| 5-Fold CV R² | 0.797 |
| 5-Fold CV RMSE | 0.309 |
| LOAO tokyo R² | 0.732（最低） |
| LOAO osaka R² | 0.797 |
| LOAO nagoya R² | 0.810 |
| LOAO fukuoka R² | 0.816（最高） |
| ML vs v3b Spearman | 0.586 |
| 特徴量数 | 28 |
| 学習データ | 9,665行（4エリア結合、sapporo除外） |

### SHAP特徴量重要度（上位10）
| 順位 | 特徴量 | mean |SHAP| |
|------|--------|-------------|
| 1 | genre_encoded | 0.219 |
| 2 | other_genre_count | 0.154 |
| 3 | google_avg_rating | 0.098 |
| 4 | neighbor_avg_restaurants | 0.078 |
| 5 | saturation_index | 0.065 |
| 6 | reviews_per_shop | 0.048 |
| 7 | genre_hhi | 0.044 |
| 8 | genre_diversity | 0.030 |
| 9 | google_density | 0.018 |
| 10 | commercial_density_rank | 0.015 |

---

## 前回セッションで完了した内容

（省略 — git log 9e832b2 以前を参照）

## 今回セッション (2026-03-23 後半) で完了した内容

### P2: コード品質改善
- [x] `genre_encoded` を `GENRE_ORDER` + `CategoricalDtype` で固定マッピングに修正
- [x] テスト追加: NaN人口ハンドリング、ジャンルマッピング整合性、未知ジャンル(-1)検証
- [x] `scripts/run_pipeline.py` の `ALL_TAGS` に `sapporo` 追加は不要と判断（Google Placesデータなし）

### P5: tokyo精度改善
- [x] 特徴量3つ追加: `commercial_density_rank`(4.8%), `station_accessibility`, `land_price` log変換
  - tokyo LOAO R²: 0.729 → 0.732 (+0.003)
- [x] サンプル重み付け（`weight=log1p(rc)`）検証 → **不採用**（全体R² 0.797→0.781、rc=1セグメント悪化）
- [x] rc=0行追加検証 → **不採用**（全体R²は0.913に上昇するが、rc≥1の全セグメントでRMSE悪化。Google特徴量が0/非0分類に使われ本来の学習を阻害）
- [x] Google特徴量（google_avg_rating, google_total_reviews）除外検証 → **不採用**（R² 0.797→0.760）
- [x] tokyo内サブエリア分割は見送り（既存特徴量が間接的に同じ役割を果たしている）

### P6: ダッシュボード機能拡充
- [x] ヒートマップのジャンルフィルタ → 既に動作済みを確認
- [x] ランキングテーブル追加（地図の下に一覧表、全フィルタ連動）
- [x] 全エリア横断比較ビュー追加（エリアドロップダウンに「全エリア比較」オプション）
- [x] MLギャップヒートマップ・人口ヒートマップ削除（不要）
- [x] メッシュグリッド線追加（JIS 4分の1メッシュ境界を薄いグレー線で描画）

### P1: コミット・プッシュ
- [x] 6ファイル +330/-60行をコミット・プッシュ（4b8ecda）

---

## 次にやるべきタスク

### P1: PR作成・マージ（未完了）
- [ ] `gh auth login --scopes repo` で再認証（repoスコープ必要）
- [ ] `feature/add-docstrings` → `master` への PR を作成
- [ ] レビュー後マージ、ブランチ削除

### P7: データ品質・運用
- [ ] 定期的なデータ収集の仕組み化（cron/スケジューラ）→ 将来の時系列データ蓄積
- [ ] HTMLレポート自動再生成（v3bスコアで全エリア）
- [ ] `add_area.py` の動作検証（実際に新エリアを追加してテスト）

### P8: モデル信頼性の改善
- [ ] **market_gapに信頼区間を付与**: foldごとの予測ばらつきからCI算出 → 推薦の信頼度表示
- [ ] **推薦フィルタ追加**: rc≥2のメッシュのみ、またはgap_count≥1.0のみを推薦対象にする等
- [ ] **market_gapの解釈問題**: 「予測>実際=機会」と「モデル誤差」を区別する仕組み
- [ ] rc=1が全データの49.9%を占める構造的課題 — 2段階モデル（分類→回帰）で解決できる可能性あり
- [ ] 時間軸の検証がない — 推薦の事後検証の仕組み検討

### P9: ダッシュボードのパフォーマンス・UX改善
- [ ] **表示速度**: `_load_and_score`で毎回`add_all_features`再計算 → キャッシュ済みCSVを直接使う等の最適化
- [ ] **メッシュグリッドのパフォーマンス**: tokyo 1,233メッシュのRectangle描画が重い可能性 → ズームレベルに応じた表示制御
- [ ] **推薦理由の質向上**: `generate_reason`が定型テンプレート → SHAPベースの個別説明で説得力を上げる

---

## プロジェクト構造（最新）

```
config/settings.py              # API設定・v3b重み（Optuna最適化済み）
src/collect/
  estat.py                      # e-Stat APIクライアント（ページネーション・cat01フィルタ対応）
  hotpepper.py                  # HotPepper APIクライアント
  collector.py                  # メッシュグリッド生成 + HotPepper統合収集
  station.py                    # 駅データ取得（HeartRails Express API）
  land_price.py                 # 地価公示データ取得（国土数値情報）
  station_passengers.py         # 駅別乗降客数データ（国土数値情報 S12）
  google_places.py              # Google Places APIクライアント（現在未使用）
  google_collector.py           # Google Placesメッシュベース収集
src/preprocess/
  cleaner.py                    # データクレンジング・ジャンルマッピング
  mesh_converter.py             # JIS 4分の1メッシュ変換（250m 10桁）
  google_matcher.py             # HotPepper↔Google Places座標マッチング
src/analyze/
  constants.py                  # 共通定数（_POP_UNIT, _COMPETITOR_OFFSET）
  utils.py                      # 共通ユーティリティ（mesh_col）
  features.py                   # 空間特徴量（28特徴量、cKDTree高速化済み）
  scoring.py                    # v1/v2/v3/v3b/v4 スコアリング（v3b: Optuna最適化済み）
  ml_model.py                   # LightGBM（LOAO CV, Optuna対応、GENRE_ORDER固定）
  backtest.py                   # バックテスト（未使用）
src/visualize/
  dashboard.py                  # Dash ダッシュボード（ランキングテーブル・全エリア比較・メッシュグリッド）
  report.py                     # HTMLレポート自動生成（ジャンルフィルタ・LayerControl）
scripts/
  add_area.py                   # 新エリア追加CLI（座標指定で一気通貫）
  tune_v3b_weights.py           # v3b重みOptuna最適化
  integrate_estat.py            # e-Stat統合パイプライン（v3b対応）
  train_model.py                # MLモデル学習CLI（v3b比較対応）
  run_pipeline.py               # 統合パイプラインCLI
  collect_area.py / collect_google.py / fetch_external.py
  validate_scores.py / generate_report.py
docs/
  ml_model_description.md       # MLモデル説明書
  system_overview.md            # システム概要
models/
  combined_lightgbm.txt         # 学習済みモデル（4エリア結合）
data/
  raw/                          # 生データ（HotPepper/e-Stat/Google Places）
  processed/                    # 統合・スコアリング済みCSV + MLギャップ
  external/                     # 外部データキャッシュ（駅・地価・乗降客数）
tests/                          # 143 tests
```

## 環境変数（設定済み）
- `HOTPEPPER_API_KEY`: settings.py にデフォルト値設定済み
- `ESTAT_API_KEY`: settings.py にデフォルト値設定済み
- `GOOGLE_PLACES_API_KEY`: コメントアウト済み（勝手にAPIアクセスしないこと）
