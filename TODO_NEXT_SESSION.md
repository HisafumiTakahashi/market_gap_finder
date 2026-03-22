# TODO: 次セッション向けタスク

## 現状サマリ (2026-03-22)

### ブランチ・インフラ
- **ブランチ**: `feature/add-docstrings`（masterからの差分が大きい）
- **テスト**: 128 passed
- **GitHub認証**: `gh auth login` 未実施 → PR作成にはこれが必要

### データパイプライン
- **5エリア対応**: tokyo / osaka / nagoya / fukuoka / sapporo
- **e-Stat**: ページネーション対応済み。全7人口・世帯カテゴリ（4分の1メッシュ 250m）取得可能
- **Google Places**: 4エリア取得済み（sapporo は Google データなし）
- **ジャンル**: izakaya / italian / chinese / yakiniku / cafe / ramen / washoku / curry / other

### MLモデル性能
| 指標 | 値 |
|------|-----|
| LOAO CV R2 | 0.795 |
| LOAO CV RMSE | 0.310 |
| ML vs v3 Spearman | 0.388 |
| 特徴量数 | 26 |
| 学習データ | 9,665行（5エリア結合） |

### 今回のセッションで完了した内容

#### e-Stat API改善
- `get_stats_data` にページネーション追加（`startPosition` ループ）
- cat01 日本語名→コード正規化（`_CAT01_PATTERN_TO_CODE` 部分一致、順序考慮）
- CSVの `dtype` 指定で先頭ゼロ消失を防止

#### コード品質改善（13件）
1. 特徴量リーケージ修正（`neighbor_avg_score` 除去）
2. 定数二重定義の統合（`POPULATION_*` を `estat.py` に一元化）
3. `_mesh_col` 統合（`src/analyze/utils.py` に共通実装、5ファイルから import）
4. ジャンルマッピング修正（`sushi`→`washoku`、テスト・キャッシュ含め全更新）
5. API失敗時の warning ログ追加
6. population 欠損の扱い（`==0.0` → `.isna()` で実際の 0 と欠損を区別）
7. cKDTree 高速化（最寄り駅 O(n×m) → O(n log m)）
8. マジックナンバー抽出（`constants.py` に集約）
9. HHI 計算コメント追加
10. 循環 import 解消（`constants.py` 分離）
11. `dashboard.py` / `report.py` の `_mesh_col` 統合
12. テストの "sushi"→"washoku" 更新
13. キャッシュCSV全エリア再生成

#### パラメータ最適化（Optuna 検証済み、根拠コメント付き）
| パラメータ | 旧値 | 新値 | 根拠 |
|-----------|------|------|------|
| `_COMPETITOR_OFFSET` | 0.1 | 0.064 | Optuna 30trial。旧値は正規化人口中央値の104%で過剰平滑化 |
| `filter_outliers` | True | False | 全閾値でフィルタなしが R2 最良。外れ値7件は繁華街の正当データ |
| `WEIGHT_DEMAND` | 1.0 | 0.17 | Optuna 100trial、v3-ML Spearman +105% |
| `WEIGHT_COMPETITOR` | 1.0 | 2.76 | 同上。供給過多ペナルティが最重要 |
| `WEIGHT_POPULATION` | 1.0 | 0.18 | 同上。需要と高相関のため低く抑制 |
| `WEIGHT_LAND_PRICE` | 1.0 | 1.65 | 同上。低地価＝参入コスト低のボーナス |

---

## 次にやるべきタスク

### P1: PR作成・マージ
- [ ] `gh auth login` で GitHub 認証を設定
- [ ] `feature/add-docstrings` → `master` への PR を作成
- [ ] レビュー後マージ、ブランチ削除

### P2: 残存する軽微なコード問題
- [ ] `genre_encoded` の `.cat.codes` がデータセットごとに異なる値を生成しうる
  → LabelEncoder で固定マッピングにするか、カテゴリ順序を明示的に定義
- [ ] `scripts/run_pipeline.py` の `ALL_TAGS` に `sapporo` を追加
- [ ] テストカバレッジ拡充: NaN 人口ハンドリング、ジャンルマッピング検証

### P3: モデル改善
- [ ] エリア別精度分析（sapporo は人口データ少→精度低下の可能性）
- [ ] v3 スコアと ML ギャップの相関 0.388 → スコアリングロジックの根本見直し
- [ ] 2段階残差モデル（Ridge→LightGBM）の効果再検証
- [ ] 時系列特徴量の検討（開店・閉店トレンド等、データ入手可能性次第）

### P4: プロダクト改善
- [ ] HTML レポートにジャンル別フィルタ機能追加
- [ ] 地図ビジュアライゼーション（folium 等）
- [ ] 新エリア追加の自動化（座標指定だけで一気通貫）

---

## プロジェクト構造（最新）

```
config/settings.py              # API設定・スコアリング重み（Optuna最適化済み）
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
  constants.py                  # 共通定数（_POP_UNIT, _COMPETITOR_OFFSET）  ← NEW
  utils.py                      # 共通ユーティリティ（mesh_col）             ← NEW
  features.py                   # 空間特徴量（26特徴量、cKDTree高速化済み）
  scoring.py                    # v3 スコアリング（重み最適化済み）
  ml_model.py                   # LightGBM（LOAO CV, Optuna対応）
  backtest.py                   # バックテスト（未使用）
src/visualize/
  dashboard.py                  # Dash ダッシュボード
  report.py                     # HTMLレポート自動生成
scripts/
  integrate_estat.py            # e-Stat統合パイプライン
  train_model.py                # MLモデル学習CLI（5エリア結合・LOAO対応）
  run_pipeline.py               # 統合パイプラインCLI
  collect_area.py / collect_google.py / fetch_external.py
  validate_scores.py / generate_report.py
models/
  combined_lightgbm.txt         # 学習済みモデル（5エリア結合）
data/
  raw/                          # 生データ（HotPepper/e-Stat/Google Places）
  processed/                    # 統合・スコアリング済みCSV + MLギャップ
  external/                     # 外部データキャッシュ（駅・地価・乗降客数）
tests/                          # 128 tests
```

## 環境変数（設定済み）
- `HOTPEPPER_API_KEY`: settings.py にデフォルト値設定済み
- `ESTAT_API_KEY`: settings.py にデフォルト値設定済み
- `GOOGLE_PLACES_API_KEY`: コメントアウト済み（勝手にAPIアクセスしないこと）
