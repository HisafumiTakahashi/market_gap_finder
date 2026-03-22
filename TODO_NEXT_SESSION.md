# TODO: 次セッション向けタスク

## 現状サマリ (2026-03-23)

### ブランチ・インフラ
- **ブランチ**: `feature/add-docstrings`（masterからの差分が大きい）
- **テスト**: 134 passed
- **GitHub認証**: `gh auth login` 済みだが `repo` スコープ不足 → PR作成に再認証が必要
- **未コミット変更**: 12ファイル（v3bスコアリング、ダッシュボード改善、新規CLI等）

### データパイプライン
- **5エリア対応**: tokyo / osaka / nagoya / fukuoka / sapporo
- **e-Stat**: ページネーション対応済み。全7人口・世帯カテゴリ（4分の1メッシュ 250m）取得可能
- **Google Places**: 4エリア取得済み（sapporo は Google データなし）
- **ジャンル**: izakaya / italian / chinese / yakiniku / cafe / ramen / washoku / curry / other

### MLモデル性能
| 指標 | 値 |
|------|-----|
| 5-Fold CV R² | 0.795 |
| 5-Fold CV RMSE | 0.310 |
| LOAO tokyo R² | 0.729（最低） |
| LOAO osaka R² | 0.796 |
| LOAO nagoya R² | 0.815（最高） |
| LOAO fukuoka R² | 0.810 |
| ML vs v3b Spearman | 0.586 |
| 特徴量数 | 26 |
| 学習データ | 9,665行（4エリア結合、sapporo除外） |

---

## 前回セッション (2026-03-22) で完了した内容

（省略 — 前回セッションの内容は git log 342ad6b 以前を参照）

## 今回セッション (2026-03-23) で完了した内容

### P3: モデル改善
- [x] **エリア別精度分析**: LOAO CV 実施。tokyo R²=0.729（最低）、nagoya/fukuoka 0.81+
- [x] **v3bスコアリング新設**: ratio形式→加算型log空間genre gapに構造変更
  - `genre_log_gap`（ジャンル別log空間の期待値差）を導入
  - `other_genre_count` を最重要シグナルに格上げ
  - Optuna 200trialで重み最適化 → Spearman 0.417→0.586（+41%）
- [x] **2段階残差モデル検証**: CV R²=0.739（悪化）→ **不採用**
- [x] **時系列特徴量検討**: HotPepper APIに履歴データなし → **データ入手不可で見送り**
- [x] **パイプライン統合**: integrate_estat.py, train_model.py を v3b 対応に更新

### P4: プロダクト改善
- [x] **HTMLレポート**: ジャンル別フィルタ（JS）、CircleMarker色分け、LayerControl追加
- [x] **地図ビジュアライゼーション**: スコア色分け + market_gapサイズ + ジャンルレイヤー切替
- [x] **新エリア追加CLI**: `scripts/add_area.py` — 座標指定で一気通貫パイプライン
- [x] **ダッシュボード改善**:
  - エリア切替ドロップダウン追加（data/processed/ 自動検出）
  - 全UIラベル日本語化（エリア名、ジャンル名、地図モード等）
  - マーカー表示を「#N ジャンル（最寄駅付近）」に変更
  - Google Mapリンク追加（ポップアップから直接開ける）
  - 理由テキスト日本語化（全CSV + ダッシュボード動的再生成）
  - v1/v2スコアバージョン切替バグ修正
  - competitor_density→restaurant_count表示バグ修正
  - デフォルトスコアバージョンをv4に変更
- [x] **MLモデル説明書**: `docs/ml_model_description.md` 作成

### 新規ファイル
- `scripts/add_area.py` — 新エリア追加CLI
- `scripts/tune_v3b_weights.py` — v3b重みOptuna最適化スクリプト
- `docs/ml_model_description.md` — MLモデル説明書

---

## 次にやるべきタスク

### P1: PR作成・マージ（未完了）
- [ ] `gh auth login --scopes repo` で再認証（repoスコープ必要）
- [ ] 今回セッションの変更をコミット
- [ ] `feature/add-docstrings` → `master` への PR を作成
- [ ] レビュー後マージ、ブランチ削除

### P2: 残存する軽微なコード問題
- [ ] `genre_encoded` の `.cat.codes` がデータセットごとに異なる値を生成しうる
  → LabelEncoder で固定マッピングにするか、カテゴリ順序を明示的に定義
- [ ] `scripts/run_pipeline.py` の `ALL_TAGS` に `sapporo` を追加（Google Places除外のまま）
- [ ] `scripts/train_model.py` の `ALL_TAGS` にも `sapporo` 未追加（意図的: Google Placesデータなし）
- [ ] テストカバレッジ拡充: NaN 人口ハンドリング、ジャンルマッピング検証

### P5: tokyo精度改善
- [ ] tokyo R²=0.729 が全エリア最低。データ数は最大だが分散も最大
- [ ] tokyo内サブエリア分割（繁華街/住宅街/郊外）の特徴量追加を検討
- [ ] エリア固有の特徴量（例: エリアダミー変数）の効果検証

### P6: ダッシュボード追加機能
- [ ] ヒートマップモードでもジャンルフィルタが効くか検証（現状マーカーのみtop_n制限）
- [ ] ランキングテーブルをダッシュボード内に追加（地図の下に一覧表）
- [ ] エリア横断比較ビュー（全エリアのtop候補を1画面で比較）

### P7: データ品質・運用
- [ ] 定期的なデータ収集の仕組み化（cron/スケジューラ）→ 将来の時系列データ蓄積
- [ ] HTMLレポート自動再生成（v3bスコアで全エリア）
- [ ] `add_area.py` の動作検証（実際に新エリアを追加してテスト）

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
  features.py                   # 空間特徴量（26特徴量、cKDTree高速化済み）
  scoring.py                    # v1/v2/v3/v3b/v4 スコアリング（v3b: Optuna最適化済み）
  ml_model.py                   # LightGBM（LOAO CV, Optuna対応）
  backtest.py                   # バックテスト（未使用）
src/visualize/
  dashboard.py                  # Dash ダッシュボード（エリア切替・日本語UI・Google Map連携）
  report.py                     # HTMLレポート自動生成（ジャンルフィルタ・LayerControl）
scripts/
  add_area.py                   # 新エリア追加CLI（座標指定で一気通貫）  ← NEW
  tune_v3b_weights.py           # v3b重みOptuna最適化                    ← NEW
  integrate_estat.py            # e-Stat統合パイプライン（v3b対応）
  train_model.py                # MLモデル学習CLI（v3b比較対応）
  run_pipeline.py               # 統合パイプラインCLI
  collect_area.py / collect_google.py / fetch_external.py
  validate_scores.py / generate_report.py
docs/
  ml_model_description.md       # MLモデル説明書                          ← NEW
  system_overview.md            # システム概要
models/
  combined_lightgbm.txt         # 学習済みモデル（4エリア結合）
data/
  raw/                          # 生データ（HotPepper/e-Stat/Google Places）
  processed/                    # 統合・スコアリング済みCSV + MLギャップ
  external/                     # 外部データキャッシュ（駅・地価・乗降客数）
tests/                          # 134 tests
```

## 環境変数（設定済み）
- `HOTPEPPER_API_KEY`: settings.py にデフォルト値設定済み
- `ESTAT_API_KEY`: settings.py にデフォルト値設定済み
- `GOOGLE_PLACES_API_KEY`: コメントアウト済み（勝手にAPIアクセスしないこと）
