# TODO: 次セッション向けタスク

## 現状サマリ (2026-03-24)

### ブランチ・インフラ
- **ブランチ**: `feature/add-docstrings`（masterからの差分が大きい）
- **テスト**: 153 passed
- **GitHub認証**: `gh auth login` 済みだが `repo` スコープ不足 → PR作成に再認証が必要
- **最新コミット**: 0c3afda（プッシュ済み）

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
| ML vs v3b Spearman | 0.598 |
| 特徴量数 | 28 |
| 学習データ | 9,665行（4エリア結合、sapporo除外） |

### 予測信頼度分布（combined）
| 信頼度 | 件数 | 割合 | 判定基準 |
|--------|------|------|----------|
| high | 4,843 | 50.1% | CI下限 > 0（統計的に有意な出店余地） |
| medium | 1,148 | 11.9% | gap > 0 だがCIが0を跨ぐ |
| low | 3,674 | 38.0% | gap ≤ 0（出店余地なし） |

---

## 今回セッション (2026-03-24) で完了した内容

### P8: モデル信頼性の改善
- [x] **予測信頼度（CI跨ぎベース）実装**: train_cvでfold別全データ予測 → 95%CI算出 → CI下限>0で信頼度判定
  - 当初はgap_stdの絶対閾値方式（high≤0.1, medium≤0.2）で実装
  - high=71%, low=1%と偏りが大きく識別力不足 → CI跨ぎベース（方法2）に変更
  - 変更後: high=50.1%, medium=11.9%, low=38.0%でバランス良好
- [x] **推薦フィルタ**: `filter_recommendations(min_rc=2, min_gap_count=1.0, min_reliability="medium")`
- [x] **ダッシュボード信頼度フィルタ**: 「高のみ / 高+中 / すべて」の3段階UI
- [x] **ポップアップに信頼度表示**: 高（統計的に有意な出店余地）/ 中（不確実）/ 低（出店余地なし）
- [x] **2段階モデル（分類→回帰）検証 → 不採用**
  - Optuna 30trialチューニング後: 分類F1=0.830（17%誤分類）、rc≥2回帰R²=0.758
  - 全データ単一モデル（R²=0.797）に及ばず、2モデル管理の複雑さに見合わない
  - 予測信頼度フィルタで実質的に対処済み
- [x] **テスト追加**: 153テスト全パス（+10件: CI計算、境界値、フィルタ）
- [x] **ドキュメント更新**: docs配下5ファイル更新・新規作成
  - `prediction_reliability.md`（新規）: 信頼度の仕組み詳細
  - `model_explanation.md`（新規）: モデル解説
  - `model_features.md`（新規）: 特徴量詳細
  - `ml_model_description.md`: 信頼度・2段階モデル検証結果追記
  - `system_overview.md`: 信頼度・テスト数・課題更新

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

### P8: モデル信頼性の改善（残り）
- [ ] 時間軸の検証がない — 推薦の事後検証の仕組み検討（P7のデータ蓄積が前提）

### P9: ダッシュボードのパフォーマンス・UX改善
- [ ] **表示速度**: `_load_and_score`で毎回`add_all_features`再計算 → キャッシュ済みCSVを直接使う等の最適化
- [ ] **メッシュグリッドのパフォーマンス**: tokyo 1,233メッシュのRectangle描画が重い可能性 → ズームレベルに応じた表示制御
- [ ] **推薦理由の質向上**: `generate_reason`が定型テンプレート → SHAPベースの個別説明で説得力を上げる

---

## 検証済み・不採用にした施策

| 施策 | 結果 | 理由 |
|------|------|------|
| サンプル重み付け（weight=log1p(rc)） | 不採用 | 全体R² 0.797→0.781に悪化 |
| rc=0行追加 | 不採用 | rc≥1全セグメントでRMSE悪化 |
| Google特徴量除外 | 不採用 | R² 0.797→0.760に悪化 |
| 2段階残差モデル（Ridge→LightGBM） | 不採用 | CV R² 0.795→0.739に悪化 |
| tokyo内サブエリア分割 | 見送り | 既存特徴量が間接的に同じ役割 |
| 外れ値フィルタ | 不採用 | 全閾値でフィルタなしがR²最良 |
| 2段階モデル（分類→回帰） | 不採用 | 分類F1=0.830、rc≥2回帰R²=0.758。単一モデルR²=0.797に及ばず |
| gap_std絶対閾値方式 | 変更 | high=71%, low=1%で識別力不足 → CI跨ぎベースに変更 |

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
  ml_model.py                   # LightGBM（LOAO CV, Optuna, 予測信頼度, 推薦フィルタ）
  backtest.py                   # バックテスト（未使用）
src/visualize/
  dashboard.py                  # Dash ダッシュボード（信頼度フィルタ・ランキングテーブル・全エリア比較・メッシュグリッド）
  report.py                     # HTMLレポート自動生成（ジャンルフィルタ・LayerControl）
scripts/
  add_area.py                   # 新エリア追加CLI（座標指定で一気通貫）
  tune_v3b_weights.py           # v3b重みOptuna最適化
  integrate_estat.py            # e-Stat統合パイプライン（v3b対応）
  train_model.py                # MLモデル学習CLI（CI付き推薦表示・信頼度サマリ）
  run_pipeline.py               # 統合パイプラインCLI
  collect_area.py / collect_google.py / fetch_external.py
  validate_scores.py / generate_report.py
docs/
  ml_model_description.md       # MLモデル説明書（信頼度・不採用施策含む）
  model_explanation.md          # モデル解説（スコアリング変遷・信頼度）
  model_features.md             # 特徴量詳細
  prediction_reliability.md     # 予測信頼度の仕組み詳細
  system_overview.md            # システム概要
models/
  combined_lightgbm.txt         # 学習済みモデル（4エリア結合）
data/
  raw/                          # 生データ（HotPepper/e-Stat/Google Places）
  processed/                    # 統合・スコアリング済みCSV + CI付きMLギャップ
  external/                     # 外部データキャッシュ（駅・地価・乗降客数）
tests/                          # 153 tests
```

## 環境変数（設定済み）
- `HOTPEPPER_API_KEY`: settings.py にデフォルト値設定済み
- `ESTAT_API_KEY`: settings.py にデフォルト値設定済み
- `GOOGLE_PLACES_API_KEY`: コメントアウト済み（勝手にAPIアクセスしないこと）
