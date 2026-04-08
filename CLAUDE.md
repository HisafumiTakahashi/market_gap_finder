# CLAUDE.md — market_gap_finder

このファイルは Claude Code がこのリポジトリで作業する際の**必須ルール**をまとめたものです。
すべての回答・ドキュメント・コミットメッセージは **日本語** で記述してください。

---

## 1. Git ワークフロー（厳守）

- **master への直接コミット禁止**。作業開始時に必ずブランチを切る。
  ```bash
  git checkout -b feature/xxx    # または fix/xxx, docs/xxx
  ```
- セッション開始時、現在ブランチが `master` の場合は必ず新規ブランチを作成してから作業する。
- 完了後は PR 作成（`gh pr create`）またはユーザー指示でマージ。
- **force push / reset --hard / branch -D はユーザー明示指示時のみ**。

---

## 2. 駆動方針（Claude + Codex 分業）

このプロジェクトは **Claude が設計・レビュー、Codex が実装・実行** を担当する。
**コード変更・スクリプト実行・テスト等は基本的に Codex に依頼し、Claude Code で直接実行しない。**

### 実装依頼の標準形
```
mcp__codex__codex を呼び出す際のパラメータ:
  sandbox: "workspace-write"         ← デフォルト（ユーザー指示）
  approval-policy: "on-failure"      ← コマンド失敗時のみ承認を求める
```

### ループ抑制ルール
- Codex への**フィードバックは最大 3 回まで**。
- 3 回目でも問題が解決しない場合は、ユーザーに状況報告して相談する。**無限ループ禁止**。
- 問題点は毎回具体的に指摘する（「直して」だけはNG）。

### 役割分担
| 担当 | 作業 |
|------|------|
| Claude | 設計方針・アーキテクチャ判断・レビュー・テスト計画・ドキュメント構成 |
| Codex | ソースコード生成・ファイル書き込み・テスト実装・リファクタ実行・ノートブック(ipynb)作成・スクリプト実行・テスト実行 |

Claude Code 側での二次書き込みは不要。Codex が直接ファイルを作成・編集する。
**実装以外の成果物（ipynb、スクリプト等）も Codex に依頼して生成させる。**

---

## 3. 禁止事項・確認事項

### 🚫 Google Places API を勝手に叩かない
- `GOOGLE_PLACES_API_KEY` はコメントアウト済み。
- **ユーザーから明示的な指示がない限り、Google Places API を呼ばない**。
- 既存の `data/raw/*_google_places.csv` の読み込みは可。

### ⚠️ 仕様変更前にユーザー確認を取る
以下に該当する変更は、**実装前にユーザー承認を得る**こと：
- スコアリングロジック（v1/v2/v3/v3b/v4）の式変更・重み変更
- 特徴量の追加・削除・計算式変更
- MLモデルのハイパーパラメータ・学習データ範囲の変更
- データ処理パイプライン（メッシュ変換、人口按分等）のロジック変更
- `config/settings.py` の定数変更

簡易な typo 修正・docstring 追加・テスト追加はこの限りではない。

---

## 4. 実行規約

### テスト
```bash
pytest tests/ -v                    # 現在 153 tests passing
pytest tests/ -v --cov=src          # カバレッジ付き
```

### Python スクリプト実行
すべてプロジェクトルートから `PYTHONPATH=.` を付けて実行：
```bash
PYTHONPATH=. python scripts/xxx.py --tag tokyo
```

### 主要 CLI
| スクリプト | 用途 |
|-----------|------|
| `scripts/collect_area.py` | HotPepper データ収集 |
| `scripts/fetch_external.py` | 駅・地価データ取得 |
| `scripts/integrate_estat.py` | e-Stat 統合 + スコアリング |
| `scripts/train_model.py` | LightGBM 学習 + 市場ギャップ算出 |
| `scripts/generate_report.py` | HTML レポート生成 |
| `scripts/add_area.py` | 新エリア追加 |
| `scripts/run_pipeline.py` | 統合パイプライン |

---

## 5. プロジェクト概要

- **目的**: 飲食チェーン本部向けに「需要高 × 競合少」のエリア×ジャンルを特定する出店意思決定支援システム
- **対象エリア**: tokyo / osaka / nagoya / fukuoka / sapporo（5エリア）
- **対象ジャンル**: izakaya / italian / chinese / yakiniku / cafe / ramen / washoku / curry / other
- **メッシュ設計**: JIS 4分の1メッシュ（250m, 10桁）
- **ML モデル**: LightGBM、5-Fold CV R²=0.797、LOAO R²=0.73〜0.82

詳細は以下を参照：
- `README.md` — セットアップ・使用方法
- `TODO_NEXT_SESSION.md` — 現状サマリ・次タスク
- `docs/system_overview.md` — システム概要
- `docs/ml_model_description.md` — MLモデル詳細
- `docs/prediction_reliability.md` — 予測信頼度（CI跨ぎベース）

---

## 6. ドキュメント方針

- `docs/` 配下は日本語 Markdown。
- 仕様変更時は関連ドキュメントを**同一 PR で更新**する。
- 数値・性能指標を記載する際は算出条件（データ範囲、CV方式、エリア）を併記する。
