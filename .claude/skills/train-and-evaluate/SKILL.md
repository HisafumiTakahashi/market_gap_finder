---
name: train-and-evaluate
description: LightGBM市場容量予測モデルの再学習と精度評価を行う。東京エリアで店舗数層化5-Fold CV、予測信頼度分布（CI跨ぎベース）、特徴量重要度、Spearman相関（vs v3bスコア）を出力。モデル精度改善の施策検証、特徴量追加・削除の影響確認、Optunaチューニング後の評価時に使う。
---

# train-and-evaluate — MLモデル再学習・評価

## 目的
LightGBM 市場容量予測モデルを再学習し、複数指標で性能を評価する。

## 前提
- `data/processed/{tag}_estat_scored.csv` が各エリアで生成済み
- 既存モデル（`models/combined_lightgbm.txt` 等）の性能を把握している

## 現状ベースライン（2026-03-24）

| 指標 | 値 |
|------|-----|
| 5-Fold CV R² | 0.797 |
| 5-Fold CV RMSE | 0.309 |
| LOAO tokyo R² | 0.732（最低） |
| LOAO fukuoka R² | 0.816（最高） |
| ML vs v3b Spearman | 0.598 |
| 特徴量数 | 13（人口特徴量除外後） |
| 学習データ | 9,665行（4エリア結合、sapporo除外） |

新モデルはこれと比較する。

## 実行手順

### 1. 学習 + 評価
```bash
PYTHONPATH=. python scripts/train_model.py --tag tokyo --top-n 20
```
> Codex サンドボックスでは `python` の代わりにフルパスを使用：
> `/c/Users/hisaf/AppData/Local/Programs/Python/Python312/python.exe`

### 2. 出力される指標
- **5-Fold CV R² / RMSE**: 汎化性能（店舗数層化 StratifiedKFold）
- **予測信頼度分布**: high / medium / low の割合
- **特徴量重要度 Top 10**
- **Spearman 相関 vs v3b スコア**

### 3. 推薦リスト出力
- CI 付き上位 20 メッシュ
- 信頼度 high のみフィルタした版

## 施策検証のフロー

### A. 特徴量追加・削除を試す場合
1. **ユーザー承認を取る**（CLAUDE.md 3章「仕様変更前確認」）
2. `src/analyze/features.py` または `src/analyze/ml_model.py` 修正
3. `train_model.py` 実行
4. ベースラインとの差分を表で提示：

| 変更 | CV R² | LOAO最低 | Spearman |
|------|-------|----------|----------|
| ベースライン | 0.797 | 0.732 | 0.598 |
| 新版 | ? | ? | ? |

5. 改善が見られなければ**不採用を明記**（TODO_NEXT_SESSION.md の検証済み表に追記）

### B. Optuna ハイパーパラメータチューニング
```bash
# v3b 重みチューニング
PYTHONPATH=. python scripts/tune_v3b_weights.py --tag combined
```
LightGBM ハイパラチューニングは `ml_model.py` 内の Optuna 関数を使う。

## 不採用施策リスト（再試行禁止）

以下は検証済み不採用。**同じことを再度試さない**：

| 施策 | 結果 |
|------|------|
| サンプル重み付け（weight=log1p(rc)） | R² 0.797→0.781 悪化 |
| rc=0 行追加 | 全セグメントで RMSE 悪化 |
| Google 特徴量除外 | R² 0.797→0.760 悪化 |
| 2段階残差モデル（Ridge→LightGBM） | CV R² 0.795→0.739 悪化 |
| 2段階モデル（分類→回帰） | F1=0.830, R²=0.758 単一モデルに及ばず |
| 外れ値フィルタ | 全閾値でフィルタなしが最良 |
| gap_std絶対閾値による信頼度 | 識別力不足（high=71%） |

## 予測信頼度（CI跨ぎベース）

- `train_cv` で fold 別に全データ予測 → 95% CI 算出
- **CI 下限 > 0** → high（統計的に有意な出店余地）
- **gap > 0 だが CI が 0 跨ぐ** → medium
- **gap ≤ 0** → low

詳細: `docs/prediction_reliability.md`

## モデル再学習時の注意
- `models/combined_lightgbm.txt` 上書き前に、ユーザー承認を取る
- 変更理由・性能差分を `docs/ml_model_description.md` に追記する
