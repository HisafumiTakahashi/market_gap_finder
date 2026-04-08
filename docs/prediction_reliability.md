# 予測信頼度（Prediction Reliability）の仕組み

## 1. 概要

予測信頼度は、5-Fold CVで得た各foldモデルの予測ばらつきから、`market_gap` の不確実性を評価する仕組みです。

---

## 2. 使う値

- `market_gap`: OOF予測値 - 実測値
- `fold_predictions`: 各foldモデルで全件に対して出した予測
- `gap_std`: `fold_predictions` の標準偏差

```python
gap_std = np.std(fold_predictions, axis=1)
gap_ci_lower = market_gap - 1.96 * gap_std
gap_ci_upper = market_gap + 1.96 * gap_std
```

---

## 3. 3段階の信頼度ラベル

| 信頼度 | 条件 | 意味 |
|---|---|---|
| high | CI下限 > 0 | 統計的に有意な出店余地 |
| medium | `market_gap > 0` かつ `CI下限 <= 0` | 出店余地の可能性はあるが不確実 |
| low | `market_gap <= 0` | 出店余地なし |

---

## 4. なぜCIベースなのか

- 予測値が正でも、foldごとの差が大きければ不確実
- ばらつき込みで正といえる場所だけを `high` にできる
- 推薦フィルタで `low` を落とし、必要なら `medium` 以上だけを使える

---

## 5. 実運用での使い方

`filter_recommendations` では、最小店舗数・最小gap・最小信頼度で候補を絞り込みます。

例:

```python
filter_recommendations(
    df,
    min_rc=2,
    min_gap_count=1.0,
    min_reliability="medium",
)
```

---

## 6. 分布（2026-04-08時点）

### tokyo（4,520件）

```text
high:   2,068件 (45.8%)
medium:   557件 (12.3%)
low:    1,895件 (41.9%)
```

具体例の値や概念説明は旧版と同じ考え方で読み替えて問題ありません。

---

## 7. 補足

- 信頼度は「精度指標」そのものではなく、予測の不確実性を示す補助指標です
- ML特徴量に人口データを含めた現在も、信頼度の計算方法自体は変わっていません
