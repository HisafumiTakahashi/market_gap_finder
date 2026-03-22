"""Shared constants for analysis modules."""

# 人口1万人あたりの飲食店密度（マーケットリサーチの標準単位）
_POP_UNIT = 10000.0

# ゼロ除算回避のスムージング定数
# Optuna 30trial で最適化 (R2: 0.8028→0.8040)
# 中央値の正規化人口 (0.096) に対して ~66% — 人口差を適度に反映しつつ安定性を確保
_COMPETITOR_OFFSET = 0.064
