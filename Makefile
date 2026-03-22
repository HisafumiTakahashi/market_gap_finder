# Market Gap Finder — パイプライン自動化
#
# 使い方:
#   make pipeline TAG=tokyo          # 単一エリア全ステップ実行
#   make pipeline-combined           # 4エリア統合学習
#   make integrate TAG=osaka         # 個別ステップ実行
#
# 前提:
#   - collect_area.py によるホットペッパー収集は事前に手動実行済みであること
#   - .env に ESTAT_API_KEY 等が設定済みであること

PYTHON   := python3
PYPATH   := PYTHONPATH=.
TAG      ?= tokyo
TOP_N    ?= 20
FOLDS    ?= 5
ALL_TAGS := tokyo osaka nagoya fukuoka

.PHONY: help fetch integrate train report validate pipeline pipeline-combined test lint clean

help:
	@echo "=== Market Gap Finder Pipeline ==="
	@echo ""
	@echo "  make fetch      TAG=tokyo    外部データ取得（駅・地価・乗降客数）"
	@echo "  make integrate  TAG=tokyo    e-Stat人口統合 + 特徴量 + スコアリング"
	@echo "  make train      TAG=tokyo    MLモデル学習"
	@echo "  make report     TAG=tokyo    HTMLレポート生成"
	@echo "  make validate   TAG=tokyo    スコア検証"
	@echo "  make pipeline   TAG=tokyo    上記を順番に実行"
	@echo ""
	@echo "  make pipeline-combined       4エリア統合学習パイプライン"
	@echo "  make test                    テスト実行"
	@echo "  make lint                    Ruff lint"

# ── 個別ステップ ──────────────────────────────

fetch:
	$(PYPATH) $(PYTHON) scripts/fetch_external.py --tag $(TAG)

integrate:
	$(PYPATH) $(PYTHON) scripts/integrate_estat.py --tag $(TAG) --skip-fetch

train:
	$(PYPATH) $(PYTHON) scripts/train_model.py --tag $(TAG) --top-n $(TOP_N) --folds $(FOLDS) --two-stage

report:
	$(PYPATH) $(PYTHON) scripts/generate_report.py --tag $(TAG) --top-n $(TOP_N)

validate:
	$(PYPATH) $(PYTHON) scripts/validate_scores.py --tag $(TAG) --top-n $(TOP_N)

# ── パイプライン ──────────────────────────────

pipeline: fetch integrate train report
	@echo "Pipeline complete for TAG=$(TAG)"

pipeline-combined:
	@for tag in $(ALL_TAGS); do \
		if [ -f "data/raw/$${tag}_hotpepper.csv" ]; then \
			echo "=== Processing $$tag ==="; \
			$(MAKE) fetch integrate TAG=$$tag; \
		else \
			echo "=== Skipping $$tag (no hotpepper data) ==="; \
		fi; \
	done
	$(PYPATH) $(PYTHON) scripts/train_model.py --combined --loao --top-n $(TOP_N) --folds $(FOLDS) --two-stage
	@for tag in $(ALL_TAGS); do \
		if [ -f "data/processed/$${tag}_integrated.csv" ]; then \
			$(MAKE) report TAG=$$tag; \
		fi; \
	done
	@echo "Combined pipeline complete"

# ── ユーティリティ ────────────────────────────

test:
	$(PYPATH) $(PYTHON) -m pytest tests/ -v

lint:
	$(PYPATH) $(PYTHON) -m ruff check src/ scripts/ config/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
