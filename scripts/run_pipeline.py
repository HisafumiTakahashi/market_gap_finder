#!/usr/bin/env python3
"""Market Gap Finder 統合パイプライン CLI。

使い方:
    python scripts/run_pipeline.py --tag tokyo                    # 単一エリア全ステップ
    python scripts/run_pipeline.py --tag tokyo --steps fetch integrate  # 個別ステップ
    python scripts/run_pipeline.py --combined                     # 4エリア統合学習

前提:
    - collect_area.py によるホットペッパー収集は事前に手動実行済みであること
    - .env に ESTAT_API_KEY 等が設定済みであること
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALL_TAGS = ["tokyo", "osaka", "nagoya", "fukuoka"]

STEPS = {
    "fetch": "外部データ取得（駅・地価・乗降客数）",
    "integrate": "e-Stat人口統合 + 特徴量 + スコアリング",
    "train": "MLモデル学習",
    "report": "HTMLレポート生成",
    "validate": "スコア検証",
}

DEFAULT_STEPS = ["fetch", "integrate", "train", "report"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Market Gap Finder 統合パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="ステップ一覧:\n" + "\n".join(f"  {k:12s} {v}" for k, v in STEPS.items()),
    )
    parser.add_argument("--tag", type=str, default="tokyo", help="対象エリアタグ (default: tokyo)")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS.keys()), default=None,
                        help="実行するステップ (default: fetch integrate train report)")
    parser.add_argument("--combined", action="store_true", help="4エリア統合学習パイプライン")
    parser.add_argument("--top-n", type=int, default=20, help="表示する上位件数")
    parser.add_argument("--folds", type=int, default=5, help="CV分割数")
    parser.add_argument("--dry-run", action="store_true", help="実行せずコマンドを表示")
    return parser.parse_args()


def run_step(cmd: list[str], step_name: str, dry_run: bool = False) -> None:
    """Run a pipeline step as a subprocess."""
    logger.info("=== %s ===", step_name)
    logger.info("CMD: %s", " ".join(cmd))

    if dry_run:
        return

    env = {**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT)}
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Step '{step_name}' failed with exit code {result.returncode}")


def build_step_cmd(step: str, tag: str, top_n: int, folds: int) -> list[str]:
    """Build the command list for a given step."""
    python = sys.executable
    if step == "fetch":
        return [python, "scripts/fetch_external.py", "--tag", tag]
    if step == "integrate":
        return [python, "scripts/integrate_estat.py", "--tag", tag, "--skip-fetch"]
    if step == "train":
        return [python, "scripts/train_model.py", "--tag", tag,
                "--top-n", str(top_n), "--folds", str(folds), "--two-stage"]
    if step == "report":
        return [python, "scripts/generate_report.py", "--tag", tag, "--top-n", str(top_n)]
    if step == "validate":
        return [python, "scripts/validate_scores.py", "--tag", tag, "--top-n", str(top_n)]
    raise ValueError(f"Unknown step: {step}")


def run_single(tag: str, steps: list[str], top_n: int, folds: int, dry_run: bool) -> None:
    """Run pipeline for a single area tag."""
    logger.info("Pipeline start: tag=%s, steps=%s", tag, steps)
    for step in steps:
        cmd = build_step_cmd(step, tag, top_n, folds)
        run_step(cmd, f"{step} ({tag})", dry_run=dry_run)
    logger.info("Pipeline complete: tag=%s", tag)


def run_combined(top_n: int, folds: int, dry_run: bool) -> None:
    """Run combined multi-area pipeline."""
    python = sys.executable
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Step 1: fetch + integrate for each area with data
    for tag in ALL_TAGS:
        if not (raw_dir / f"{tag}_hotpepper.csv").exists():
            logger.info("Skipping %s (no hotpepper data)", tag)
            continue
        run_single(tag, ["fetch", "integrate"], top_n, folds, dry_run)

    # Step 2: combined train
    cmd = [python, "scripts/train_model.py", "--combined", "--loao",
           "--top-n", str(top_n), "--folds", str(folds), "--two-stage"]
    run_step(cmd, "train (combined)", dry_run=dry_run)

    # Step 3: report for each area
    for tag in ALL_TAGS:
        if (processed_dir / f"{tag}_integrated.csv").exists():
            cmd = build_step_cmd("report", tag, top_n, folds)
            run_step(cmd, f"report ({tag})", dry_run=dry_run)

    logger.info("Combined pipeline complete")


def main() -> int:
    """Run the pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        if args.combined:
            run_combined(args.top_n, args.folds, args.dry_run)
        else:
            steps = args.steps or DEFAULT_STEPS
            run_single(args.tag, steps, args.top_n, args.folds, args.dry_run)
        return 0
    except Exception:
        logger.exception("Pipeline failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
