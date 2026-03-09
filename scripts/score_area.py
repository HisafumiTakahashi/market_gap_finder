#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config.settings import PROCESSED_DATA_DIR
from src.analyze.scoring import run_scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score aggregated area data.")
    parser.add_argument("--tag", type=str, default="result", help="Input/output file tag.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top areas to score.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        input_path = PROCESSED_DATA_DIR / f"{args.tag}_aggregated.csv"
        output_path = PROCESSED_DATA_DIR / f"{args.tag}_scored.csv"

        df = pd.read_csv(input_path)
        scored_df = run_scoring(df, top_n=args.top_n)
        scored_df.to_csv(output_path, index=False)

        for i, reason in enumerate(scored_df["reason"].head(3), start=1):
            logging.info("Top %d reason: %s", i, reason)

        return 0
    except Exception:
        logging.exception("Failed to score area data")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
