#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys

from src.preprocess.cleaner import run_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess aggregated area data.")
    parser.add_argument("--tag", type=str, default="result", help="Input/output file tag.")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        result = run_preprocess(tag=args.tag)
        logging.info("Preprocessed %d rows", len(result))
        return 0
    except Exception:
        logging.exception("Failed to preprocess area data")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
