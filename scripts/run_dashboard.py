#!/usr/bin/env python3
"""市場ギャップ可視化ダッシュボードを起動する CLI スクリプト。"""

from __future__ import annotations

import argparse
import logging
import sys

from src.visualize.dashboard import build_dash_app


def parse_args() -> argparse.Namespace:
    """ダッシュボード起動時のホスト・ポート・タグ引数を解析する。

    Returns:
        起動設定を保持した名前空間。
    """
    parser = argparse.ArgumentParser(description="Run the area dashboard.")
    parser.add_argument("--tag", type=str, default="result", help="Data file tag to load.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server to.")
    parser.add_argument("--port", type=int, default=8050, help="Port to bind the server to.")
    return parser.parse_args()


def main() -> int:
    """ダッシュボードを起動し、終了コードを返す。

    Returns:
        正常終了時は `0`、例外発生時は `1`。
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    try:
        app = build_dash_app(args.tag)
        app.run(host=args.host, port=args.port, debug=False)
        return 0
    except Exception:
        logging.exception("Failed to run dashboard")
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
