#!/usr/bin/env python
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions and write summaries.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    raise SystemExit(f"Evaluation scaffold ready for predictions: {args.predictions}")


if __name__ == "__main__":
    main()
