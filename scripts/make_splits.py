#!/usr/bin/env python
from __future__ import annotations

import argparse

from colorrevision.data.manifest import read_manifest
from colorrevision.data.splits import split_manifest, write_split


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic train/val/test splits.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-name", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    split = split_manifest(manifest, args.split_name, seed=args.seed, train=args.train, val=args.val)
    write_split(split, args.output)
    counts = split["split"].value_counts().to_dict()
    print(f"Wrote split to {args.output}: {counts}")


if __name__ == "__main__":
    main()
