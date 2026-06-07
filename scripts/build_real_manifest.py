#!/usr/bin/env python
from __future__ import annotations

import argparse

from colorrevision.data.manifest import build_real_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-world manifest from old pickle data.")
    parser.add_argument(
        "--pickle",
        default="data/mulmed_final_project/machine_learning/dataset/colordata2/real_world_dataset_2.pkl",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--output", default="newcode/manifests/real_world_dataset_2.csv")
    args = parser.parse_args()

    manifest = build_real_manifest(args.pickle, dataset_name=args.dataset_name)
    write_manifest(manifest, args.output)
    print(f"Wrote {len(manifest)} real-world samples to {args.output}")
    print("Lights:", sorted(manifest["source_light_id"].unique()))
    print("Objects:", manifest["object_id"].nunique())


if __name__ == "__main__":
    main()
