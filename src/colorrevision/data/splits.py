from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _assign_groups(groups: list[str], seed: int, train: float, val: float) -> dict[str, str]:
    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n_total = len(shuffled)
    n_train = int(round(n_total * train))
    n_val = int(round(n_total * val))
    train_groups = set(shuffled[:n_train])
    val_groups = set(shuffled[n_train : n_train + n_val])
    return {
        group: "train" if group in train_groups else "val" if group in val_groups else "test"
        for group in shuffled
    }


def split_manifest(
    manifest: pd.DataFrame,
    split_name: str,
    seed: int = 42,
    train: float = 0.7,
    val: float = 0.15,
) -> pd.DataFrame:
    """Create deterministic train/val/test split assignments."""
    if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
        raise ValueError("Expected 0 < train < 1, 0 <= val < 1, and train + val < 1")
    if "sample_id" not in manifest:
        raise ValueError("Manifest requires sample_id")

    frame = manifest.copy()
    if split_name == "random_pair":
        frame["_group_key"] = frame["sample_id"].astype(str)
    elif split_name == "object_holdout":
        frame["_group_key"] = frame["object_id"].astype(str)
    elif split_name == "light_holdout":
        frame["_group_key"] = frame["source_light_id"].astype(str) + "__to__" + frame["target_light_id"].astype(str)
    elif split_name == "target_light_holdout":
        frame["_group_key"] = frame["target_light_id"].astype(str)
    elif split_name == "object_and_light_holdout":
        object_assignments = _assign_groups(sorted(frame["object_id"].astype(str).unique()), seed, train, val)
        light_pair_key = frame["source_light_id"].astype(str) + "__to__" + frame["target_light_id"].astype(str)
        light_assignments = _assign_groups(sorted(light_pair_key.unique()), seed + 1, train, val)
        object_split = frame["object_id"].astype(str).map(object_assignments)
        light_split = light_pair_key.map(light_assignments)
        out = frame[["sample_id"]].copy()
        out["split"] = [
            obj if obj == light else "unused"
            for obj, light in zip(object_split, light_split, strict=True)
        ]
        return out
    else:
        raise ValueError(f"Unknown split_name: {split_name}")

    assignments = _assign_groups(sorted(frame["_group_key"].unique()), seed, train, val)
    out = frame[["sample_id", "_group_key"]].copy()
    out["split"] = out["_group_key"].map(assignments)
    return out.drop(columns=["_group_key"])


def write_split(split_frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(output_path, index=False)
