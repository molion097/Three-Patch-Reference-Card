from __future__ import annotations

import itertools
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


SYNTHETIC_NAME_RE = re.compile(
    r"rendered_l(?P<light_hue>\d+|white)_oh(?P<object_hue>\d+|white)_os(?P<object_sat>\d+(?:\.\d+)?|white)\.png$"
)


@dataclass(frozen=True)
class SyntheticRender:
    image_path: str
    object_id: str
    light_id: str
    object_hue: str
    object_saturation: str
    light_hue: str


def parse_synthetic_filename(path: str | Path) -> SyntheticRender:
    path = Path(path)
    match = SYNTHETIC_NAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected synthetic filename: {path.name}")
    groups = match.groupdict()
    object_id = f"oh{groups['object_hue']}_os{groups['object_sat']}"
    light_id = f"l{groups['light_hue']}"
    return SyntheticRender(
        image_path=str(path),
        object_id=object_id,
        light_id=light_id,
        object_hue=groups["object_hue"],
        object_saturation=groups["object_sat"],
        light_hue=groups["light_hue"],
    )


def build_synthetic_render_manifest(image_dir: str | Path) -> pd.DataFrame:
    """Build one-row-per-render manifest from Blender PNG filenames."""
    image_dir = Path(image_dir)
    records = [parse_synthetic_filename(path).__dict__ for path in sorted(image_dir.glob("*.png"))]
    if not records:
        raise FileNotFoundError(f"No PNG renders found in {image_dir}")
    frame = pd.DataFrame.from_records(records)
    frame.insert(0, "render_id", frame["object_id"] + "__" + frame["light_id"])
    return frame


def build_synthetic_pair_manifest(render_manifest: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-source-target-light-pair manifest.

    The rendered image set contains one render for each object/light combination.
    Pairing each object's source render with each target render gives the 361*37*37
    relighting samples described in the manuscript.
    """
    required = {"object_id", "light_id", "image_path"}
    missing = required - set(render_manifest.columns)
    if missing:
        raise ValueError(f"Render manifest missing columns: {sorted(missing)}")

    records: list[dict[str, str]] = []
    for object_id, group in render_manifest.groupby("object_id", sort=True):
        by_light = {row.light_id: row.image_path for row in group.itertuples(index=False)}
        for source_light_id, target_light_id in itertools.product(sorted(by_light), repeat=2):
            records.append(
                {
                    "sample_id": f"synthetic__{object_id}__{source_light_id}__to__{target_light_id}",
                    "dataset": "synthetic",
                    "object_id": object_id,
                    "source_light_id": source_light_id,
                    "target_light_id": target_light_id,
                    "source_image_path": by_light[source_light_id],
                    "target_image_path": by_light[target_light_id],
                }
            )
    return pd.DataFrame.from_records(records)


def write_manifest(frame: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def read_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _rgb_columns(prefix: str, values: object) -> dict[str, float]:
    array = list(values)
    if len(array) != 3:
        raise ValueError(f"Expected RGB triplet for {prefix}, got {values}")
    return {f"{prefix}_{channel}": float(value) for channel, value in zip(("r", "g", "b"), array)}


def _patch_columns(prefix: str, values: object) -> dict[str, float]:
    patches = list(values)
    if len(patches) != 3:
        raise ValueError(f"Expected three patches for {prefix}, got {values}")
    names = ("patch_r", "patch_g", "patch_b")
    record: dict[str, float] = {}
    for patch_name, patch_values in zip(names, patches):
        record.update(_rgb_columns(f"{prefix}_{patch_name}", patch_values))
    return record


def build_real_manifest(pickle_path: str | Path, dataset_name: str | None = None) -> pd.DataFrame:
    """Build a real-world manifest from the old list-of-dicts pickle format."""
    pickle_path = Path(pickle_path)
    dataset_name = dataset_name or pickle_path.stem
    with pickle_path.open("rb") as handle:
        rows = pickle.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"Expected list pickle, got {type(rows)} from {pickle_path}")

    records: list[dict[str, object]] = []
    pair_counts: dict[tuple[object, object], int] = {}
    for row in rows:
        source = row["source"]
        target = row["dest"]
        pair_key = (source, target)
        object_index = pair_counts.get(pair_key, 0)
        pair_counts[pair_key] = object_index + 1
        record: dict[str, object] = {
            "sample_id": f"{dataset_name}__obj{object_index:03d}__l{source}__to__l{target}",
            "dataset": "real",
            "dataset_name": dataset_name,
            "object_id": f"obj{object_index:03d}",
            "source_light_id": f"l{source}",
            "target_light_id": f"l{target}",
        }
        record.update(_rgb_columns("source_object_rgb", row["source_object_color"]))
        record.update(_rgb_columns("target_object_rgb", row["dest_object_color"]))
        record.update(_patch_columns("source", row["source_patch_value"]))
        record.update(_patch_columns("target", row["dest_patch_value"]))
        records.append(record)
    return pd.DataFrame.from_records(records)
