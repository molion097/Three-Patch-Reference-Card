import pandas as pd

from colorrevision.data.splits import split_manifest


def _manifest():
    return pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "object_id": [f"o{i // 3}" for i in range(12)],
            "source_light_id": [f"l{i % 3}" for i in range(12)],
            "target_light_id": [f"l{(i + 1) % 3}" for i in range(12)],
        }
    )


def _grid_manifest(object_count=20, light_count=6):
    records = []
    for object_idx in range(object_count):
        for source_idx in range(light_count):
            for target_idx in range(light_count):
                records.append(
                    {
                        "sample_id": f"o{object_idx}_l{source_idx}_l{target_idx}",
                        "object_id": f"o{object_idx}",
                        "source_light_id": f"l{source_idx}",
                        "target_light_id": f"l{target_idx}",
                    }
                )
    return pd.DataFrame.from_records(records)


def test_split_is_deterministic():
    manifest = _manifest()
    a = split_manifest(manifest, "object_holdout", seed=7)
    b = split_manifest(manifest, "object_holdout", seed=7)
    assert a.equals(b)


def test_object_holdout_has_no_object_leakage():
    manifest = _grid_manifest()  # needs enough objects to produce all three splits
    split = split_manifest(manifest, "object_holdout", seed=7)
    merged = manifest.merge(split, on="sample_id")
    assert {"train", "val", "test"}.issubset(set(split["split"])), "all three splits must be non-empty"
    groups = merged.groupby("object_id")["split"].nunique()
    assert groups.max() == 1


def test_object_and_light_holdout_has_no_object_or_light_pair_leakage():
    manifest = _grid_manifest()
    split = split_manifest(manifest, "object_and_light_holdout", seed=7)
    merged = manifest.merge(split, on="sample_id")
    used = merged[merged["split"].isin(["train", "val", "test"])].copy()
    used["light_pair"] = used["source_light_id"] + "__to__" + used["target_light_id"]

    assert {"train", "val", "test"}.issubset(set(used["split"]))
    assert used.groupby("object_id")["split"].nunique().max() == 1
    assert used.groupby("light_pair")["split"].nunique().max() == 1
    assert "unused" in set(merged["split"])


def test_light_holdout_has_no_light_pair_leakage():
    manifest = _grid_manifest()
    split = split_manifest(manifest, "light_holdout", seed=7)
    merged = manifest.merge(split, on="sample_id")
    merged["light_pair"] = merged["source_light_id"] + "__to__" + merged["target_light_id"]
    groups = merged.groupby("light_pair")["split"].nunique()
    assert groups.max() == 1


def test_light_holdout_is_deterministic():
    manifest = _grid_manifest()
    a = split_manifest(manifest, "light_holdout", seed=7)
    b = split_manifest(manifest, "light_holdout", seed=7)
    assert a.equals(b)
