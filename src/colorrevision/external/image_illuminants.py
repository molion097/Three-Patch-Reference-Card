from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage


FULL_IMAGE_RE = re.compile(r"(?P<light_id>\d+)_(?P<same_id>\d+)_(?P<light_name>.+?)_.+?_full\.jpg$")


def read_rgb_image(path: str | Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float64) / 255.0


def gray_world_illuminant(image: np.ndarray) -> np.ndarray:
    illum = np.mean(image.reshape(-1, 3), axis=0)
    return illum / np.linalg.norm(illum)


def white_patch_illuminant(image: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    luminance = np.mean(image, axis=2)
    threshold = np.percentile(luminance, percentile)
    pixels = image[luminance >= threshold]
    illum = np.mean(pixels.reshape(-1, 3), axis=0)
    return illum / np.linalg.norm(illum)


def shades_of_gray_illuminant(image: np.ndarray, p: float = 6.0) -> np.ndarray:
    illum = np.mean(np.power(np.clip(image, 0.0, 1.0), p).reshape(-1, 3), axis=0) ** (1.0 / p)
    return illum / np.linalg.norm(illum)


def gray_edge_illuminant(image: np.ndarray, sigma: float = 1.0, p: float = 6.0) -> np.ndarray:
    smoothed = ndimage.gaussian_filter(image, sigma=(sigma, sigma, 0))
    gx = ndimage.sobel(smoothed, axis=1)
    gy = ndimage.sobel(smoothed, axis=0)
    edge_mag = np.sqrt(gx * gx + gy * gy)
    illum = np.mean(np.power(np.clip(edge_mag, 0.0, None), p).reshape(-1, 3), axis=0) ** (1.0 / p)
    if np.linalg.norm(illum) == 0:
        return gray_world_illuminant(image)
    return illum / np.linalg.norm(illum)


def estimate_illuminant(image: np.ndarray, method: str) -> np.ndarray:
    if method == "gray_world":
        return gray_world_illuminant(image)
    if method == "white_patch":
        return white_patch_illuminant(image)
    if method == "shades_of_gray":
        return shades_of_gray_illuminant(image)
    if method == "gray_edge":
        return gray_edge_illuminant(image)
    raise ValueError(f"Unknown image illuminant method: {method}")


def representative_light_images(image_dir: str | Path) -> pd.DataFrame:
    """Find one representative identity-pair full image per light ID."""
    records = []
    for path in sorted(Path(image_dir).glob("*_full.jpg")):
        match = FULL_IMAGE_RE.match(path.name)
        if not match:
            continue
        groups = match.groupdict()
        if groups["light_id"] != groups["same_id"]:
            continue
        records.append(
            {
                "light_id": f"l{groups['light_id']}",
                "light_name": groups["light_name"],
                "image_path": str(path),
            }
        )
    frame = pd.DataFrame.from_records(records).drop_duplicates("light_id")
    if frame.empty:
        raise FileNotFoundError(f"No identity-pair *_full.jpg files found in {image_dir}")
    return frame.sort_values("light_id")


def build_illuminant_table(light_images: pd.DataFrame, manifest: pd.DataFrame, method: str) -> pd.DataFrame:
    estimates = {}
    for row in light_images.itertuples(index=False):
        estimates[row.light_id] = estimate_illuminant(read_rgb_image(row.image_path), method)
    records = []
    for row in manifest.itertuples(index=False):
        source = estimates[row.source_light_id]
        target = estimates[row.target_light_id]
        records.append(
            {
                "sample_id": row.sample_id,
                "source_illuminant_r": source[0],
                "source_illuminant_g": source[1],
                "source_illuminant_b": source[2],
                "target_illuminant_r": target[0],
                "target_illuminant_g": target[1],
                "target_illuminant_b": target[2],
            }
        )
    return pd.DataFrame.from_records(records)
