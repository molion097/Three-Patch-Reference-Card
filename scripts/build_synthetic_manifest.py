"""
Extract patch RGB values from synthetic Blender renders and produce a
manifest CSV in the same column format as real_world_dataset_2.csv.

Strategy
--------
1. Detect each reference patch (blue, red, yellow) and the object patch
   by color thresholding on a NEAR-NEUTRAL calibration image
   (os=0.1 → very low saturation → object appears near-white).
   All renders share the same Blender camera, so detected pixel
   coordinates are fixed for every image.
2. For every synthetic pair (source_image, target_image), sample those
   fixed pixel windows and record the average RGB.
3. Write newcode/manifests/synthetic_with_patches.csv.
"""

import cv2
import numpy as np
import pandas as pd
import pathlib
import sys

# ── paths ─────────────────────────────────────────────────────────────────────
REPO    = pathlib.Path(r"D:\colorrevision")
IMG_DIR = REPO / "data" / "object_light_combination_hue_saturation"
PAIRS   = REPO / "newcode" / "manifests" / "synthetic_pairs.csv"
OUT     = REPO / "newcode" / "manifests" / "synthetic_with_patches.csv"
OFFSET  = 12   # pixel half-window for patch averaging


# ── step 1: auto-detect patch coordinates on one reference image ──────────────
def detect_patch_centers(img_bgr):
    """
    Find centroids of the 4 key patches.

    Reference patches (red, blue, yellow) are detected by HSV color thresholding.
    Object (sphere) cannot be detected by color because it varies across renders.
    Instead, after locating the card patches, the object is found as the largest
    bright foreground blob that lies OUTSIDE a generous card exclusion zone.
    Returns dict: {name: (cx, cy)}
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = img_hsv[:,:,0], img_hsv[:,:,1], img_hsv[:,:,2]
    H, W = img_bgr.shape[:2]

    def largest_blob_center(mask):
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        if M["m00"] == 0:
            return None
        return (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))

    bright = v > 80

    red_mask  = (((h < 15) | (h > 165)) & (s > 100) & bright).astype(np.uint8) * 255
    blue_mask = ((h > 100) & (h < 130) & (s > 100) & bright).astype(np.uint8) * 255
    yel_mask  = ((h > 20)  & (h < 40)  & (s > 80)  & bright).astype(np.uint8) * 255

    patch_r_center = largest_blob_center(red_mask)
    patch_g_center = largest_blob_center(blue_mask)
    patch_b_center = largest_blob_center(yel_mask)

    # Detect object by exclusion: foreground blob outside the card zone.
    # Card zone = bounding box of the 3 reference patch centers + large margin.
    obj_center = None
    known_centers = [c for c in [patch_r_center, patch_g_center, patch_b_center]
                     if c is not None]
    if known_centers:
        margin = 150  # generous pixel margin around the card
        xs = [c[0] for c in known_centers]
        ys = [c[1] for c in known_centers]
        x0 = max(0,     min(xs) - margin)
        x1 = min(W - 1, max(xs) + margin)
        y0 = max(0,     min(ys) - margin)
        y1 = min(H - 1, max(ys) + margin)

        # Foreground = anything clearly brighter than typical background (v>60)
        fg_mask = (v > 60).astype(np.uint8) * 255

        # Zero out the card zone
        fg_mask[y0:y1, x0:x1] = 0

        obj_center = largest_blob_center(fg_mask)

    return {
        "patch_r": patch_r_center,
        "patch_g": patch_g_center,
        "patch_b": patch_b_center,
        "object":  obj_center,
    }


def avg_patch_rgb(img_bgr, cx, cy, offset=OFFSET):
    """Mean RGB (float 0-255) of a square window centred at (cx, cy)."""
    roi = img_bgr[cy-offset:cy+offset, cx-offset:cx+offset]
    mean_bgr = roi.mean(axis=(0, 1))
    return mean_bgr[[2, 1, 0]]   # BGR → RGB


# ── calibrate: pick a near-neutral image (os0.1) so object is detectable ─────
# Prefer l0 (first light) with oh0 os0.1; fall back to any os0.1 image.
def find_calibration_image():
    preferred = IMG_DIR / "rendered_l0_oh0_os0.1.png"
    if preferred.exists():
        return preferred
    candidates = sorted(IMG_DIR.glob("*_os0.1.png"))
    if candidates:
        return candidates[0]
    # Last resort: any image
    return sorted(IMG_DIR.glob("*.png"))[0]

ref_img_path = find_calibration_image()
print(f"Calibrating from: {ref_img_path.name}")
ref_img = cv2.imread(str(ref_img_path))
if ref_img is None:
    sys.exit(f"Could not load {ref_img_path}")

print(f"Image size: {ref_img.shape[1]}x{ref_img.shape[0]}")
centers = detect_patch_centers(ref_img)
print("Detected centers:", centers)

missing = [k for k, v in centers.items() if v is None]
if missing:
    sys.exit(f"Could not detect patches: {missing}")

cx_r, cy_r = centers["patch_r"]
cx_g, cy_g = centers["patch_g"]
cx_b, cy_b = centers["patch_b"]
cx_o, cy_o = centers["object"]

for name, (cx, cy) in centers.items():
    rgb = avg_patch_rgb(ref_img, cx, cy).astype(int)
    print(f"  {name}: ({cx},{cy})  RGB={rgb}")

# ── step 2: cache all unique images first, then build manifest ────────────────
pairs = pd.read_csv(PAIRS)
print(f"\nFound {len(pairs):,} synthetic pairs.")

all_paths = pd.unique(
    pd.concat([pairs["source_image_path"], pairs["target_image_path"]])
)
print(f"Loading {len(all_paths):,} unique images into cache...")

img_cache = {}
errors_load = 0
for i, rel_path in enumerate(all_paths):
    full = REPO / rel_path
    img = cv2.imread(str(full))
    if img is None:
        errors_load += 1
    else:
        img_cache[rel_path] = img
    if (i + 1) % 2000 == 0:
        print(f"  cached {i+1:,} / {len(all_paths):,}")

print(f"Cache ready: {len(img_cache):,} images ({errors_load} failed)")

def ex(img, cx, cy):
    # Store as 0-255 integers (same format as real_world_dataset_2.csv)
    # so that patches_from_row / rgb_from_row from row_access.py work unchanged.
    return avg_patch_rgb(img, cx, cy).round().astype(int)

print("Building manifest rows...")
rows, errors_pair = [], 0
for i, row in pairs.iterrows():
    src_img = img_cache.get(row["source_image_path"])
    tgt_img = img_cache.get(row["target_image_path"])

    if src_img is None or tgt_img is None:
        errors_pair += 1
        continue

    sp_r = ex(src_img, cx_r, cy_r); sp_g = ex(src_img, cx_g, cy_g)
    sp_b = ex(src_img, cx_b, cy_b); so   = ex(src_img, cx_o, cy_o)
    tp_r = ex(tgt_img, cx_r, cy_r); tp_g = ex(tgt_img, cx_g, cy_g)
    tp_b = ex(tgt_img, cx_b, cy_b); to_  = ex(tgt_img, cx_o, cy_o)

    rows.append({
        "sample_id": row["sample_id"], "dataset": "synthetic",
        "object_id": row["object_id"],
        "source_light_id": row["source_light_id"],
        "target_light_id": row["target_light_id"],
        "source_object_rgb_r": so[0],  "source_object_rgb_g": so[1],  "source_object_rgb_b": so[2],
        "target_object_rgb_r": to_[0], "target_object_rgb_g": to_[1], "target_object_rgb_b": to_[2],
        "source_patch_r_r": sp_r[0], "source_patch_r_g": sp_r[1], "source_patch_r_b": sp_r[2],
        "source_patch_g_r": sp_g[0], "source_patch_g_g": sp_g[1], "source_patch_g_b": sp_g[2],
        "source_patch_b_r": sp_b[0], "source_patch_b_g": sp_b[1], "source_patch_b_b": sp_b[2],
        "target_patch_r_r": tp_r[0], "target_patch_r_g": tp_r[1], "target_patch_r_b": tp_r[2],
        "target_patch_g_r": tp_g[0], "target_patch_g_g": tp_g[1], "target_patch_g_b": tp_g[2],
        "target_patch_b_r": tp_b[0], "target_patch_b_g": tp_b[1], "target_patch_b_b": tp_b[2],
    })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False)
print(f"\nSaved {len(out_df):,} rows → {OUT}")
if errors_pair:
    print(f"WARNING: {errors_pair} pairs skipped")
