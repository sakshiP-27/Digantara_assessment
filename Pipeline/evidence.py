"""Save before/after detection masks: one threshold per tile versus a local background."""
import json
import os

import cv2
import numpy as np

from Pipeline import config
from Pipeline.detect import build_binary_mask, local_background, tile_threshold
from Pipeline.preprocess import stretch_to_8bit


# One crowded CAM tile and one quieter UUID tile.
SAMPLES = [
    (
        "CAM_B_20260815T154244_manual_f000276",
        "CAM_B_20260815T154244_manual_f000276_r3_c6.png",
    ),
    (
        "1a600998-b97c-4307-8632-6fcf573621ff",
        "1a600998-b97c-4307-8632-6fcf573621ff_r0_c1.png",
    ),
]


def paint(tile, mask):
    """Stretch a tile for viewing and paint kept pixels green."""
    gray = stretch_to_8bit(tile, config.STRETCH_LOW_PCT, config.STRETCH_HIGH_PCT)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image[mask] = (0, 220, 0)
    return image


def compare_tile(image_id, tile_name, out_dir):
    """Write the old and new masks for one tile, plus the pixel counts."""
    tile = cv2.imread(
        os.path.join(str(config.TILES_DIR), image_id, tile_name),
        cv2.IMREAD_UNCHANGED,
    ).astype(np.float32)
    old_cut, _, _ = tile_threshold(tile)
    old_mask = build_binary_mask(tile, old_cut)
    med_map, sigma_map = local_background(tile)
    new_mask = build_binary_mask(tile, med_map + config.DETECT_SIGMA_K * sigma_map)
    stem = tile_name.replace(".png", "")
    cv2.imwrite(os.path.join(out_dir, f"{stem}_before.png"), paint(tile, old_mask))
    cv2.imwrite(os.path.join(out_dir, f"{stem}_after.png"), paint(tile, new_mask))
    return {
        "tile": tile_name,
        "before_pixels": int(old_mask.sum()),
        "after_pixels": int(new_mask.sum()),
    }


def main():
    """CLI entry point: write before/after images for the sample tiles."""
    out_dir = str(config.EVIDENCE_DIR)
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for image_id, tile_name in SAMPLES:
        row = compare_tile(image_id, tile_name, out_dir)
        rows.append(row)
        print(f"{row['tile']}: before={row['before_pixels']} after={row['after_pixels']}")
    with open(os.path.join(out_dir, "before_after.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"Wrote before/after images to {out_dir}")


if __name__ == "__main__":
    main()
