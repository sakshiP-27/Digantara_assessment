"""Stage 5: stitch tiles back to the full frame and overlay the annotations."""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from Pipeline import config
from Pipeline.preprocess import stretch_to_8bit


# BGR colors so blobs and streaks stay distinct on the grayscale frame.
BLOB_COLOR = (0, 220, 0)
STREAK_COLOR = (0, 0, 255)


def stitch_tiles(manifest, tiles_dir, detect_dir):
    """Place every tile and its class mask back onto the padded canvas."""
    geometry = manifest["geometry"]
    # The canvas matches the padded size used in Stage 2, so every tile lands exactly.
    canvas = np.zeros((geometry["padded_height"], geometry["padded_width"]), dtype=np.uint16)
    # Class mask uses the same values as Stage 3: 0 background, 1 blob, 2 streak.
    mask = np.zeros((geometry["padded_height"], geometry["padded_width"]), dtype=np.uint8)
    stem = manifest["image_id"]
    for tile in manifest["tiles"]:
        # y0/x0 are the top-left of this tile inside the padded frame.
        y0, x0 = tile["y0"], tile["x0"]
        tile_size = geometry["tile"]
        block = cv2.imread(os.path.join(tiles_dir, stem, tile["name"]), cv2.IMREAD_UNCHANGED)
        canvas[y0:y0 + tile_size, x0:x0 + tile_size] = block
        mask_name = tile["name"].replace(".png", "_mask.png")
        class_mask = cv2.imread(os.path.join(detect_dir, stem, mask_name), cv2.IMREAD_UNCHANGED)
        mask[y0:y0 + tile_size, x0:x0 + tile_size] = class_mask
    # Drop the bottom/right padding so the frame is the original 9568 x 6380 again.
    height, width = geometry["orig_height"], geometry["orig_width"]
    return canvas[:height, :width], mask[:height, :width]


def overlay_annotations(gray8, class_mask):
    """Paint blob and streak pixels onto a viewable 8-bit frame."""
    # Start from the stretched grayscale image so stars are visible under the color.
    image = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(image)
    color[class_mask == 1] = BLOB_COLOR
    color[class_mask == 2] = STREAK_COLOR
    marked = class_mask > 0
    # Blend so the mask color is obvious without hiding the underlying pixels.
    blended = image.astype(np.float32)
    blended[marked] = 0.35 * image[marked] + 0.65 * color[marked]
    return blended.astype(np.uint8)


def repatch_image(tiles_dir, detect_dir, out_dir):
    """Rebuild one full frame and write the plain and annotated inspection images."""
    stem = os.path.basename(tiles_dir.rstrip("\\/"))
    manifest = json.load(open(os.path.join(tiles_dir, "tiles_manifest.json"), encoding="utf-8"))
    frame16, class_mask = stitch_tiles(manifest, os.path.dirname(tiles_dir), detect_dir)
    # The same percentile stretch used in Stage 1 makes the dark sky viewable.
    gray8 = stretch_to_8bit(frame16.astype(np.float32), config.STRETCH_LOW_PCT, config.STRETCH_HIGH_PCT)
    overlay = overlay_annotations(gray8, class_mask)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_repatched.png"), gray8)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"), overlay)
    n_blob = int((class_mask == 1).sum())
    n_streak = int((class_mask == 2).sum())
    return stem, gray8.shape, n_blob, n_streak


def main():
    """CLI entry point: repatch every tiled image to full size with an annotation overlay."""
    parser = argparse.ArgumentParser(description="Stage 5 repatch tiles to the full frame.")
    parser.add_argument("--tiles", default=str(config.TILES_DIR), help="Folder of per-image tile subfolders.")
    parser.add_argument("--detections", default=str(config.DETECT_DIR), help="Folder of per-image class masks.")
    parser.add_argument("--out", default=str(config.REPATCH_DIR), help="Output folder for full-frame images.")
    args = parser.parse_args()

    image_dirs = sorted(d for d in glob.glob(os.path.join(args.tiles, "*")) if os.path.isdir(d))
    os.makedirs(args.out, exist_ok=True)

    for i, tiles_dir in enumerate(image_dirs):
        stem, shape, n_blob, n_streak = repatch_image(tiles_dir, args.detections, args.out)
        print(f"[{i + 1}/{len(image_dirs)}] {stem}: {shape[1]}x{shape[0]} "
              f"blob_px={n_blob} streak_px={n_streak}")

    print(f"\nWrote {len(image_dirs)} repatched frames and overlays to {args.out}")
    print("Green = blob, red = streak")


if __name__ == "__main__":
    main()
