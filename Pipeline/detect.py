"""Stage 3: detect features per tile and classify each as a blob (star) or streak (object)."""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from skimage import measure, morphology

from Pipeline import config


def tile_threshold(tile, k=config.DETECT_SIGMA_K):
    """Compute one threshold for a whole tile at median + k*sigma. Kept for before/after checks."""
    # Median is the background level for this tile.
    med = float(np.median(tile))
    # MAD gives an outlier-resistant spread that ignores bright features.
    mad = float(np.median(np.abs(tile - med)))
    # Convert MAD to Gaussian sigma; fall back to std if the tile is nearly flat.
    sigma = 1.4826 * mad if mad > 0 else float(tile.std()) + 1.0
    # Features must exceed background by k noise sigmas to be detected.
    return med + k * sigma, med, sigma


def local_background(tile, block=config.BG_BLOCK):
    """Estimate background and noise on a grid of blocks, then expand back to every pixel."""
    # A single median for the whole tile is too coarse when one side is brighter than the other.
    height, width = tile.shape
    rows, cols = height // block, width // block
    # Reshape into non-overlapping blocks so each block gets its own median and MAD.
    patches = tile[:rows * block, :cols * block].reshape(rows, block, cols, block).swapaxes(1, 2)
    flat = patches.reshape(rows, cols, -1)
    med = np.median(flat, axis=2)
    mad = np.median(np.abs(flat - med[:, :, None]), axis=2)
    # Same MAD-to-sigma conversion as the full-frame background estimate.
    sigma = np.where(mad > 0, 1.4826 * mad, flat.std(axis=2) + 1.0).astype(np.float32)
    # Smooth the block values across the tile so the threshold does not jump on a grid.
    med_map = cv2.resize(med.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    sigma_map = cv2.resize(sigma, (width, height), interpolation=cv2.INTER_LINEAR)
    return med_map, sigma_map


def build_binary_mask(tile, threshold, min_area=config.MIN_FEATURE_AREA):
    """Threshold a tile and clean the binary mask to suppress single-pixel noise."""
    # Pixels brighter than the adaptive threshold are candidate feature pixels.
    mask = tile > threshold
    # Close 1-pixel gaps so broken streak segments join into one component.
    mask = morphology.closing(mask, morphology.footprint_rectangle((3, 3)))
    # Label components so we can drop the small ones by pixel count.
    labels = measure.label(mask, connectivity=2)
    # Count pixels per label (index 0 is background).
    counts = np.bincount(labels.ravel())
    # Labels with fewer pixels than the area floor are noise to be removed.
    too_small = np.where(counts < min_area)[0]
    # Zero out every under-sized component in one vectorized pass.
    mask[np.isin(labels, too_small)] = False
    return mask


def classify_region(region):
    """Label a connected region as streak or blob from its elongation and area."""
    # Major/minor axis lengths come from the region's second moments (PCA of pixel cloud).
    major = float(region.axis_major_length)
    # Guard the minor axis against zero for near-linear regions.
    minor = float(region.axis_minor_length) if region.axis_minor_length > 0 else 0.5
    # Elongation is the aspect ratio of the fitted ellipse; high = line-like.
    elongation = major / minor
    # Long, sufficiently large regions are space-object streaks.
    is_streak = elongation >= config.STREAK_ELONGATION and region.area >= config.STREAK_MIN_AREA
    # Everything else compact is treated as a star blob.
    cls = config.CLASS_STREAK if is_streak else config.CLASS_BLOB
    return cls, elongation


def detect_tile(tile, k=config.DETECT_SIGMA_K, min_area=config.MIN_FEATURE_AREA):
    """Run threshold, cleanup, labeling and classification on a single tile."""
    # Work in float for stable threshold arithmetic.
    tile_f = tile.astype(np.float32)
    # Background and noise vary inside the tile, so the cut follows a local map.
    med_map, sigma_map = local_background(tile_f)
    threshold = med_map + k * sigma_map
    med = float(np.median(med_map))
    sigma = float(np.median(sigma_map))
    # Produce a cleaned binary feature mask.
    binary = build_binary_mask(tile_f, threshold, min_area)
    # Label connected components with 8-connectivity to keep diagonal streaks intact.
    labels = measure.label(binary, connectivity=2)
    # Class-labeled mask: 0 background, 1 blob, 2 streak (offset so classes stay non-zero).
    class_mask = np.zeros(tile.shape, dtype=np.uint8)
    # Per-feature records feed the annotation and reporting stages.
    features = []
    for region in measure.regionprops(labels):
        # Skip anything below the area floor that survived closing.
        if region.area < min_area:
            continue
        # Assign a class from the region's shape.
        cls, elongation = classify_region(region)
        # Paint this region into the class mask using class id + 1 as the paint value.
        class_mask[labels == region.label] = cls + 1
        # Record geometry and shape metrics for YOLO export and analysis.
        minr, minc, maxr, maxc = region.bbox
        features.append({
            "class_id": cls,
            "class_name": config.CLASS_NAMES[cls],
            "area": int(region.area),
            "elongation": round(elongation, 3),
            "eccentricity": round(float(region.eccentricity), 3),
            "bbox": [int(minc), int(minr), int(maxc), int(maxr)],
            "centroid": [round(float(region.centroid[1]), 1), round(float(region.centroid[0]), 1)],
        })
    # Summarize the background/threshold used so the report can cite real numbers.
    meta = {
        "threshold": round(float(np.median(threshold)), 2),
        "bg_median": round(med, 2),
        "sigma": round(sigma, 3),
    }
    return class_mask, features, meta


def process_image(tiles_dir, out_dir, k, min_area):
    """Detect features across every tile of one image and persist masks + records."""
    # Identify the image id from the tile folder name.
    stem = os.path.basename(tiles_dir.rstrip("/"))
    # Load the tile manifest to iterate tiles in grid order.
    manifest = json.load(open(os.path.join(tiles_dir, "tiles_manifest.json")))
    # Prepare a per-image output folder for masks and records.
    img_out = os.path.join(out_dir, stem)
    os.makedirs(img_out, exist_ok=True)
    # Aggregate counts and all per-tile feature records.
    n_blob = n_streak = 0
    tile_results = []
    for t in manifest["tiles"]:
        # Load the 16-bit tile written by Stage 2.
        tile = cv2.imread(os.path.join(tiles_dir, t["name"]), cv2.IMREAD_UNCHANGED)
        # Detect and classify features in this tile.
        class_mask, features, meta = detect_tile(tile, k, min_area)
        # Save the class mask as a PNG (values 0/1/2) for the annotation stage.
        mask_name = t["name"].replace(".png", "_mask.png")
        cv2.imwrite(os.path.join(img_out, mask_name), class_mask)
        # Tally class counts for the summary.
        n_blob += sum(1 for f in features if f["class_id"] == config.CLASS_BLOB)
        n_streak += sum(1 for f in features if f["class_id"] == config.CLASS_STREAK)
        # Keep the per-tile record with its grid position for later stages.
        tile_results.append({"tile": t["name"], "row": t["row"], "col": t["col"],
                             "meta": meta, "features": features})
    # Persist all detections for this image in one JSON.
    with open(os.path.join(img_out, "detections.json"), "w") as fh:
        json.dump({"image_id": stem, "params": {"sigma_k": k, "min_area": min_area},
                   "tiles": tile_results}, fh)
    return stem, n_blob, n_streak


def main():
    """CLI entry point: run detection + classification on all tiled images."""
    parser = argparse.ArgumentParser(description="Stage 3 detection and blob/streak classification.")
    parser.add_argument("--tiles", default=str(config.TILES_DIR), help="Folder of per-image tile subfolders.")
    parser.add_argument("--out", default=str(config.DETECT_DIR), help="Output folder for detections.")
    parser.add_argument("--sigma-k", type=float, default=config.DETECT_SIGMA_K, help="Threshold in sigmas.")
    parser.add_argument("--min-area", type=int, default=config.MIN_FEATURE_AREA, help="Min feature area (px).")
    args = parser.parse_args()

    # Find every per-image tile folder (those containing a manifest).
    image_dirs = sorted(d for d in glob.glob(os.path.join(args.tiles, "*")) if os.path.isdir(d))
    # Ensure the detections output folder exists.
    os.makedirs(args.out, exist_ok=True)

    total_blob = total_streak = 0
    for i, d in enumerate(image_dirs):
        # Detect features for one image and accumulate class counts.
        stem, n_blob, n_streak = process_image(d, args.out, args.sigma_k, args.min_area)
        total_blob += n_blob
        total_streak += n_streak
        print(f"[{i + 1}/{len(image_dirs)}] {stem}: blobs={n_blob} streaks={n_streak}")

    print(f"\nTotal across {len(image_dirs)} images: blobs={total_blob} streaks={total_streak}")
    print(f"Masks + detections.json written to {args.out}")


if __name__ == "__main__":
    main()
