"""Summarise detection counts, elongation, and whether tiling is reversible."""
import glob
import json
import os

import cv2
import numpy as np

from Pipeline import config


# Bins for the elongation histogram. 3 is the streak cut.
ELONG_BINS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 1e9]


def load_features():
    """Read every detection record written by Stage 3."""
    per_image = []
    elongations = []
    for path in sorted(glob.glob(os.path.join(str(config.DETECT_DIR), "*", "detections.json"))):
        data = json.load(open(path, encoding="utf-8"))
        blobs = streaks = 0
        for tile in data["tiles"]:
            for feature in tile["features"]:
                elongations.append(feature["elongation"])
                if feature["class_id"] == config.CLASS_BLOB:
                    blobs += 1
                else:
                    streaks += 1
        per_image.append({
            "image_id": data["image_id"],
            "blobs": blobs,
            "streaks": streaks,
        })
    return per_image, np.asarray(elongations, dtype=np.float64)


def elongation_summary(values):
    """Percentiles plus a coarse histogram of elongation."""
    if values.size == 0:
        return {"count": 0}
    hist, _ = np.histogram(values, bins=ELONG_BINS)
    labels = ["1-1.5", "1.5-2", "2-3", "3-5", "5-10", "10+"]
    return {
        "count": int(values.size),
        "p50": round(float(np.percentile(values, 50)), 3),
        "p90": round(float(np.percentile(values, 90)), 3),
        "p99": round(float(np.percentile(values, 99)), 3),
        "max": round(float(values.max()), 3),
        "hist": {label: int(count) for label, count in zip(labels, hist)},
    }


def reversibility():
    """Stitch saved tiles and compare them with the preprocessed array. Difference should be 0."""
    rows = []
    for manifest_path in sorted(glob.glob(os.path.join(str(config.TILES_DIR), "*", "tiles_manifest.json"))):
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        stem = manifest["image_id"]
        geometry = manifest["geometry"]
        canvas = np.zeros((geometry["padded_height"], geometry["padded_width"]), dtype=np.uint16)
        tile_size = geometry["tile"]
        for tile in manifest["tiles"]:
            block = cv2.imread(
                os.path.join(str(config.TILES_DIR), stem, tile["name"]),
                cv2.IMREAD_UNCHANGED,
            )
            y0, x0 = tile["y0"], tile["x0"]
            canvas[y0:y0 + tile_size, x0:x0 + tile_size] = block
        original = np.load(os.path.join(str(config.PREPROCESS_DIR), f"{stem}_subtracted16.npy"))
        height, width = geometry["orig_height"], geometry["orig_width"]
        diff = np.abs(canvas[:height, :width].astype(np.int32) - original.astype(np.int32))
        rows.append({
            "image_id": stem,
            "max_abs_diff": int(diff.max()),
            "n_pixels_differ": int((diff > 0).sum()),
        })
    return rows


def main():
    """CLI entry point: write report_stats.json and print the headline numbers."""
    per_image, elongations = load_features()
    checks = reversibility()
    summary = {
        "images": per_image,
        "blobs": sum(row["blobs"] for row in per_image),
        "streaks": sum(row["streaks"] for row in per_image),
        "elongation": elongation_summary(elongations),
        "tiling_reversibility": checks,
        "tiling_exact": all(row["max_abs_diff"] == 0 for row in checks),
    }
    out_dir = str(config.STATS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "report_stats.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"blobs={summary['blobs']} streaks={summary['streaks']}")
    print(f"elongation p50={summary['elongation'].get('p50')} p90={summary['elongation'].get('p90')}")
    print(f"tiling exact={summary['tiling_exact']}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
