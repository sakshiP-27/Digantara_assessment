"""Stage 1: per-image adaptive preprocessing of raw 16-bit FITS SSA imagery."""
import argparse
import glob
import json
import os

import cv2
import numpy as np
from astropy.io import fits

from Pipeline import config


def load_fits(path):
    """Read the primary HDU of a FITS file as a 2D numpy array."""
    with fits.open(path) as hdul:
        # The provided images store the science frame in the primary HDU.
        data = hdul[0].data
    # Guarantee a native-endian float array for stable arithmetic downstream.
    return np.asarray(data, dtype=np.float32)


def robust_background_stats(image):
    """Estimate background level and noise using median and MAD (outlier-safe)."""
    # Median is the robust background level since most pixels are sky/noise.
    median = float(np.median(image))
    # MAD (median absolute deviation) resists bright stars/streaks skewing the noise estimate.
    mad = float(np.median(np.abs(image - median)))
    # Convert MAD to a Gaussian-equivalent sigma using the standard 1.4826 scale factor.
    sigma = 1.4826 * mad if mad > 0 else float(image.std())
    return median, sigma


def subtract_background(image, median):
    """Remove the constant sky background and clip negative values to zero."""
    # Subtracting the median flattens the dark background toward zero.
    subtracted = image - median
    # Negative residuals carry no feature signal, so clamp them at zero.
    return np.clip(subtracted, 0, None)


def stretch_to_8bit(image, low_pct, high_pct):
    """Percentile-stretch a background-subtracted frame into an 8-bit preview."""
    # Black point at a low percentile suppresses residual background noise.
    lo = np.percentile(image, low_pct)
    # White point at a high percentile keeps faint features visible without saturating on the brightest.
    hi = np.percentile(image, high_pct)
    # Guard against a degenerate range where lo == hi.
    if hi <= lo:
        hi = lo + 1.0
    # Linearly map [lo, hi] to [0, 1] and clip the tails.
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    # Convert the normalized frame to 8-bit for PNG export and CV inspection.
    return (scaled * 255.0).astype(np.uint8)


def preprocess_image(path):
    """Run the full Stage-1 preprocessing on one FITS file and return artifacts + stats."""
    # Load the raw 16-bit frame as float for processing.
    raw = load_fits(path)
    # Compute robust background level and noise sigma for this specific image.
    median, sigma = robust_background_stats(raw)
    # Flatten the background so features stand out on a near-zero floor.
    subtracted = subtract_background(raw, median)
    # Produce a human-viewable 8-bit stretch for inspection and overlays.
    preview8 = stretch_to_8bit(subtracted, config.STRETCH_LOW_PCT, config.STRETCH_HIGH_PCT)
    # Keep the background-subtracted signal as 16-bit to preserve dynamic range for detection.
    subtracted16 = np.clip(subtracted, 0, 65535).astype(np.uint16)
    # Bundle the numeric stats so later stages and the report can reuse them.
    stats = {
        "file": os.path.basename(path),
        "shape": list(raw.shape),
        "raw_min": float(raw.min()),
        "raw_max": float(raw.max()),
        "bg_median": median,
        "noise_sigma": sigma,
        # A per-image detection threshold suggestion at median + 5*sigma above background.
        "suggested_threshold": 5.0 * sigma,
    }
    return subtracted16, preview8, stats


def main():
    """CLI entry point: preprocess every FITS image and write previews + stats."""
    parser = argparse.ArgumentParser(description="Stage 1 preprocessing of FITS SSA imagery.")
    parser.add_argument("--dataset", default=str(config.DATASET_DIR), help="Folder of raw FITS files.")
    parser.add_argument("--out", default=str(config.PREPROCESS_DIR), help="Output folder for artifacts.")
    args = parser.parse_args()

    # Collect all FITS files in a stable, reproducible order.
    files = sorted(glob.glob(os.path.join(args.dataset, "*.fits")))
    # Create the output directory tree if it does not exist yet.
    os.makedirs(args.out, exist_ok=True)
    # Accumulate per-image statistics for a single summary JSON.
    all_stats = []

    for i, path in enumerate(files):
        # Process one image end-to-end through Stage 1.
        subtracted16, preview8, stats = preprocess_image(path)
        # Use the FITS stem as the base name for all derived files.
        stem = os.path.splitext(os.path.basename(path))[0]
        # Save the 8-bit stretched preview as PNG for visual inspection.
        cv2.imwrite(os.path.join(args.out, f"{stem}_preview.png"), preview8)
        # Save the 16-bit background-subtracted frame as a compressed .npy for the detection stage.
        np.save(os.path.join(args.out, f"{stem}_subtracted16.npy"), subtracted16)
        # Record the stats and log progress to the console.
        all_stats.append(stats)
        print(f"[{i + 1}/{len(files)}] {stats['file']} "
              f"bg={stats['bg_median']:.1f} sigma={stats['noise_sigma']:.2f} "
              f"thr={stats['suggested_threshold']:.1f}")

    # Write the combined statistics so downstream stages and the report can consume them.
    with open(os.path.join(args.out, "preprocess_stats.json"), "w") as fh:
        json.dump(all_stats, fh, indent=2)
    print(f"\nWrote {len(all_stats)} previews + arrays and preprocess_stats.json to {args.out}")


if __name__ == "__main__":
    main()
