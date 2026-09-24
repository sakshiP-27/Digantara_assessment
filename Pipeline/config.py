"""Shared configuration constants for the SSA annotation pipeline."""
from pathlib import Path

# Root of the project (parent of this pipeline package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Folder holding the 10 raw FITS images provided for the assessment.
DATASET_DIR = PROJECT_ROOT / "Dataset"

# All generated artifacts live under this output folder.
OUTPUT_DIR = PROJECT_ROOT / "Output"

# Stage 1 writes 8-bit preview PNGs and 16-bit background-subtracted arrays here.
PREPROCESS_DIR = OUTPUT_DIR / "preprocessed"

# Full-frame sensor dimensions from the FITS spec (width x height in pixels).
IMAGE_WIDTH = 9568
IMAGE_HEIGHT = 6380

# Target tile edge length required by the assessment.
TILE_SIZE = 1024

# Number of tile columns needed to cover the padded width (ceil(9568 / 1024) = 10).
TILE_COLS = -(-IMAGE_WIDTH // TILE_SIZE)

# Number of tile rows needed to cover the padded height (ceil(6380 / 1024) = 7).
TILE_ROWS = -(-IMAGE_HEIGHT // TILE_SIZE)

# Padded canvas width so tiling is exact (10 * 1024 = 10240).
PADDED_WIDTH = TILE_COLS * TILE_SIZE

# Padded canvas height so tiling is exact (7 * 1024 = 7168).
PADDED_HEIGHT = TILE_ROWS * TILE_SIZE

# Stage 2 writes tiled images (and later their masks) under this folder.
TILES_DIR = OUTPUT_DIR / "tiles"

# Stage 3 writes detection results (masks + per-tile feature records) here.
DETECT_DIR = OUTPUT_DIR / "detections"

# Stage 4 writes Ultralytics segmentation labels paired with the tile images.
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations"

# Stage 5 writes full-frame images with the tile annotations painted back on.
REPATCH_DIR = OUTPUT_DIR / "repatched"

# Detection threshold expressed as this many robust sigmas above the tile background.
DETECT_SIGMA_K = 8.0

# Minimum connected-component area (pixels) to count as a real feature, not noise.
MIN_FEATURE_AREA = 5

# Elongation (major/minor axis ratio) at or above this marks a region as a streak.
STREAK_ELONGATION = 3.0

# A region also counts as a streak only if its area is at least this large.
STREAK_MIN_AREA = 10

# Class ids for the two annotation classes (YOLO convention: 0-indexed).
CLASS_BLOB = 0   # stars / points
CLASS_STREAK = 1  # space objects / lines

# Human-readable class names paired with the ids above.
CLASS_NAMES = {CLASS_BLOB: "blob", CLASS_STREAK: "streak"}

# Lower percentile used to set the black point when stretching for visualization.
STRETCH_LOW_PCT = 50.0

# Upper percentile used to set the white point when stretching for visualization.
STRETCH_HIGH_PCT = 99.9
