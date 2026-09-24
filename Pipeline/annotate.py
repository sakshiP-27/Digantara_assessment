"""Stage 4: convert per-tile class masks into YOLO Ultralytics segmentation polygons."""
import argparse
import glob
import json
import os
import shutil

import cv2
import numpy as np

from Pipeline import config


def dedup_points(points):
    """Drop consecutive duplicate vertices, including a repeated closing point."""
    # An empty contour has nothing to simplify.
    if len(points) == 0:
        return points
    # Keep a point only when it differs from the previous one.
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(np.diff(points, axis=0) != 0, axis=1)
    cleaned = points[keep]
    # OpenCV sometimes repeats the first vertex at the end; YOLO wants an open ring.
    if len(cleaned) > 1 and np.all(cleaned[0] == cleaned[-1]):
        cleaned = cleaned[:-1]
    return cleaned


def contour_points(contour, width, height):
    """Return a polygon with at least 3 vertices, falling back to the bounding box."""
    # Contour vertices are integer (x, y) pixel locations.
    points = dedup_points(contour.reshape(-1, 2).astype(np.float64))
    # A valid segmentation polygon needs three corners.
    if len(points) >= 3:
        return points
    # Tiny components can collapse to one or two vertices after simplification.
    x, y, box_w, box_h = cv2.boundingRect(contour)
    # Expand a zero-size box by one pixel so the rectangle still has area.
    x2 = min(width, x + max(int(box_w), 1))
    y2 = min(height, y + max(int(box_h), 1))
    return np.array([[x, y], [x2, y], [x2, y2], [x, y2]], dtype=np.float64)


def polygon_line(points, class_id, width, height):
    """Format one polygon as a YOLO segmentation row with normalized coordinates."""
    # Divide by the tile size so coordinates sit in [0, 1], as Ultralytics requires.
    xs = np.clip(points[:, 0] / float(width), 0.0, 1.0)
    ys = np.clip(points[:, 1] / float(height), 0.0, 1.0)
    # Interleave x and y at 6 decimal places, which is the usual YOLO precision.
    coords = " ".join(f"{value:.6f}" for pair in zip(xs, ys) for value in pair)
    return f"{class_id} {coords}"


def foreground_contours(binary):
    """Return outer boundaries of each object, including objects nested inside holes."""
    # The full tree keeps a contour for every object, not only the outermost ones.
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    kept = []
    for index, contour in enumerate(contours):
        # Walk parents: even depth is an object, odd depth is a hole in an object.
        depth = 0
        parent = hierarchy[index][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[parent][3]
        if depth % 2 == 0 and len(contour) > 0:
            kept.append(contour)
    return kept


def mask_to_yolo_lines(class_mask):
    """Trace blob and streak regions in a class mask into YOLO polygon lines."""
    # Mask values are class id + 1 so background stays 0 (see detect.py).
    height, width = class_mask.shape[:2]
    lines = []
    counts = {config.CLASS_BLOB: 0, config.CLASS_STREAK: 0}
    for pixel_value, class_id in ((1, config.CLASS_BLOB), (2, config.CLASS_STREAK)):
        # Isolate one class so each contour is a single instance of that class.
        binary = np.where(class_mask == pixel_value, np.uint8(255), np.uint8(0))
        for contour in foreground_contours(binary):
            points = contour_points(contour, width, height)
            lines.append(polygon_line(points, class_id, width, height))
            counts[class_id] += 1
    return lines, counts


def link_tile_image(src, dst):
    """Point the annotation image at the existing tile without duplicating pixel data."""
    # Re-runs should refresh the link if a previous copy is stale.
    if os.path.lexists(dst):
        os.remove(dst)
    try:
        # A hard link keeps one copy of the 16-bit tile on disk.
        os.link(src, dst)
    except OSError:
        # OneDrive and some volumes reject hard links, so copy in that case.
        shutil.copy2(src, dst)


def annotate_image(detect_dir, tiles_dir, out_dir):
    """Write one YOLO label file per tile mask and link the matching tile image."""
    stem = os.path.basename(detect_dir.rstrip("\\/"))
    label_dir = os.path.join(out_dir, "labels", stem)
    image_dir = os.path.join(out_dir, "images", stem)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    n_blob = n_streak = 0
    n_tiles = 0
    # Masks were saved beside detections.json using the tile name plus _mask.
    mask_paths = sorted(glob.glob(os.path.join(detect_dir, "*_mask.png")))
    for mask_path in mask_paths:
        mask_name = os.path.basename(mask_path)
        # Drop the _mask suffix so the label stem matches the tile image stem.
        tile_name = mask_name.replace("_mask.png", ".png")
        tile_path = os.path.join(tiles_dir, stem, tile_name)
        class_mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        lines, counts = mask_to_yolo_lines(class_mask)
        label_path = os.path.join(label_dir, tile_name.replace(".png", ".txt"))
        # An empty file is a valid Ultralytics label for a tile with no objects.
        with open(label_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            if lines:
                fh.write("\n")
        link_tile_image(tile_path, os.path.join(image_dir, tile_name))
        n_blob += counts[config.CLASS_BLOB]
        n_streak += counts[config.CLASS_STREAK]
        n_tiles += 1
    return stem, n_tiles, n_blob, n_streak


def write_dataset_yaml(out_dir):
    """Write the Ultralytics dataset file that names the two segmentation classes."""
    # path '.' is resolved relative to this yaml, so the export stays portable.
    text = (
        "# YOLO Ultralytics segmentation labels for SSA tiles.\n"
        "# Each label row is: class_id x1 y1 x2 y2 ... (coordinates normalized to the tile).\n"
        "path: .\n"
        "train: images\n"
        "val: images\n"
        "names:\n"
        "  0: blob\n"
        "  1: streak\n"
    )
    with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write(text)


def main():
    """CLI entry point: export every detection mask as a YOLO segmentation label."""
    parser = argparse.ArgumentParser(description="Stage 4 YOLO Ultralytics segmentation export.")
    parser.add_argument("--detections", default=str(config.DETECT_DIR),
                        help="Folder of per-image detection masks.")
    parser.add_argument("--tiles", default=str(config.TILES_DIR),
                        help="Folder of per-image 1024x1024 tiles.")
    parser.add_argument("--out", default=str(config.ANNOTATIONS_DIR),
                        help="Output folder for the Ultralytics dataset.")
    args = parser.parse_args()

    image_dirs = sorted(
        d for d in glob.glob(os.path.join(args.detections, "*")) if os.path.isdir(d)
    )
    os.makedirs(args.out, exist_ok=True)

    summary = []
    total_blob = total_streak = total_tiles = 0
    for i, detect_dir in enumerate(image_dirs):
        stem, n_tiles, n_blob, n_streak = annotate_image(detect_dir, args.tiles, args.out)
        total_tiles += n_tiles
        total_blob += n_blob
        total_streak += n_streak
        summary.append({
            "image_id": stem,
            "tiles": n_tiles,
            "blobs": n_blob,
            "streaks": n_streak,
        })
        print(f"[{i + 1}/{len(image_dirs)}] {stem}: tiles={n_tiles} "
              f"blobs={n_blob} streaks={n_streak}")

    write_dataset_yaml(args.out)
    with open(os.path.join(args.out, "annotation_summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"tiles": total_tiles, "blobs": total_blob, "streaks": total_streak,
                   "images": summary}, fh, indent=2)
    print(f"\nWrote {total_tiles} label files: blobs={total_blob} streaks={total_streak}")
    print(f"Ultralytics dataset written to {args.out}")


if __name__ == "__main__":
    main()
