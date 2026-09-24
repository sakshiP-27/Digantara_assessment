"""Stage 2: split full frames into exact 1024x1024 tiles via reversible edge padding."""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from Pipeline import config


def compute_padding(height, width, tile=config.TILE_SIZE):
    """Return the bottom/right padding needed to reach the next multiple of the tile size."""
    # Rows required to cover the height, rounded up so no data is dropped.
    rows = -(-height // tile)
    # Columns required to cover the width, rounded up so no data is dropped.
    cols = -(-width // tile)
    # Extra pixels to append at the bottom to reach an exact multiple of the tile size.
    pad_bottom = rows * tile - height
    # Extra pixels to append at the right to reach an exact multiple of the tile size.
    pad_right = cols * tile - width
    return rows, cols, pad_bottom, pad_right


def pad_image(image, pad_bottom, pad_right):
    """Pad the bottom and right edges with zeros so the canvas tiles exactly."""
    # Zero padding is used because the background floor is already ~0 after Stage 1.
    return np.pad(image, ((0, pad_bottom), (0, pad_right)), mode="constant", constant_values=0)


def tile_image(image, tile=config.TILE_SIZE):
    """Slice a padded image into a grid of non-overlapping tile-sized blocks."""
    # Record original size so the geometry can be reversed exactly during repatching.
    height, width = image.shape
    # Determine grid shape and required padding for this specific image.
    rows, cols, pad_bottom, pad_right = compute_padding(height, width, tile)
    # Pad the frame so every tile is a full tile x tile block.
    padded = pad_image(image, pad_bottom, pad_right)
    # Collect tiles together with their grid position for reproducible naming.
    tiles = []
    for r in range(rows):
        for c in range(cols):
            # Crop one tile-sized block at grid cell (r, c).
            y0, x0 = r * tile, c * tile
            block = padded[y0:y0 + tile, x0:x0 + tile]
            tiles.append((r, c, y0, x0, block))
    # Return tiles plus the geometry metadata needed to stitch them back later.
    geometry = {
        "orig_height": int(height),
        "orig_width": int(width),
        "tile": int(tile),
        "rows": int(rows),
        "cols": int(cols),
        "pad_bottom": int(pad_bottom),
        "pad_right": int(pad_right),
        "padded_height": int(padded.shape[0]),
        "padded_width": int(padded.shape[1]),
    }
    return tiles, geometry


def process_file(npy_path, out_dir, tile=config.TILE_SIZE):
    """Tile one preprocessed 16-bit array and save each tile as an 8-bit PNG + metadata."""
    # Load the Stage-1 background-subtracted 16-bit frame.
    image = np.load(npy_path)
    # Derive a clean image id from the filename (drop the _subtracted16 suffix).
    stem = os.path.basename(npy_path).replace("_subtracted16.npy", "")
    # Create a per-image tile folder to keep outputs organized.
    img_out = os.path.join(out_dir, stem)
    os.makedirs(img_out, exist_ok=True)
    # Split the frame into exact tiles and capture the reversible geometry.
    tiles, geometry = tile_image(image, tile)
    # Track lightweight per-tile records for the manifest.
    tile_records = []
    for r, c, y0, x0, block in tiles:
        # Name tiles by grid row/column for stable, sortable ordering.
        name = f"{stem}_r{r}_c{c}.png"
        # Save the 16-bit tile losslessly as a PNG so detection keeps full range.
        cv2.imwrite(os.path.join(img_out, name), block.astype(np.uint16))
        # Record where this tile sits in the original frame for repatching.
        tile_records.append({"name": name, "row": r, "col": c, "y0": y0, "x0": x0})
    # Persist the geometry + tile list so Stage 5 can reconstruct the full image.
    manifest = {"image_id": stem, "geometry": geometry, "tiles": tile_records}
    with open(os.path.join(img_out, "tiles_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return stem, len(tile_records), geometry


def main():
    """CLI entry point: tile every preprocessed frame into exact 1024x1024 blocks."""
    parser = argparse.ArgumentParser(description="Stage 2 tiling into exact 1024x1024 tiles.")
    parser.add_argument("--in", dest="inp", default=str(config.PREPROCESS_DIR),
                        help="Folder of Stage-1 *_subtracted16.npy arrays.")
    parser.add_argument("--out", default=str(config.TILES_DIR), help="Output folder for tiles.")
    parser.add_argument("--tile", type=int, default=config.TILE_SIZE, help="Tile edge length.")
    args = parser.parse_args()

    # Gather all preprocessed arrays in a stable order.
    arrays = sorted(glob.glob(os.path.join(args.inp, "*_subtracted16.npy")))
    # Ensure the tiles output directory exists.
    os.makedirs(args.out, exist_ok=True)

    total_tiles = 0
    for i, path in enumerate(arrays):
        # Tile one image and log its grid shape.
        stem, count, geom = process_file(path, args.out, args.tile)
        total_tiles += count
        print(f"[{i + 1}/{len(arrays)}] {stem}: {geom['rows']}x{geom['cols']} grid = {count} tiles "
              f"(padded to {geom['padded_width']}x{geom['padded_height']})")

    print(f"\nWrote {total_tiles} tiles across {len(arrays)} images to {args.out}")


if __name__ == "__main__":
    main()
