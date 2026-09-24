Stages to do:

Stage 1 — Preprocessing (preprocess.py) Per-image adaptive: estimate background (median/MAD), subtract, then percentile-clip + normalize to 8-bit for visualization. Keep 16-bit for detection.

Stage 2 — Tiling (tiling.py) Solve the 9568×6380 → 1024×1024 problem. 9568 = 9×1024 + 352, 6380 = 6×1024 + 236. I'll pad to the next multiple (10×1024=10240 wide, 7×1024=7168 tall) so tiling is exact and fully reversible for repatching. This directly answers Q2(b).

Stage 3 — Detection + classification (detect.py) Adaptive threshold → connected components → classify each blob vs streak by shape (aspect ratio / elongation / area / eccentricity via PCA on the pixel cloud). This answers Q2(c) and Q2(d).

Stage 4 — Masks + YOLO export (annotate.py) Build pixel masks per tile, convert to YOLO Ultralytics segmentation polygons (class 0 = blob/star, class 1 = streak/object).

Stage 5 — Repatch (repatch.py) Stitch tiles + overlay annotations back to full 9568×6380 for visual inspection.

Stage 6 — Report + packaging 4-page Q2 write-up, README with run instructions, zip.