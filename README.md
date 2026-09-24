# SSA blob and streak annotation

Annotates stars (blobs) and space objects (streaks) in the 10 assessment FITS frames, then exports YOLO Ultralytics segmentation labels and full-frame overlays.

The written answers to Question 2 are in `Q2_response.md`.

## Setup

From this folder, with Python 3.12:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install astropy numpy opencv-python-headless scikit-image scipy
```

Put the 10 raw `.fits` files in `Dataset/`.

## Run

Run the stages in order. Each one reads the previous stage's output.

```powershell
python -m Pipeline.preprocess
python -m Pipeline.tiling
python -m Pipeline.detect
python -m Pipeline.annotate
python -m Pipeline.repatch
```

Shared settings (tile size, sigma factor, class ids) are in `Pipeline/config.py`.

## What you get

| Folder | Contents |
|---|---|
| `Output/preprocessed/` | 8-bit preview PNGs and 16-bit background-subtracted arrays |
| `Output/tiles/` | 70 exact 1024x1024 tiles per image, plus `tiles_manifest.json` |
| `Output/detections/` | Class masks and `detections.json` per image |
| `Output/annotations/` | YOLO dataset: `images/`, `labels/`, `data.yaml` |
| `Output/repatched/` | Full 9568x6380 frames for inspection |

## How to read the results

An overlay (`Output/repatched/*_overlay.png`) is the full frame. Green pixels are blobs (stars). Red pixels are streaks. Zoom to 100% or closer; the marks are only a few pixels wide. The matching `*_repatched.png` is the same frame with no color.

A label file (`Output/annotations/labels/<image>/<image>_r<row>_c<col>.txt`) matches one tile. Each line is one object:

```text
class_id x1 y1 x2 y2 ... xn yn
```

`0` is a blob, `1` is a streak. The coordinates are fractions of the 1024 x 1024 tile (0 to 1). An empty file means that tile had no detection. `data.yaml` names the two classes for Ultralytics.
