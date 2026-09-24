# Question 2 Annotation method

The 10 FITS frames go through a short pipeline in the Pipeline folder. Background is removed first, then each frame is cut into exact 1024 x 1024 tiles. Sources are found on each tile and split into blobs and streaks. The kept pixels are stored as a mask, and the outline of each mask region is saved as a YOLO segmentation polygon. Tiles are also stitched back to full size so the overlays can be checked. The numbers below are what the scripts printed.

## (a) Pre-processing

Each file is a single-band 16-bit FITS frame of 9568 x 6380 pixels, read from the primary HDU. The sky level is not the same in every frame, so the correction is computed per image rather than with one global constant.

For each frame the sky level is the median. The noise is the median absolute deviation times 1.4826. Median and MAD are used because a handful of bright stars would pull a mean and a standard deviation up.

On the eight UUID frames the sky is about 1 to 3 counts, and the noise is about 0.9 to 4.4. The two CAM_B frames are brighter: sky about 45 to 46, noise about 43.

The median is subtracted, and anything below zero is set to zero. The sky then sits near zero and the sources stay positive. This is saved as a 16-bit array (`*_subtracted16.npy`). Detection uses that array. An 8-bit preview (`*_preview.png`) is also saved, by stretching the 50th to the 99.9th percentile onto 0-255. The preview is only for looking at the frame. It is not used for the threshold.

## (b) Image boundaries when tiling

9568 and 6380 are not multiples of 1024.

- Width: 9 x 1024 = 9216, with 352 pixels left over. The next multiple is 10 x 1024 = 10240, so 672 columns are added on the right.
- Height: 6 x 1024 = 6144, with 236 pixels left over. The next multiple is 7 x 1024 = 7168, so 788 rows are added on the bottom.

The pad is zeros. That matches the background-subtracted floor, and it does not invent sources. The original pixels are never cropped, scaled, or overlapped. Each frame becomes a 10 x 7 grid of exact 1024 x 1024 tiles (70 tiles, 700 in total). `tiles_manifest.json` stores the original size and the top-left of every tile. Repatching copies each tile back to that position and cuts the canvas to 9568 x 6380, which removes only the pad.

Assumption for this step: a source that crosses a tile edge is stored as two annotations, one in each tile. The padded strip is not part of the original image and is removed in the repatched frames.

## (c) Short or fat streaks vs blobs

Detection and classification happen on each tile separately, because the noise is not uniform across the frame.

A pixel is kept if it is above that tile's median plus 8 sigma. Sigma comes from the MAD, using the same 1.4826 factor. The mask is then closed with a 3 x 3 kernel, so a line with a one-pixel gap stays one object. Components smaller than 5 pixels are dropped. Most of those are single hot pixels.

Each remaining component is fit with an ellipse from the pixel second moments. Elongation is the long axis divided by the short axis. A component is a streak only when elongation is at least 3 and area is at least 10 pixels. A shorter or fatter object stays a blob. A normal star is roughly round (elongation near 1), so it stays a blob even if it is bright. Across all 10 frames this gave 142,183 blobs and 10,747 streaks.

Those pixels are written into a class mask: 0 is background, 1 is a blob, 2 is a streak. The YOLO file does not store a box around the object. Each line is the outline of one region in that mask. The first number is the class (0 blob, 1 streak). The rest are the outline corners, with x and y divided by 1024 so they fall between 0 and 1. One line is one object.

Assumption for this step: compact sources are stars and elongated ones are space objects. The same 8 sigma cut is used on every tile, including the two CAM_B frames. Those frames have noise around 43 counts, so many more pixels pass the cut. On the second CAM_B frame, 15 tiles each contain thousands of components. Where those pixels sit next to each other, the 3 x 3 closing joins the gaps and the tile looks like a solid green square in the overlay. The cut was left the same there, with no separate tuning.

## (d) Faint or small blobs as stars

There is no separate pass for faint stars. A small or faint source is labelled as a star (blob, class 0) when it passes the same tests as any other blob and is not long enough to be a streak.

That comes down to three checks:

- Brighter than the local median + 8 sigma. Eight sigma is high enough that ordinary noise in a quiet tile almost never reaches it, so a faint star is marked only when it clearly stands above the local sky. A lower cut would start painting noise. A star that only shows up because of the display stretch, but still sits inside the noise, is left out.
- At least 5 pixels. One to four pixels is the size of a hot pixel or a noise spike, not a star, so those are not labelled.
- Elongation under 3, or area under 10, which keeps it as a blob rather than a streak.

On the quieter frames a lot of faint stars still appear white in the preview and in the repatched image. They were not marked, because they did not clear 8 sigma. The green marks are the sharper ones.

## Where to look

Full frames for checking are in `Output/repatched`. `*_repatched.png` is the stretched frame. `*_overlay.png` is the same frame with blobs in green and streaks in red. YOLO text files are in `Output/annotations/labels`, one file per tile. Class names are in `data.yaml`.