# Landmark CSV schema (Kalen -> Pranav / Mustafa)

One CSV per preprocessing variant in `data/landmarks/`, named `landmarks_<variant>.csv`
(e.g. `landmarks_clahe_2.csv`). Generate them with `python scripts/extract_landmarks.py`.
Landmarks are RAW MediaPipe output - normalization happens in the next stage.

| column | meaning |
|---|---|
| image_path | path relative to the dataset folder (`DATASET_DIR` in `config.py`), e.g. `A/A1.jpg` |
| label | letter A-Y (no J/Z) |
| variant | preprocessing variant name, one of the keys of `VARIANTS` in `config.py` (none, gaussian_5, median_3, gamma_1_5, gamma_2_5, clahe_2, clahe_4) |
| detected | 1 if MediaPipe found a hand, else 0 (row is kept; handedness and landmark cells are empty) |
| handedness | Left / Right as reported by MediaPipe (see note below) |
| handedness_score | confidence of handedness |
| x0,y0,z0 ... x20,y20,z20 | 21 landmarks, in MediaPipe order |

## Notes

- **Coordinates.** x and y are normalized to the image (0-1, x by width, y by height). z is
  relative depth with the wrist as the reference, so it is small (some values print in
  scientific notation, e.g. `-1.2e-07`). The values are not yet normalized for hand position,
  size or orientation.
- **Handedness.** MediaPipe's Left/Right label assumes the input is a mirrored, selfie-style
  view, so on an unflipped image it can be reversed. Checked on the Kaggle dataset: the 4
  detected images in the first test run (letter A) were all physically right hands and were
  all labelled `Right`, so the labels match the real hand there. That is a small sample, so
  spot-check a few other letters. Webcam frames from the live demo may behave differently
  depending on whether the frame is flipped before it reaches MediaPipe - check this once
  the webcam data exists (Pranav: relevant to the "mirror left hands" step).
- **Missed detections.** Rows with `detected = 0` are kept so the detection rate can be reported.
  Drop them before training (the proposal says to report the rate and train on detections only).
- **Splits.** The Kaggle images are numbered frames from continuous capture, so neighbouring
  frames (e.g. `A100`, `A101`) are near-duplicates. A random split will leak between train and
  test. The dataset has no person IDs, so the group needs to choose another scheme, such as
  splitting by blocks of consecutive file numbers. Open question for Mustafa.
