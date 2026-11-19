# Landmark CSV schema (Kalen -> Pranav / Mustafa)

One CSV per preprocessing variant and lighting condition in `data/landmarks/`:
`landmarks_<variant>.csv` for normal lighting (e.g. `landmarks_clahe_2.csv`) and
`landmarks_<variant>_<lighting>.csv` for synthetic dark runs (e.g. `landmarks_clahe_2_dark_moderate.csv`).
Generate them with `python scripts/extract_landmarks.py` (see `--lighting`).
Landmarks are RAW MediaPipe output - normalization happens in the next stage.

| column | meaning |
|---|---|
| image_path | path relative to the dataset folder (`DATASET_DIR` in `config.py`), e.g. `A/A1.jpg` |
| label | letter A-Y (no J/Z) |
| variant | preprocessing variant name, one of the keys of `VARIANTS` in `config.py` (none, gaussian_5, median_3, gamma_1_5, gamma_2_5, clahe_2, clahe_4, resize_128, resize_300, resize_400) |
| lighting | lighting condition, one of the keys of `LIGHTING_LEVELS` in `config.py`: `normal` (images untouched) or synthetic `dark_mild` / `dark_moderate` / `dark_severe` / `dark_moderate_nonoise` (see `src/asl/lighting.py`) |
| detected | 1 if MediaPipe found a hand, else 0 (row is kept; handedness and landmark cells are empty) |
| handedness | Left / Right as reported by MediaPipe (see note below) |
| handedness_score | confidence of handedness |
| x0,y0,z0 ... x20,y20,z20 | 21 landmarks, in MediaPipe order |

## Notes

- **Coordinates.** x and y are normalized to the image (0-1, x by width, y by height). z is
  relative depth with the wrist as the reference, so it is small (some values print in
  scientific notation, e.g. `-1.2e-07`). The values are not yet normalized for hand position,
  size or orientation.
- **Handedness (read before the mirroring step).** MediaPipe's Left/Right label assumes a mirrored,
  selfie-style input, so on an unflipped image it can be reversed. In the full run (normal
  lighting) about 95% of detected hands are labelled `Right` and about 5% `Left` (none: 52,649 /
  2,719; gamma_2_5: 58,426 / 2,524). Checking many images by eye, the `Left`-labelled ones look
  like the same hand as the `Right`-labelled ones: the Kaggle set appears to show one hand
  throughout. So the `Left` labels are most likely MediaPipe mistakes, not real left hands. Which
  physical side that hand is has not been verified. Consequences:
  - Do NOT mirror hands in the Kaggle data based on `handedness`; that would flip about 2,500
    correctly-oriented hands.
  - For webcam / other data, the mirroring rule has to be calibrated: record a known right and a
    known left hand through the same capture path (flipped or not) and see which label each gets.
    Then mirror so that every hand ends up matching the Kaggle hand's orientation.
- **Class balance after dropping misses.** Detection varies by letter (full run, gamma_2_5: most
  letters 82-98%, but M = 58% and N = 47%; `none`: M = 53%, N = 42%). So after dropping
  `detected = 0` rows there are far fewer M and N samples than other letters (about 1,400-1,750
  of 3,000), and the surviving M/N images are not a random sample. Mention this when reporting
  per-letter accuracy for M and N.
- **Missed detections.** Rows with `detected = 0` are kept so the detection rate can be reported.
  Drop them before training (the proposal says to report the rate and train on detections only).
- **Splits.** The Kaggle images are numbered frames from continuous capture, so neighbouring
  frames (e.g. `A100`, `A101`) are near-duplicates. A random split will leak between train and
  test. The dataset has no person IDs, so the group needs to choose another scheme, such as
  splitting by blocks of consecutive file numbers. Open question for Mustafa.
