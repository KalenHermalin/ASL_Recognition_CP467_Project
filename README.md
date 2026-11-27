# CP467 - Static Hand Sign Recognition with MediaPipe Hands

ASL fingerspelling (24 static letters) using MediaPipe Hands landmarks, compared against a
classical skin-segmentation baseline. Course: CP467 Image Processing & Recognition.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_dataset.py     # one-time, needs a Kaggle API token
python scripts/extract_landmarks.py    # caches landmarks to data/landmarks/
```

## Landmark extraction (Kalen)
```bash
python scripts/extract_landmarks.py --variants none gamma_2_5            # full Kaggle run (72,000 images)
python scripts/extract_landmarks.py --letters A B C D E --limit 200      # quick subset (every Nth image)
python scripts/extract_landmarks.py --variants none gamma_2_5 \
       --lighting dark_mild dark_moderate --only-split test              # dark runs, test images only
python scripts/extract_landmarks.py --source webcam --variants none      # data/webcam/<person>/<letter>/
python scripts/check_handedness.py --csv data/landmarks/landmarks_webcam_none.csv
```
- Variants and lighting levels are defined in `src/asl/config.py` (`VARIANTS`, `LIGHTING_LEVELS`);
  `src/asl/lighting.py` makes the synthetic dark images (seeded per image, so every variant sees the same one).
- Output: `data/landmarks/landmarks_<variant>[_<lighting>].csv` (columns in `docs/SCHEMA.md`) plus
  `results/tables/detection_<tag>.csv` and `timing_<tag>.csv`. Existing CSVs are skipped unless `--overwrite`.
- `--only-split test` reads `results/splits/split.csv` (make it first with `scripts/make_split.py`) and
  only processes the test images. Use it for dark runs: classifiers train on normal light only, so
  dark CSVs need just the test images (about 8 minutes per run instead of 30). It is refused for normal light.
- Webcam CSVs are named `landmarks_webcam_<variant>.csv`; folder names under `data/webcam/` are the
  groups for the handedness check (for example `lucas_right/`, `lucas_left/`).
- Report figures: `python scripts/make_report_figures.py` (tables, heatmap, per-letter chart) and
  `python scripts/make_qualitative_figures.py` (sample outputs). Both write to `results/report/` and need
  the A-E subset runs (`data/landmarks_subset_A-E/`, `results/tables_subset_A-E/`), which are kept local.

## Classifiers + evaluation (Mustafa)
```bash
python scripts/make_split.py             # ONE shared split -> results/splits/split.csv (baseline must use it too)
python scripts/train_eval.py --variant gamma_2_5 --features combined --classifier svm \
       --test-lighting normal dark_mild dark_moderate dark_severe
python scripts/run_ablations.py          # one stage at a time -> results/tables/ablation_<stage>.csv
python -m pytest tests                   # (set PYTHONPATH=src if the package isn't pip-installed)
```
Add `--sample` to any of them to run on the small CSVs in `data/samples/`.
- Split: blocks of 150 consecutive frames per letter (~25% test), 15-frame gap, seeded.
- Train on detected frames only; test keeps misses, so `detection_rate`, `accuracy` (detected
  frames) and `e2e_accuracy` (miss = wrong; compare this one against the baseline) are all reported.
- `src/asl/feature_modes.py` is a placeholder for Pranav's normalization/features
  (`build_features(df, mode)`); swap in his code there.

## Layout
```
src/asl/       shared package (config, preprocessing, landmarks, ...)
scripts/       runnable entry points (download, extract, train, evaluate)
data/          raw/ webcam/ landmarks/   (git-ignored, see .gitignore)
notebooks/     exploration
results/       figures/ and tables/ for the report
docs/          schema + notes
tests/
```

## Pipeline owners
Data collection / webcam / demo: Lucas - Preprocessing + landmarks: Kalen -
Normalization + features: Pranav - Baseline: Bibek - Classifiers + evaluation: Mustafa
