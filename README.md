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
