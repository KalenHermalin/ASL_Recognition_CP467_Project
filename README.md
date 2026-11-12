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
