# Landmark CSV schema (Kalen -> Pranav / Mustafa)

One CSV per preprocessing variant in `data/landmarks/`, e.g. `landmarks_clahe.csv`.
Landmarks are RAW MediaPipe output - normalization happens in the next stage.

| column | meaning |
|---|---|
| image_path | path relative to data/raw (kept so splits can be made later) |
| label | letter A-Y (no J/Z) |
| variant | none / gaussian / median / gamma / clahe |
| detected | 1 if MediaPipe found a hand, else 0 (row kept, landmarks empty) |
| handedness | Left / Right as reported by MediaPipe |
| handedness_score | confidence of handedness |
| x0,y0,z0 ... x20,y20,z20 | 21 landmarks, MediaPipe image-normalized coords |

Open question for the group: the Kaggle set has no person IDs, so how do we split by person?
