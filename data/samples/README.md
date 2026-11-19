# Sample landmark CSVs

Small development samples of the full landmark CSVs (the full files are 70-80 MB and live in the
shared Drive folder, not git). Same columns as `docs/SCHEMA.md`.

- `landmarks_none_sample.csv` - no preprocessing
- `landmarks_gamma_2_5_sample.csv` - gamma 2.5 (best detection rate)

Each has 100 images per letter (24 letters, 2400 rows), evenly spaced by file number through the
full set, so the two files contain the same images. Rows with `detected = 0` are included (about
15-23% of rows); drop them before training. Normal lighting only.
