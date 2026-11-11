# Speech Emotion Recognition with EmbeddingGemma 2

4-class Speech Emotion Recognition (`angry`, `happy`, `neutral`, `sad`) using frozen `google/embeddinggemma-2` whole-clip audio embeddings (768-d) and a calibrated linear SVM (`C=0.1`).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> `ffmpeg` is required by `librosa` for non-WAV formats (e.g., `.mp3`, `.mpeg`, `.mp4`).

## Inference

Run emotion prediction on one or more audio files using `scripts/predict.py` (uses `models/ser_classifier_svm_cremad_pilot.joblib` by default if present, otherwise `models/ser_classifier_svm.joblib`):

```bash
.venv/bin/python3 scripts/predict.py path/to/audio.wav
.venv/bin/python3 scripts/predict.py audio.wav --model models/ser_classifier_svm.joblib
.venv/bin/python3 scripts/predict.py --random-test 3
.venv/bin/python3 scripts/predict.py audio.wav --json
```

Audio preprocessing: resampled to 16 kHz mono, silence-trimmed (`top_db=25`), and peak-normalized before encoding.

## Training & Evaluation Pipeline

### RAVDESS (864 clips, 24 actors; `calm` merged into `neutral`)

```bash
.venv/bin/python3 scripts/01_build_manifest.py
.venv/bin/python3 scripts/02_extract_pass_w.py
.venv/bin/python3 scripts/04_train_classifiers.py
.venv/bin/python3 scripts/export_final_model.py
```

### CREMA-D (with ESC-50 noise, pitch-shift, and RIR augmentation)

```bash
.venv/bin/python3 scripts/01_build_cremad_manifest.py
.venv/bin/python3 scripts/02_extract_cremad_pass_w.py --pilot
.venv/bin/python3 scripts/04_train_cremad_svm.py --pilot
```

Omit `--pilot` to run on all 91 CREMA-D actors instead of the 10-actor pilot split.

## Results

| Dataset / Experiment | Classifier | UAR (%) | Macro-F1 (%) |
| :--- | :--- | :---: | :---: |
| RAVDESS (6-fold speaker-independent `GroupKFold`, 3 seeds) | `SVM_linear_C0.1` | 60.85 | 60.15 |
| CREMA-D Pilot (Joint RAVDESS + CREMA-D clean+aug train -> clean test) | `SVM_linear_C0.1` | 58.07 | 58.49 |
| CREMA-D Pilot (In-domain clean train -> clean test) | `SVM_linear_C0.1` | 57.69 | 58.17 |

## Project Layout

- `scripts/` — Manifest builders, embedding extraction (`02_extract_*.py`, `03_extract_pass_v.py`), audio augmentation (`augment.py`), training, model export, and inference (`predict.py`).
- `models/` — Exported `.joblib` SVM pipelines and JSON metadata.
- `data/` — Dataset manifests, SHA-256 checksums, noise samples, and RIRs.
- `embeddings/` — Cached `.npy` / `.npz` embeddings.
- `results/` — Classifier benchmark CSVs and evaluation reports.
- `embded_audio.py` — Standalone script for extracting `google/embeddinggemma-2` audio embeddings and computing text-audio similarity.
