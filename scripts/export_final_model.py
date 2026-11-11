#!/usr/bin/env python3
"""
Exports the production-ready SER classifier trained on the complete Pass W dataset.
Winning Configuration: SVM linear C=0.1
Validation UAR: 60.85% (6-fold GroupKFold by actor)
"""

import os
import json
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NPZ_PATH = os.path.join(WORKSPACE_DIR, "embeddings", "pass_w.npz")
MODEL_DIR = os.path.join(WORKSPACE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ser_classifier_svm.joblib")
META_PATH = os.path.join(MODEL_DIR, "model_metadata.json")

CLASSES = ["angry", "happy", "neutral", "sad"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"Loading cached embeddings from {NPZ_PATH}...")
    data = np.load(NPZ_PATH)
    X = data["embeddings"].astype(np.float32)
    raw_labels = data["labels"]
    y = np.array([CLASS_TO_IDX[c] for c in raw_labels], dtype=np.int32)
    print(f"Loaded {len(y)} clips across classes: {CLASSES}")

    # Base SVC
    base_svc = SVC(kernel="linear", C=0.1, class_weight="balanced", random_state=42)
    # Calibrated probability output
    calibrated_clf = CalibratedClassifierCV(estimator=base_svc, cv=5)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", calibrated_clf),
    ])

    print("Fitting production classifier with probability calibration on 864 clips...")
    pipeline.fit(X, y)

    # Save artifact
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    metadata = {
        "model_name": "SVM_linear_C0.1",
        "model_path": MODEL_PATH,
        "classes": CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "encoder": "google/embeddinggemma-2",
        "embedding_dim": 768,
        "validation_metric": "UAR (Macro-Recall)",
        "validation_score": 0.6085,
        "validation_f1": 0.6015,
        "validation_protocol": "GroupKFold(n_splits=6) by actor across 3 random seeds",
        "preprocessing": {
            "sample_rate": 16000,
            "mono": True,
            "silence_trim_top_db": 25,
            "peak_normalized": True,
            "text_prompt": None
        }
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {META_PATH}")

if __name__ == "__main__":
    main()
