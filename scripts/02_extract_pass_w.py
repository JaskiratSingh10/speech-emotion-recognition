#!/usr/bin/env python3
"""
Phase 2 - Pass W: Whole-Clip Utterance Embedding Extraction.
Model: google/embeddinggemma-2 (frozen, bfloat16, vision_config=None)
D8: mono 16 kHz, silence-trimmed (top_db=25), peak-normalised, no text prefix.
Features: 100% resumable caching per clip, hard gate validation, and sanity probe (Gate 2.2).
"""

import os
import sys
import csv
import time
import shutil
import hashlib
import tempfile
import numpy as np
import soundfile as sf
import librosa
import torch
from sentence_transformers import SentenceTransformer

# Paths
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")
EMB_DIR = os.path.join(WORKSPACE_DIR, "embeddings", "pass_w")
CLIPS_DIR = os.path.join(EMB_DIR, "clips")
CONSOLIDATED_NPY = os.path.join(EMB_DIR, "embeddings.npy")
CONSOLIDATED_NPZ = os.path.join(WORKSPACE_DIR, "embeddings", "pass_w.npz")
INDEX_CSV = os.path.join(EMB_DIR, "index.csv")
CHECKSUM_FILE = os.path.join(EMB_DIR, "checksum.sha256")
SCRATCH_TMP = os.path.join(WORKSPACE_DIR, "tmp")

# Set TMPDIR to workspace storage partition to preserve root SSD space
os.environ["TMPDIR"] = SCRATCH_TMP
os.makedirs(SCRATCH_TMP, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

TARGET_SR = 16000
DIM = 768

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def preprocess_audio(audio_path: str, tmp_out_path: str):
    """
    D8 Compliance:
    - mono 16 kHz
    - silence-trimmed (top_db=25)
    - peak-normalised to 1.0
    """
    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    
    # Fallback safety: if trimming was too aggressive (< 0.5s), keep original
    if len(y_trimmed) < int(0.5 * TARGET_SR):
        y_trimmed = y

    peak = np.max(np.abs(y_trimmed))
    y_norm = y_trimmed / (peak + 1e-9)
    sf.write(tmp_out_path, y_norm, TARGET_SR)
    return len(y_norm) / TARGET_SR

def main():
    print("=" * 65)
    print("PHASE 2 - PASS W: WHOLE-CLIP EMBEDDING EXTRACTION")
    print("=" * 65)

    assert os.path.exists(MANIFEST_CSV), f"Manifest missing: {MANIFEST_CSV}"

    torch.set_num_threads(min(4, os.cpu_count() or 4))

    # Read authoritative manifest
    rows = []
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_clips = len(rows)
    print(f"Authoritative manifest loaded: {total_clips} clips.")
    assert total_clips == 864, f"Manifest must contain exactly 864 rows, found {total_clips}"

    # Load frozen model in bfloat16
    print("\n[Step 1] Loading google/embeddinggemma-2 (frozen, bfloat16, vision_config=None)...")
    t0 = time.time()
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        device="cpu",
        config_kwargs={"vision_config": None},
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    print(f"Model loaded successfully in {time.time() - t0:.1f}s.")

    # Check for existing cached clips
    existing_clips = set(os.listdir(CLIPS_DIR))
    cached_count = sum(1 for r in rows if f"{os.path.splitext(r['filename'])[0]}.npy" in existing_clips)
    print(f"Found {cached_count}/{total_clips} already cached clips. Resuming extraction...")

    # Process clips
    print("\n[Step 2] Extracting embeddings...")
    start_time = time.time()
    
    with tempfile.TemporaryDirectory(dir=SCRATCH_TMP) as tmpdir:
        for idx, r in enumerate(rows, 1):
            stem = os.path.splitext(r["filename"])[0]
            clip_npy = os.path.join(CLIPS_DIR, f"{stem}.npy")
            
            if os.path.exists(clip_npy):
                continue
            
            full_audio_path = os.path.join(WORKSPACE_DIR, r["path"])
            tmp_wav = os.path.join(tmpdir, f"prep_{idx}.wav")
            
            # Preprocess
            proc_dur = preprocess_audio(full_audio_path, tmp_wav)
            
            # Forward pass: D8 no text prefix
            emb = model.encode(
                [{"audio": tmp_wav}],
                truncate_dim=DIM,
                normalize_embeddings=True,
                batch_size=1,
                show_progress_bar=False,
            )
            
            # Save vector as float32
            vec = emb[0].astype(np.float32)
            np.save(clip_npy, vec)

            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

            if idx % 10 == 0 or idx == total_clips:
                elapsed = time.time() - start_time
                completed = cached_count + (idx - cached_count)
                rate = completed / max(elapsed, 1e-5)
                remaining = (total_clips - completed) / max(rate, 1e-5)
                print(f"[{completed:03d}/{total_clips}] Extracted '{r['filename']}' ({proc_dur:.2f}s) | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m", flush=True)

    print(f"\nExtraction complete in {(time.time() - start_time)/60:.1f} minutes.")

    # Consolidate embeddings in manifest row order
    print("\n[Step 3] Consolidating embeddings into immutable matrix...")
    matrix = np.zeros((total_clips, DIM), dtype=np.float32)
    index_rows = []

    for i, r in enumerate(rows):
        stem = os.path.splitext(r["filename"])[0]
        clip_npy = os.path.join(CLIPS_DIR, f"{stem}.npy")
        assert os.path.exists(clip_npy), f"Missing embedding for {r['filename']}"
        vec = np.load(clip_npy)
        assert vec.shape == (DIM,), f"Unexpected shape {vec.shape} for {r['filename']}"
        matrix[i] = vec
        
        index_rows.append({
            "index": i,
            "filename": r["filename"],
            "class4": r["class4"],
            "actor": r["actor"],
            "gender": r["gender"],
            "intensity": r["intensity"],
            "sentence": r["sentence"],
            "take": r["take"],
        })

    # Re-normalize matrix in float32 to eliminate bfloat16 mantissa quantization residual (~0.0039)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)

    # Save consolidated artifacts
    np.save(CONSOLIDATED_NPY, matrix)
    print(f"Saved consolidated numpy matrix: {CONSOLIDATED_NPY} (shape {matrix.shape})")

    filenames_arr = np.array([r["filename"] for r in rows])
    labels_arr = np.array([r["class4"] for r in rows])
    actors_arr = np.array([r["actor"] for r in rows])
    np.savez_compressed(CONSOLIDATED_NPZ, embeddings=matrix, filenames=filenames_arr, labels=labels_arr, actors=actors_arr)
    print(f"Saved consolidated npz package: {CONSOLIDATED_NPZ}")

    with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Saved index registry: {INDEX_CSV}")

    # Checksum
    npy_sha = compute_sha256(CONSOLIDATED_NPY)
    npz_sha = compute_sha256(CONSOLIDATED_NPZ)
    with open(CHECKSUM_FILE, "w", encoding="utf-8") as f:
        f.write(f"{npy_sha}  embeddings.npy\n")
        f.write(f"{npz_sha}  pass_w.npz\n")
    print(f"Saved checksums to {CHECKSUM_FILE}")

    # VERIFY HARD ACCEPTANCE GATES
    print("\n" + "=" * 65)
    print("VERIFYING PASS W HARD ACCEPTANCE GATES")
    print("=" * 65)

    # Gate 2.1: Integrity Check
    print(f"[Gate 2.1] Dimension check (864, 768) ... ", end="")
    assert matrix.shape == (864, 768), f"FAILED: Shape is {matrix.shape}"
    print("PASSED")

    print(f"[Gate 2.1] NaN and Inf check ... ", end="")
    assert not np.isnan(matrix).any(), "FAILED: NaNs found in embeddings"
    assert not np.isinf(matrix).any(), "FAILED: Infs found in embeddings"
    print("PASSED")

    print(f"[Gate 2.1] L2 unit norm check ... ", end="")
    norms = np.linalg.norm(matrix, axis=1)
    norm_dev = np.abs(norms - 1.0)
    assert np.max(norm_dev) < 1e-4, f"FAILED: Max norm deviation {np.max(norm_dev)} exceeds tolerance"
    print(f"PASSED (mean norm: {np.mean(norms):.6f}, max dev: {np.max(norm_dev):.6e})")

    # Gate 2.2: Sanity Probe Baseline
    print("\n[Gate 2.2] Running sanity check baseline (6-fold GroupKFold by actor)...")
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import recall_score, f1_score

    gkf = GroupKFold(n_splits=6)
    recalls = []
    f1s = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(matrix, labels_arr, groups=actors_arr), 1):
        X_tr, y_tr = matrix[train_idx], labels_arr[train_idx]
        X_va, y_val = matrix[val_idx], labels_arr[val_idx]
        
        clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced")
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_va)
        
        uar = recall_score(y_val, preds, average="macro")
        macro_f1 = f1_score(y_val, preds, average="macro")
        recalls.append(uar)
        f1s.append(macro_f1)
        print(f"  Fold {fold}: Macro-Recall (UAR) = {uar*100:.2f}%, Macro-F1 = {macro_f1*100:.2f}%")

    mean_uar = np.mean(recalls) * 100
    mean_f1 = np.mean(f1s) * 100
    print(f"\nSanity Probe Mean UAR (Macro-Recall): {mean_uar:.2f}% (std: {np.std(recalls)*100:.2f}%)")
    print(f"Sanity Probe Mean Macro-F1: {mean_f1:.2f}%")
    print(f"Baselines: Random = 25.0%, Majority = 33.3%, Target threshold = 45.0%")

    assert mean_uar >= 45.0, f"FAILED: Sanity probe UAR {mean_uar:.2f}% below 45% gate!"
    print("\n[Gate 2.2] PASSED! Whole-clip embeddings exceed 45% sanity baseline.")
    print("=" * 65)
    print("PASS W EXECUTION COMPLETE AND VALIDATED!")
    print("=" * 65)

if __name__ == "__main__":
    main()
