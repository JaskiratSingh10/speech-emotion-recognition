#!/usr/bin/env python3
"""
Phase 2 - Pass V: Sliding-Window Frame Embedding Extraction.
Window size: 2.5 s (40,000 samples @ 16 kHz)
Hop size: 1.25 s (20,000 samples @ 16 kHz, 50% overlap)
Stores raw [k, 768] tensors per clip in embeddings/pass_v/raw/
Strictly run after Pass W clears Gate 2.2.
"""

import os
import sys
import csv
import time
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
PASS_W_DIR = os.path.join(WORKSPACE_DIR, "embeddings", "pass_w")
PASS_W_NPY = os.path.join(PASS_W_DIR, "embeddings.npy")
EMB_V_DIR = os.path.join(WORKSPACE_DIR, "embeddings", "pass_v")
RAW_V_DIR = os.path.join(EMB_V_DIR, "raw")
REGISTRY_CSV = os.path.join(EMB_V_DIR, "registry.csv")
CHECKSUM_FILE = os.path.join(EMB_V_DIR, "checksum.sha256")
SCRATCH_TMP = os.path.join(WORKSPACE_DIR, "tmp")

os.environ["TMPDIR"] = SCRATCH_TMP
os.makedirs(SCRATCH_TMP, exist_ok=True)
os.makedirs(RAW_V_DIR, exist_ok=True)

TARGET_SR = 16000
DIM = 768
WIN_SEC = 2.5
HOP_SEC = 1.25
WIN_SAMPLES = int(WIN_SEC * TARGET_SR)  # 40,000
HOP_SAMPLES = int(HOP_SEC * TARGET_SR)  # 20,000

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def extract_sliding_windows(y: np.ndarray):
    """
    Extract sliding windows: 2.5 s window, 1.25 s hop.
    Returns: list of (window_samples, start_time_s, end_time_s)
    """
    n_samples = len(y)
    windows = []
    
    if n_samples <= WIN_SAMPLES:
        # If shorter than window, pad with zeros or reflect to 2.5 s
        padded = np.zeros(WIN_SAMPLES, dtype=y.dtype)
        padded[:n_samples] = y
        windows.append((padded, 0.0, round(n_samples / TARGET_SR, 4)))
        return windows

    # Slide window
    start = 0
    while start + WIN_SAMPLES <= n_samples:
        seg = y[start : start + WIN_SAMPLES]
        t_start = round(start / TARGET_SR, 4)
        t_end = round((start + WIN_SAMPLES) / TARGET_SR, 4)
        windows.append((seg, t_start, t_end))
        start += HOP_SAMPLES

    # If trailing tail remainder is >= 0.5 * WIN_SAMPLES (1.25s) and not covered
    if n_samples - start >= HOP_SAMPLES:
        final_seg = y[n_samples - WIN_SAMPLES : n_samples]
        t_start = round((n_samples - WIN_SAMPLES) / TARGET_SR, 4)
        t_end = round(n_samples / TARGET_SR, 4)
        windows.append((final_seg, t_start, t_end))

    return windows

def main():
    print("=" * 65)
    print("PHASE 2 - PASS V: SLIDING-WINDOW EMBEDDING EXTRACTION")
    print("=" * 65)

    # Enforce gate: Pass W must exist and be verified
    if not os.path.exists(PASS_W_NPY):
        print(f"ERROR: Pass W embeddings not found at {PASS_W_NPY}!")
        print("Pass V can only be executed after Pass W clears Gate 2.2.")
        sys.exit(1)

    torch.set_num_threads(min(4, os.cpu_count() or 4))

    # Read authoritative manifest
    rows = []
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_clips = len(rows)
    print(f"Authoritative manifest loaded: {total_clips} clips.")

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
    existing_raw = set(os.listdir(RAW_V_DIR))
    cached_count = sum(1 for r in rows if f"{os.path.splitext(r['filename'])[0]}.npy" in existing_raw)
    print(f"Found {cached_count}/{total_clips} already cached clips. Resuming Pass V extraction...")

    registry_entries = []
    start_time = time.time()

    print("\n[Step 2] Extracting sliding-window embeddings [k, 768] per clip...")
    with tempfile.TemporaryDirectory(dir=SCRATCH_TMP) as tmpdir:
        for idx, r in enumerate(rows, 1):
            stem = os.path.splitext(r["filename"])[0]
            clip_npy = os.path.join(RAW_V_DIR, f"{stem}.npy")

            if os.path.exists(clip_npy):
                # Load metadata for registry
                arr = np.load(clip_npy)
                k = arr.shape[0]
                registry_entries.append({
                    "filename": r["filename"],
                    "class4": r["class4"],
                    "actor": r["actor"],
                    "window_count_k": k,
                })
                continue

            full_audio_path = os.path.join(WORKSPACE_DIR, r["path"])

            # 1. Preprocessing (D8)
            y, _ = librosa.load(full_audio_path, sr=TARGET_SR, mono=True)
            y_trimmed, _ = librosa.effects.trim(y, top_db=25)
            if len(y_trimmed) < int(0.5 * TARGET_SR):
                y_trimmed = y
            peak = np.max(np.abs(y_trimmed))
            y_norm = y_trimmed / (peak + 1e-9)

            # 2. Extract windows
            win_tuples = extract_sliding_windows(y_norm)
            k = len(win_tuples)

            # 3. Encode windows
            tmp_files = []
            for w_idx, (w_samples, _, _) in enumerate(win_tuples):
                tmp_w_path = os.path.join(tmpdir, f"win_{idx}_{w_idx}.wav")
                sf.write(tmp_w_path, w_samples, TARGET_SR)
                tmp_files.append(tmp_w_path)

            embs = model.encode(
                [{"audio": p} for p in tmp_files],
                truncate_dim=DIM,
                normalize_embeddings=True,
                batch_size=min(k, 4),
                show_progress_bar=False,
            )

            # Clean up window temp files
            for p in tmp_files:
                if os.path.exists(p):
                    os.remove(p)

            # Store raw [k, 768] array as float32
            k_tensor = embs.astype(np.float32)
            assert k_tensor.shape == (k, DIM), f"Shape mismatch: {k_tensor.shape} != ({k}, {DIM})"
            np.save(clip_npy, k_tensor)

            registry_entries.append({
                "filename": r["filename"],
                "class4": r["class4"],
                "actor": r["actor"],
                "window_count_k": k,
            })

            if idx % 10 == 0 or idx == total_clips:
                elapsed = time.time() - start_time
                completed = cached_count + (idx - cached_count)
                rate = completed / max(elapsed, 1e-5)
                remaining = (total_clips - completed) / max(rate, 1e-5)
                print(f"[{completed:03d}/{total_clips}] Extracted '{r['filename']}' (k={k} windows) | "
                      f"Elapsed: {elapsed/60:.1f}m | ETA: {remaining/60:.1f}m", flush=True)

    print(f"\nPass V extraction complete in {(time.time() - start_time)/60:.1f} minutes.")

    # Save registry
    print(f"\n[Step 3] Writing Pass V registry to {REGISTRY_CSV}...")
    with open(REGISTRY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class4", "actor", "window_count_k"])
        writer.writeheader()
        writer.writerows(registry_entries)

    # Compute checksum of registry
    reg_sha = compute_sha256(REGISTRY_CSV)
    with open(CHECKSUM_FILE, "w", encoding="utf-8") as f:
        f.write(f"{reg_sha}  registry.csv\n")
    print(f"Saved registry and checksum.")

    # Hard Acceptance Gate 2.3
    print("\n" + "=" * 65)
    print("VERIFYING PASS V HARD ACCEPTANCE GATES")
    print("=" * 65)

    raw_files = [f for f in os.listdir(RAW_V_DIR) if f.endswith(".npy")]
    print(f"[Gate 2.3] Exactly 864 raw window tensor files ... ", end="")
    assert len(raw_files) == 864, f"FAILED: Expected 864 files, found {len(raw_files)}"
    print(f"PASSED ({len(raw_files)} files)")

    print(f"[Gate 2.3] Verifying all tensors [k, 768] (no NaNs, k >= 1) ... ", end="")
    total_windows = 0
    for fn in raw_files:
        arr = np.load(os.path.join(RAW_V_DIR, fn))
        assert arr.ndim == 2 and arr.shape[1] == DIM, f"FAILED: Invalid shape {arr.shape} in {fn}"
        assert arr.shape[0] >= 1, f"FAILED: k < 1 in {fn}"
        assert not np.isnan(arr).any(), f"FAILED: NaNs in {fn}"
        assert not np.isinf(arr).any(), f"FAILED: Infs in {fn}"
        total_windows += arr.shape[0]
    print(f"PASSED (Total windows: {total_windows}, avg {total_windows/864:.2f} per clip)")
    print("=" * 65)
    print("PASS V EXECUTION COMPLETE AND VALIDATED!")
    print("=" * 65)

if __name__ == "__main__":
    main()
