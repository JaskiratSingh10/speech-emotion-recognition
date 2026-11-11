#!/usr/bin/env python3
"""
Step 3: Whole-Clip Utterance Embedding Extraction on CREMA-D with Data Augmentation.
Model: google/embeddinggemma-2 (frozen, bfloat16, vision_config=None)

Extraction Rules:
  - Test set clips: Clean ONLY (Strict Data Hygiene: NO augmentation on test data).
  - Train set clips: Both Clean ({stem}.npy) and Augmented ({stem}_aug.npy) variants.
  - Fully resumable caching: each vector saved individually to disk.
  - Supports `--pilot` flag for Phase A pilot validation (10 actors).
"""

import os
import sys

# Immediate directory creation before heavy imports
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
MANIFEST_CSV = os.path.join(DATA_DIR, "cremad_manifest.csv")
EMB_BASE_DIR = os.path.join(WORKSPACE_DIR, "embeddings", "cremad_pass_w")
CLIPS_DIR = os.path.join(EMB_BASE_DIR, "clips")
SCRATCH_TMP = os.path.join(WORKSPACE_DIR, "tmp")

os.environ["TMPDIR"] = SCRATCH_TMP
os.makedirs(SCRATCH_TMP, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)

print(f"[Init] Initialized directories. CLIPS_DIR: {CLIPS_DIR}", flush=True)

import csv
import time
import argparse
import tempfile
import numpy as np
import soundfile as sf
import librosa
print("[Init] Audio libraries imported.", flush=True)

import torch
print("[Init] PyTorch imported.", flush=True)

from sentence_transformers import SentenceTransformer
print("[Init] SentenceTransformer imported.", flush=True)

# Add scripts directory to sys.path so augment can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment import AudioAugmentor
print("[Init] Augmentation engine imported.", flush=True)

TARGET_SR = 16000
DIM = 768

def preprocess_audio(audio_path: str, augmentor: AudioAugmentor = None, aug_mode: str = None) -> np.ndarray:
    """
    Load mono 16 kHz -> optionally augment -> silence trim (top_db=25) -> peak-normalise to 1.0.
    """
    y, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)

    if augmentor is not None and aug_mode is not None:
        y = augmentor.augment(y, sr=TARGET_SR, mode=aug_mode)

    # Silence trimming
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    if len(y_trimmed) < int(0.5 * TARGET_SR):
        y_trimmed = y

    peak = np.max(np.abs(y_trimmed))
    y_norm = y_trimmed / (peak + 1e-9)
    return y_norm.astype(np.float32)

def extract_clip(model, audio_arr: np.ndarray, out_npy_path: str, tmpdir: str, prefix="clip") -> None:
    """
    Write audio to tmp wav -> forward through EmbeddingGemma-2 -> save normalized float32 vector.
    """
    tmp_wav = os.path.join(tmpdir, f"{prefix}_{os.getpid()}_{time.time_ns()}.wav")
    sf.write(tmp_wav, audio_arr, TARGET_SR)

    try:
        emb = model.encode(
            [{"audio": tmp_wav}],
            truncate_dim=DIM,
            normalize_embeddings=True,
            batch_size=1,
            show_progress_bar=False,
        )
        vec = emb[0].astype(np.float32)
        np.save(out_npy_path, vec)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)

def main():
    parser = argparse.ArgumentParser(description="Extract CREMA-D embeddings with augmentation")
    parser.add_argument("--pilot", action="store_true", help="Extract pilot split only (10 actors for fast validation)")
    parser.add_argument("--max_clips", type=int, default=None, help="Optional max clips for quick dry-run test")
    args = parser.parse_args()

    print("=" * 65, flush=True)
    mode_str = "PILOT (10 Actors)" if args.pilot else "FULL DATASET (91 Actors)"
    print(f"STEP 3: CREMA-D EMBEDDING EXTRACTION - {mode_str}", flush=True)
    print("=" * 65, flush=True)

    assert os.path.exists(MANIFEST_CSV), f"Manifest not found: {MANIFEST_CSV}"

    torch.set_num_threads(min(4, os.cpu_count() or 4))

    # Read manifest
    rows = []
    with open(MANIFEST_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if args.pilot and r["is_pilot"] != "1":
                continue
            rows.append(r)

    print(f"Loaded manifest: {len(rows)} clips to process.", flush=True)
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"  Train: {len(train_rows)} clips (will produce Clean + Augmented embeddings)", flush=True)
    print(f"  Test:  {len(test_rows)} clips (Clean ONLY for strict evaluation)", flush=True)

    total_tasks = len(train_rows) * 2 + len(test_rows)
    print(f"Total target embedding vectors: {total_tasks}", flush=True)

    # Initialize augmentor
    augmentor = AudioAugmentor(seed=42)

    # Load frozen model in bfloat16
    print("\nLoading google/embeddinggemma-2 (frozen, bfloat16, vision_config=None)...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(
        "google/embeddinggemma-2",
        device="cpu",
        config_kwargs={"vision_config": None},
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    print(f"Model loaded successfully in {time.time() - t0:.1f}s.", flush=True)

    # Check cache
    existing_files = set(os.listdir(CLIPS_DIR))
    
    # Process tasks
    tasks = []
    # Test clips (clean only)
    for r in test_rows:
        stem = os.path.splitext(r["filename"])[0]
        out_p = os.path.join(CLIPS_DIR, f"{stem}.npy")
        tasks.append((r, out_p, False, None, "test_clean"))

    # Train clips (clean)
    for r in train_rows:
        stem = os.path.splitext(r["filename"])[0]
        out_p = os.path.join(CLIPS_DIR, f"{stem}.npy")
        tasks.append((r, out_p, False, None, "train_clean"))

    # Train clips (augmented)
    for r in train_rows:
        stem = os.path.splitext(r["filename"])[0]
        out_p = os.path.join(CLIPS_DIR, f"{stem}_aug.npy")
        tasks.append((r, out_p, True, "random", "train_aug"))

    if args.max_clips:
        tasks = tasks[:args.max_clips]
        print(f"Dry-run limited to first {args.max_clips} tasks.", flush=True)

    cached_count = sum(1 for _, out_p, _, _, _ in tasks if os.path.basename(out_p) in existing_files)
    print(f"Cached embeddings already present on disk: {cached_count}/{len(tasks)}", flush=True)

    start_time = time.time()
    completed = cached_count

    with tempfile.TemporaryDirectory(dir=SCRATCH_TMP) as tmpdir:
        for idx, (r, out_p, is_aug, aug_mode, task_type) in enumerate(tasks, 1):
            bname = os.path.basename(out_p)
            if bname in existing_files or os.path.exists(out_p):
                continue

            full_audio_path = os.path.join(WORKSPACE_DIR, r["path"])
            aug_engine = augmentor if is_aug else None
            audio_proc = preprocess_audio(full_audio_path, augmentor=aug_engine, aug_mode=aug_mode)

            extract_clip(model, audio_proc, out_p, tmpdir, prefix=task_type)
            existing_files.add(bname)
            completed += 1

            elapsed = time.time() - start_time
            done_this_session = completed - cached_count
            rate = done_this_session / max(elapsed, 1e-5)
            remaining = (len(tasks) - completed) / max(rate, 1e-5) if rate > 0 else 0
            print(f"[{completed:04d}/{len(tasks)}] Extracted '{bname}' ({task_type}) | "
                  f"Elapsed: {elapsed/60:.1f}m | Rate: {rate*60:.1f} clips/min | ETA: {remaining/60:.1f}m", flush=True)

    print(f"\nAll {len(tasks)} embeddings ready on disk in {(time.time() - start_time)/60:.1f}m.", flush=True)

    # Consolidate into structured .npz package
    if not args.max_clips:
        print("\nConsolidating embeddings into organized matrices...", flush=True)
        suffix = "_pilot" if args.pilot else ""
        npz_out = os.path.join(WORKSPACE_DIR, "embeddings", f"cremad_pass_w{suffix}.npz")

        def compile_partition(row_list, aug_suffix=""):
            embs = np.zeros((len(row_list), DIM), dtype=np.float32)
            fnames = []
            labels = []
            actors = []
            for i, r in enumerate(row_list):
                stem = os.path.splitext(r["filename"])[0]
                f_npy = os.path.join(CLIPS_DIR, f"{stem}{aug_suffix}.npy")
                assert os.path.exists(f_npy), f"Missing {f_npy}"
                v = np.load(f_npy)
                embs[i] = v
                fnames.append(f"{stem}{aug_suffix}")
                labels.append(r["class4"])
                actors.append(r["actor"])
            embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
            return embs, np.array(fnames), np.array(labels), np.array(actors)

        train_clean_X, train_clean_f, train_clean_y, train_clean_act = compile_partition(train_rows, aug_suffix="")
        train_aug_X, train_aug_f, train_aug_y, train_aug_act = compile_partition(train_rows, aug_suffix="_aug")
        test_clean_X, test_clean_f, test_clean_y, test_clean_act = compile_partition(test_rows, aug_suffix="")

        np.savez_compressed(
            npz_out,
            train_clean_X=train_clean_X, train_clean_y=train_clean_y,
            train_clean_actors=train_clean_act, train_clean_filenames=train_clean_f,
            train_aug_X=train_aug_X, train_aug_y=train_aug_y,
            train_aug_actors=train_aug_act, train_aug_filenames=train_aug_f,
            test_clean_X=test_clean_X, test_clean_y=test_clean_y,
            test_clean_actors=test_clean_act, test_clean_filenames=test_clean_f,
        )
        print(f"Saved consolidated package to: {npz_out}", flush=True)
        print(f"  Train Clean: {train_clean_X.shape}", flush=True)
        print(f"  Train Aug:   {train_aug_X.shape}", flush=True)
        print(f"  Test Clean:  {test_clean_X.shape}", flush=True)

if __name__ == "__main__":
    main()
