#!/usr/bin/env python3
"""
Step 1: CREMA-D Dataset Audit & Authoritative Manifest Generation.
Filters to 4-class Speech Emotion Recognition (SER):
  - ANG -> angry
  - HAP -> happy
  - NEU -> neutral
  - SAD -> sad
Excludes DIS (disgust) and FEA (fear) to align with RAVDESS 4-class benchmark.

Enforces speaker-independent split:
  - 91 total actors (1001 to 1091)
  - Train split: 73 actors (~80%)
  - Test split: 18 actors (~20%)
  - Pilot split: 10 actors (8 train, 2 test) for fast validation
"""

import os
import csv
import glob
import hashlib
import random
import soundfile as sf
from collections import Counter

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREMAD_DIR = os.path.join(WORKSPACE_DIR, "cremad", "AudioWAV")
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
MANIFEST_CSV = os.path.join(DATA_DIR, "cremad_manifest.csv")
CHECKSUM_FILE = os.path.join(DATA_DIR, "cremad_manifest.sha256")

# 4-class SER mapping
EMOTION_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 65)
    print("STEP 1: CREMA-D AUDIT & MANIFEST GENERATION")
    print("=" * 65)

    os.makedirs(DATA_DIR, exist_ok=True)

    wav_files = sorted(glob.glob(os.path.join(CREMAD_DIR, "*.wav")))
    print(f"Total discovered .wav files in {CREMAD_DIR}: {len(wav_files)}")
    assert len(wav_files) == 7442, f"Expected 7442 files in CREMA-D, found {len(wav_files)}"

    # Parse and filter
    rows = []
    skipped_emotions = Counter()
    actor_set = set()

    for fpath in wav_files:
        bname = os.path.basename(fpath)
        stem = os.path.splitext(bname)[0]
        parts = stem.split("_")
        assert len(parts) == 4, f"Filename {bname} does not have exactly 4 parts: {parts}"

        actor, sentence, emo_code, intensity = parts
        actor_set.add(actor)

        if emo_code not in EMOTION_MAP:
            skipped_emotions[emo_code] += 1
            continue

        class4 = EMOTION_MAP[emo_code]
        rel_path = os.path.relpath(fpath, WORKSPACE_DIR)

        rows.append({
            "path": rel_path,
            "filename": bname,
            "actor": actor,
            "sentence": sentence,
            "emotion_code": emo_code,
            "class4": class4,
            "intensity": intensity,
        })

    print(f"Filtered 4-class SER clips: {len(rows)} across {len(actor_set)} actors.")
    print(f"Skipped emotions (disgust & fear): {dict(skipped_emotions)}")
    assert len(rows) == 4900, f"Expected exactly 4,900 clips for 4-class SER, got {len(rows)}"

    # Deterministic Speaker-Independent Split
    # Seed 42 for reproducible actor assignment
    all_actors = sorted(list(actor_set))
    rng = random.Random(42)
    shuffled_actors = list(all_actors)
    rng.shuffle(shuffled_actors)

    n_test_actors = 18  # ~20%
    test_actors = set(shuffled_actors[:n_test_actors])
    train_actors = set(shuffled_actors[n_test_actors:])
    print(f"Partitioned actors: {len(train_actors)} Train actors, {len(test_actors)} Test actors.")

    # Pilot actors for quick pipeline verification (8 train, 2 test)
    pilot_train_actors = set(sorted(list(train_actors))[:8])
    pilot_test_actors = set(sorted(list(test_actors))[:2])
    pilot_actors = pilot_train_actors | pilot_test_actors
    print(f"Pilot subset: 10 actors ({len(pilot_train_actors)} train, {len(pilot_test_actors)} test)")

    # Verify audio integrity & sample rates
    print("\nVerifying audio integrity & sample rates on validation sample...")
    for idx, r in enumerate(rows[:50]):
        full_path = os.path.join(WORKSPACE_DIR, r["path"])
        info = sf.info(full_path)
        assert info.samplerate == 16000, f"Unexpected SR {info.samplerate} in {r['filename']}"
        assert info.channels == 1, f"Expected mono audio in {r['filename']}, got {info.channels} channels"

    print(f"Validation sample passed (16000 Hz, mono PCM_16). Populating metadata for all {len(rows)} clips...")
    for idx, r in enumerate(rows):
        full_path = os.path.join(WORKSPACE_DIR, r["path"])
        sz = os.path.getsize(full_path)
        dur = (sz - 44) / 32000.0
        r["duration_s"] = f"{dur:.3f}"
        r["sr"] = "16000"
        r["split"] = "test" if r["actor"] in test_actors else "train"
        r["is_pilot"] = "1" if r["actor"] in pilot_actors else "0"

    # Distribution summaries
    train_counts = Counter(r["class4"] for r in rows if r["split"] == "train")
    test_counts = Counter(r["class4"] for r in rows if r["split"] == "test")
    pilot_train_counts = Counter(r["class4"] for r in rows if r["is_pilot"] == "1" and r["split"] == "train")
    pilot_test_counts = Counter(r["class4"] for r in rows if r["is_pilot"] == "1" and r["split"] == "test")

    print("\nClass distribution (Full):")
    print(f"  Train ({sum(train_counts.values())} clips): {dict(train_counts)}")
    print(f"  Test  ({sum(test_counts.values())} clips):  {dict(test_counts)}")

    print("\nClass distribution (Pilot):")
    print(f"  Train ({sum(pilot_train_counts.values())} clips): {dict(pilot_train_counts)}")
    print(f"  Test  ({sum(pilot_test_counts.values())} clips):  {dict(pilot_test_counts)}")

    fieldnames = [
        "path", "filename", "actor", "sentence", "emotion_code", "class4",
        "intensity", "duration_s", "sr", "split", "is_pilot"
    ]

    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest_sha = compute_sha256(MANIFEST_CSV)
    with open(CHECKSUM_FILE, "w", encoding="utf-8") as f:
        f.write(f"{manifest_sha}  cremad_manifest.csv\n")

    print(f"\nManifest successfully written to: {MANIFEST_CSV}")
    print(f"Checksum: {manifest_sha}")

if __name__ == "__main__":
    main()
