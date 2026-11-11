#!/usr/bin/env python3
"""
Phase 1: Dataset Audit & Authoritative Manifest Generation.
Strictly adheres to hard acceptance gates. Uses standard library `csv`.
"""

import os
import glob
import csv
import hashlib
import soundfile as sf
from collections import Counter

# Canonical directories
WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAVDESS_DIR = os.path.join(WORKSPACE_DIR, "Ravdess")
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
MANIFEST_CSV = os.path.join(DATA_DIR, "manifest.csv")
MANIFEST_SHA256 = os.path.join(DATA_DIR, "manifest.sha256")

# Target emotion mapping for 4-class SER
# 01 = neutral, 02 = calm (merged into neutral per D2)
# 03 = happy, 04 = sad, 05 = angry
EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

EXPECTED_COUNTS = {
    "neutral": 288,
    "happy": 192,
    "sad": 192,
    "angry": 192
}

COLUMNS_ORDER = [
    "path", "filename", "modality", "lexical", "emotion_code", "class4",
    "intensity", "sentence", "take", "actor", "gender", "duration_s", "sr"
]

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 60)
    print("PHASE 1: DATASET AUDIT & AUTHORITATIVE MANIFEST GENERATION")
    print("=" * 60)

    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Recursive scan & Canonical deduplication
    print("\n[Step 1] Scanning for audio files in Ravdess...")
    all_files = glob.glob(os.path.join(RAVDESS_DIR, "**", "*.wav"), recursive=True)
    print(f"Total .wav files discovered across all directories: {len(all_files)}")

    # Group files by basename to handle nested directory duplicates safely
    files_by_basename = {}
    for fpath in all_files:
        bname = os.path.basename(fpath)
        if bname not in files_by_basename:
            files_by_basename[bname] = []
        files_by_basename[bname].append(fpath)

    print(f"Unique audio file basenames: {len(files_by_basename)}")

    # Canonical selection rule: Prefer top-level Ravdess/Actor_XX/*.wav
    canonical_files = []
    for bname, paths in sorted(files_by_basename.items()):
        # If there are duplicates, verify they are byte-identical
        if len(paths) > 1:
            h0 = compute_sha256(paths[0])
            for p in paths[1:]:
                assert compute_sha256(p) == h0, f"Hash mismatch between duplicate paths for {bname}: {paths[0]} vs {p}"
        # Pick the canonical path under Ravdess/Actor_XX/ if available
        pref = [p for p in paths if "audio_speech_actors_01-24" not in p]
        chosen = pref[0] if pref else paths[0]
        canonical_files.append(chosen)

    print(f"Canonical file list prepared: {len(canonical_files)} files.")

    # 2. Parse 7-field filename & filter
    print("\n[Step 2] Parsing 7-field filename format...")
    rows = []
    skipped_modality_lexical = 0
    skipped_emotion = 0

    for fpath in canonical_files:
        bname = os.path.basename(fpath)
        parts = bname.replace(".wav", "").split("-")
        
        # Hard check on 7 fields
        assert len(parts) == 7, f"File {bname} does not have exactly 7 fields separated by '-': {parts}"
        
        modality, lexical, emotion_code, intensity, sentence, take, actor = parts

        # Gate: Modality == 03 (audio-only) and Lexical == 01 (speech)
        if modality != "03" or lexical != "01":
            skipped_modality_lexical += 1
            continue

        # Check emotion code: 01 to 08 are valid RAVDESS codes
        assert emotion_code in [f"{i:02d}" for i in range(1, 9)], f"Invalid emotion code {emotion_code} in {bname}"

        # Target 4-class SER filter (drop 06 fearful, 07 disgust, 08 surprised)
        if emotion_code not in EMOTION_MAP:
            skipped_emotion += 1
            continue

        class4 = EMOTION_MAP[emotion_code]

        # Gender assignment: Zenodo official doc confirms Odd = Male, Even = Female
        actor_int = int(actor)
        assert 1 <= actor_int <= 24, f"Actor ID out of range [1, 24]: {actor_int}"
        gender = "male" if (actor_int % 2 != 0) else "female"

        # Relpath for portability
        rel_path = os.path.relpath(fpath, WORKSPACE_DIR)

        rows.append({
            "path": rel_path,
            "filename": bname,
            "modality": modality,
            "lexical": lexical,
            "emotion_code": emotion_code,
            "class4": class4,
            "intensity": intensity,
            "sentence": sentence,
            "take": take,
            "actor": actor,
            "gender": gender,
        })

    print(f"Rows matching modality=03, lexical=01, class4: {len(rows)}")
    print(f"Skipped non-speech files: {skipped_modality_lexical}")
    print(f"Skipped other emotions (06, 07, 08): {skipped_emotion}")

    # 3. Audio decoding & metadata extraction
    print("\n[Step 3] Decoding audio files to verify integrity, sample rate, and durations...")
    for idx, r in enumerate(rows):
        full_path = os.path.join(WORKSPACE_DIR, r["path"])
        assert os.path.exists(full_path), f"File not found: {full_path}"
        
        info = sf.info(full_path)
        r["duration_s"] = round(info.duration, 4)
        r["sr"] = info.samplerate

    # Sort deterministically by actor, sentence, emotion_code, intensity, take, filename
    rows.sort(key=lambda r: (r["actor"], r["sentence"], r["emotion_code"], r["intensity"], r["take"], r["filename"]))

    # 4. HARD ACCEPTANCE GATES (STOP IF ANY FAILS)
    print("\n" + "=" * 60)
    print("VERIFYING HARD ACCEPTANCE GATES")
    print("=" * 60)

    # Gate 1.1: Total parsed rows == 864
    print(f"[Gate 1.1] Total rows == 864 ... ", end="")
    assert len(rows) == 864, f"FAILED: Expected 864 rows, found {len(rows)}"
    print(f"PASSED (Count: {len(rows)})")

    # Gate 1.2: Per-class counts match expected distribution
    print(f"[Gate 1.2] Per-class counts match table ... ", end="")
    class_counts = Counter(r["class4"] for r in rows)
    for cls, expected in EXPECTED_COUNTS.items():
        actual = class_counts.get(cls, 0)
        assert actual == expected, f"FAILED: Class '{cls}' expected {expected}, found {actual}"
    print(f"PASSED ({dict(class_counts)})")

    # Verify breakdown of neutral: 96 neutral (code 01) + 192 calm (code 02)
    neutral_01 = sum(1 for r in rows if r["emotion_code"] == "01")
    calm_02 = sum(1 for r in rows if r["emotion_code"] == "02")
    assert neutral_01 == 96, f"Neutral (01) count mismatch: {neutral_01} != 96"
    assert calm_02 == 192, f"Calm (02) count mismatch: {calm_02} != 192"
    print(f"          (neutral breakdown: 96 neutral [01] + 192 calm [02] = 288)")

    # Gate 1.3: Actors == 24 unique (01-24), exactly 36 clips each
    print(f"[Gate 1.3] Exactly 24 unique actors, IDs 01-24 ... ", end="")
    unique_actors = sorted(list(set(r["actor"] for r in rows)))
    expected_actors = [f"{i:02d}" for i in range(1, 25)]
    assert unique_actors == expected_actors, f"FAILED: Actor list mismatch: {unique_actors}"
    actor_counts = Counter(r["actor"] for r in rows)
    for act, cnt in actor_counts.items():
        assert cnt == 36, f"FAILED: Actor {act} has {cnt} clips, expected 36"
    print("PASSED (24 actors, 36 clips each)")

    # Gate 1.4: Decodability, SR == 48000, duration within ~2.5 - 7.0 s
    print(f"[Gate 1.4] Audio decoding, SR=48000, duration in [2.5, 7.0]s ... ", end="")
    unique_srs = set(r["sr"] for r in rows)
    assert unique_srs == {48000}, f"FAILED: Unexpected sample rates: {unique_srs}"
    durations = [r["duration_s"] for r in rows]
    min_dur = min(durations)
    max_dur = max(durations)
    mean_dur = sum(durations) / len(durations)
    assert 2.5 <= min_dur, f"FAILED: Min duration {min_dur} < 2.5s"
    assert max_dur <= 7.0, f"FAILED: Max duration {max_dur} > 7.0s"
    print(f"PASSED (SR: 48000 Hz, min: {min_dur:.2f}s, max: {max_dur:.2f}s, mean: {mean_dur:.2f}s)")

    # Gate 1.5: Zero duplicate file paths / filenames; zero unparsed / null fields
    print(f"[Gate 1.5] Zero duplicate paths / filenames, zero nulls ... ", end="")
    paths = [r["path"] for r in rows]
    filenames = [r["filename"] for r in rows]
    assert len(paths) == len(set(paths)), "FAILED: Duplicate paths found"
    assert len(filenames) == len(set(filenames)), "FAILED: Duplicate filenames found"
    for r in rows:
        for k in COLUMNS_ORDER:
            assert r[k] is not None and r[k] != "", f"FAILED: Missing/null field {k} in row {r}"
    print("PASSED (All unique, zero nulls)")

    # Gate 1.6: Gender hypothesis confirmed against Zenodo README
    print(f"[Gate 1.6] Gender distribution check (12 male, 12 female) ... ", end="")
    gender_counts = Counter(r["gender"] for r in rows)
    assert gender_counts["male"] == 432 and gender_counts["female"] == 432, f"FAILED: Gender mismatch: {gender_counts}"
    print(f"PASSED (432 male clips [12 actors], 432 female clips [12 actors])")

    # 5. Save manifest and compute checksum
    print(f"\n[Step 4] Saving manifest to {MANIFEST_CSV}...")
    with open(MANIFEST_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS_ORDER)
        writer.writeheader()
        writer.writerows(rows)
    
    sha256_hash = compute_sha256(MANIFEST_CSV)
    with open(MANIFEST_SHA256, "w", encoding="utf-8") as f:
        f.write(f"{sha256_hash}  manifest.csv\n")
    
    print(f"Manifest written successfully: {len(rows)} rows.")
    print(f"Manifest SHA-256: {sha256_hash}")
    print(f"Checksum written to {MANIFEST_SHA256}")
    print("=" * 60)
    print("ALL PHASE 1 HARD ACCEPTANCE GATES PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
