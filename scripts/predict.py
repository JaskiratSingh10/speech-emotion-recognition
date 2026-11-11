#!/usr/bin/env python3
"""
Single-Audio Inference & Testing Script for 4-Class Speech Emotion Recognition (SER)
Using Frozen EmbeddingGemma 2 & Calibrated Linear SVM.

Supported Emotions:
  - angry
  - happy
  - neutral (merged with calm)
  - sad

Usage:
  # Test custom audio file:
  .venv/bin/python3 scripts/predict.py path/to/your_audio.wav

  # Test multiple audio files:
  .venv/bin/python3 scripts/predict.py audio1.wav audio2.mp3 audio3.flac

  # Test on random samples from RAVDESS dataset:
  .venv/bin/python3 scripts/predict.py --random-test 3

  # Output machine-readable JSON:
  .venv/bin/python3 scripts/predict.py audio.wav --json
"""

import os
import sys
import glob
import json
import random
import argparse
import tempfile
import numpy as np
import soundfile as sf
import librosa
import torch
import joblib
from sentence_transformers import SentenceTransformer

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(WORKSPACE_DIR, "models", "ser_classifier_svm_cremad_pilot.joblib")
if not os.path.exists(DEFAULT_MODEL):
    DEFAULT_MODEL = os.path.join(WORKSPACE_DIR, "models", "ser_classifier_svm.joblib")

MODEL_PATH = DEFAULT_MODEL
META_PATH = os.path.join(WORKSPACE_DIR, "models", "model_metadata.json")
SCRATCH_TMP = os.path.join(WORKSPACE_DIR, "tmp")

os.environ["TMPDIR"] = SCRATCH_TMP
os.makedirs(SCRATCH_TMP, exist_ok=True)

TARGET_SR = 16000
CLASSES = ["angry", "happy", "neutral", "sad"]
EMOJI_MAP = {
    "angry": "😡",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢"
}

RAVDESS_CODE_MAP = {
    "01": "neutral",
    "02": "neutral",  # calm merged with neutral
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

CREMAD_CODE_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

def parse_audio_ground_truth(filename: str):
    """
    Checks if filename matches RAVDESS format (03-01-05-...) or CREMA-D format (1001_DFA_ANG_XX)
    and extracts ground truth emotion if present.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    
    # Check CREMA-D pattern
    if "_" in base:
        parts = base.split("_")
        if len(parts) == 4 and parts[2] in CREMAD_CODE_MAP:
            return {
                "dataset": "CREMA-D",
                "ground_truth": CREMAD_CODE_MAP[parts[2]],
                "actor": parts[0],
                "sentence": parts[1]
            }

    # Check RAVDESS pattern
    if "-" in base:
        parts = base.split("-")
        if len(parts) == 7 and parts[0] == "03" and parts[1] == "01":
            emo_code = parts[2]
            actor = parts[6]
            if emo_code in RAVDESS_CODE_MAP:
                return {
                    "dataset": "RAVDESS",
                    "ground_truth": RAVDESS_CODE_MAP[emo_code],
                    "actor": actor,
                    "original_code": emo_code
                }
    return None

def preprocess_audio(audio_path: str, tmp_out_path: str):
    """
    D8 Preprocessing:
      - 16 kHz mono
      - Silence trimming (top_db=25)
      - Peak-normalisation to 1.0
    """
    y, orig_sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    orig_duration = len(y) / TARGET_SR
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    if len(y_trimmed) < int(0.5 * TARGET_SR):
        y_trimmed = y

    trimmed_duration = len(y_trimmed) / TARGET_SR
    peak = np.max(np.abs(y_trimmed))
    y_norm = y_trimmed / (peak + 1e-9)
    sf.write(tmp_out_path, y_norm, TARGET_SR)

    return {
        "original_sr": orig_sr,
        "original_duration_s": round(orig_duration, 2),
        "trimmed_duration_s": round(trimmed_duration, 2),
        "samples": len(y_norm)
    }

class EmotionPredictor:
    def __init__(self, model_path=None):
        target_model = model_path or MODEL_PATH
        print(f"Loading SER classification pipeline from: {target_model}")
        if not os.path.exists(target_model):
            raise FileNotFoundError(
                f"Trained model not found at {target_model}."
            )
        self.pipeline = joblib.load(target_model)

        print("Loading frozen EmbeddingGemma 2 encoder (CPU, bfloat16)...")
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        self.encoder = SentenceTransformer(
            "google/embeddinggemma-2",
            device="cpu",
            config_kwargs={"vision_config": None},
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        print("Ready for inference.\n")

    def predict_single(self, audio_path: str):
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with tempfile.TemporaryDirectory(dir=SCRATCH_TMP) as tmpdir:
            tmp_wav = os.path.join(tmpdir, "prep_input.wav")
            audio_info = preprocess_audio(audio_path, tmp_wav)

            # Pass W: whole-clip audio forward pass without text prompt
            emb = self.encoder.encode(
                [{"audio": tmp_wav}],
                truncate_dim=768,
                normalize_embeddings=True,
                batch_size=1,
                show_progress_bar=False,
            )
            # L2 unit re-normalization
            vec = emb[0].astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            vec = vec.reshape(1, -1)

            # Predict probabilities
            probs = self.pipeline.predict_proba(vec)[0]
            pred_idx = int(np.argmax(probs))
            pred_class = CLASSES[pred_idx]
            confidence = float(probs[pred_idx])

            prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(CLASSES, probs)}

            gt_info = parse_audio_ground_truth(audio_path)

            return {
                "file": audio_path,
                "predicted_emotion": pred_class,
                "confidence": round(confidence, 4),
                "probabilities": prob_dict,
                "audio_info": audio_info,
                "ground_truth_info": gt_info
            }

def render_ascii_bar(val: float, width: int = 24) -> str:
    filled = int(round(val * width))
    empty = width - filled
    return "█" * filled + "░" * empty

def print_result_card(res: dict):
    pred = res["predicted_emotion"]
    conf = res["confidence"] * 100
    emoji = EMOJI_MAP.get(pred, "")
    audio = res["audio_info"]
    gt_info = res["ground_truth_info"]

    print("┌" + "─" * 68 + "┐")
    print(f"│ Audio File: {os.path.basename(res['file']):<54} │")
    print(f"│ Path: {res['file'][:60]:<60} │")
    print(f"│ Duration: {audio['original_duration_s']}s (Trimmed: {audio['trimmed_duration_s']}s) | Sample Rate: {audio['original_sr']} Hz   │")
    print("├" + "─" * 68 + "┤")
    print(f"│ PREDICTED EMOTION: {emoji}  {pred.upper():<12} (Confidence: {conf:5.1f}%)        │")

    if gt_info:
        gt = gt_info["ground_truth"]
        gt_emoji = EMOJI_MAP.get(gt, "")
        match_str = "MATCH ✅" if gt == pred else "MISCLASSIFIED ❌"
        ds = gt_info.get("dataset", "GT")
        act = gt_info.get("actor", "")
        print(f"│ Ground Truth ({ds}): {gt_emoji}  {gt.upper():<10} [{match_str}] (Actor {act:<3})       │")

    print("├" + "─" * 68 + "┤")
    print("│ Probability Distribution:                                          │")

    sorted_probs = sorted(res["probabilities"].items(), key=lambda x: x[1], reverse=True)
    for cls_name, p in sorted_probs:
        bar = render_ascii_bar(p, width=22)
        star = "◄ [PREDICTED]" if cls_name == pred else ""
        print(f"│   {cls_name:<8} [{bar}] {p*100:5.1f}%  {star:<18} │")

    print("└" + "─" * 68 + "┘\n")

def main():
    parser = argparse.ArgumentParser(description="Test custom audio with the 4-class SER emotion classifier.")
    parser.add_argument("audio_files", nargs="*", help="Path(s) to audio file(s) (.wav, .mp3, .flac, etc.)")
    parser.add_argument("--model", default=None, help="Path to custom trained .joblib classifier model")
    parser.add_argument("--random-test", type=int, default=0, help="Test N random clips from the RAVDESS dataset")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of human-readable cards")
    args = parser.parse_args()

    file_list = list(args.audio_files)

    if args.random_test > 0:
        dataset_clips = glob.glob(os.path.join(WORKSPACE_DIR, "Ravdess", "Actor_*", "*.wav"))
        valid_clips = [c for c in dataset_clips if parse_audio_ground_truth(c) is not None]
        if not valid_clips:
            print("No valid RAVDESS clips found in Ravdess/Actor_*/")
            sys.exit(1)
        selected = random.sample(valid_clips, min(args.random_test, len(valid_clips)))
        file_list.extend(selected)

    if not file_list:
        parser.print_help()
        print("\n[TIP] Quick test with a sample file:")
        print("  .venv/bin/python3 scripts/predict.py your_audio.wav")
        print("\n[TIP] Or specify model:")
        print("  .venv/bin/python3 scripts/predict.py your_audio.wav --model models/ser_classifier_svm_cremad_pilot.joblib")
        sys.exit(0)

    predictor = EmotionPredictor(model_path=args.model)
    all_results = []

    for audio_path in file_list:
        try:
            res = predictor.predict_single(audio_path)
            all_results.append(res)
            if not args.json:
                print_result_card(res)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    if args.json:
        print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()
