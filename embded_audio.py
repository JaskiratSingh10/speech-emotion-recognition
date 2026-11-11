#!/usr/bin/env python3
"""
Embed an audio file with google/embeddinggemma-2 (multimodal, 768-d).

Install (inside your venv):
    pip install -U librosa soundfile
    # ffmpeg is needed for mp3/m4a:  sudo apt install ffmpeg

Usage:
    python embed_audio.py my_audio.wav
    python embed_audio.py my_audio.mp3 --query "someone talking about football"
    python embed_audio.py my_audio.wav --dim 256 --out my_audio_emb.npy
"""
import argparse
import os
import tempfile

import librosa
import numpy as np
import soundfile as sf
import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "google/embeddinggemma-2"
SR = 16_000          # model expects mono, 16 kHz
CHUNK_SEC = 60     # model context holds ~327 s of audio (25 tokens/sec)


def split_to_wavs(path: str, tmpdir: str) -> list[str]:
    """Load any audio file -> mono 16 kHz -> write <=300 s WAV chunks."""
    y, _ = librosa.load(path, sr=SR, mono=True)
    print(f"Audio length: {len(y) / SR:.1f} s")
    step = SR * CHUNK_SEC
    paths = []
    for i in range(0, len(y), step):
        seg = y[i:i + step]
        if paths and len(seg) < SR:      # drop a trailing sliver under 1 s
            continue
        p = os.path.join(tmpdir, f"chunk_{len(paths):04d}.wav")
        sf.write(p, seg, SR)
        paths.append(p)
    return paths


def encode_audio(model, paths, dim):
    kw = dict(
        truncate_dim=dim,
        normalize_embeddings=True,
        batch_size=1,
    )
    try:
        return model.encode([{"audio": p} for p in paths], **kw)
    except Exception as e:  # fallback: explicit placeholder token in text
        print(f"Plain audio input failed ({e!r}); retrying with <|audio|> placeholder")
        return model.encode(
            [{"text": "<|audio|>", "audio": p} for p in paths], **kw
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--dim", type=int, default=768, choices=[128, 256, 512, 768])
    ap.add_argument("--query", help="optional text to compare with the audio")
    ap.add_argument("--out", default=None, help="save embedding to .npy")
    args = ap.parse_args()

    # Text + audio only (~570M); skip the vision encoder to save RAM.
    # float32 on CPU -- do NOT use float16 (gives NaNs with this model).
    model = SentenceTransformer(
        MODEL_ID,
        device="cpu",
        config_kwargs={"vision_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )

    with tempfile.TemporaryDirectory() as tmp:
        chunks = split_to_wavs(args.audio, tmp)
        embs = encode_audio(model, chunks, args.dim)   # (n_chunks, dim)

    # One vector per file: mean of chunk embeddings, re-normalised.
    emb = embs.mean(axis=0)
    emb = emb / np.linalg.norm(emb)
    print(f"Chunks: {len(chunks)} | embedding shape: {emb.shape}")
    print("First 8 values:", np.round(emb[:8], 4))

    if args.out:
        np.save(args.out, emb)
        print(f"Saved -> {args.out}")

    if args.query:
        # Text gets a task prefix; audio does not.
        q = model.encode(
            args.query,
            prompt_name="SearchQuery",
            truncate_dim=args.dim,
            normalize_embeddings=True,
        )
        score = model.similarity(q, emb)
        print(f'Similarity("{args.query}", audio) = {float(score):.4f}')


if __name__ == "__main__":
    main()


# python embed_audio.py my_audio.mp3 --query "someone talking about football"