#!/usr/bin/env python3
"""
Audio Augmentation Engine for Speech Emotion Recognition (SER).
Supports:
  1. Additive Background Noise Injection (from ESC-50 environmental sounds) at SNR [10, 25] dB.
  2. Pitch Shifting (±1 to ±2 semitones) via librosa.effects.pitch_shift.
  3. Room Impulse Response (RIR) acoustic reverberation (Image Source Method) with T60 in [0.2, 0.5]s.
  4. Composite augmentation combining transformations.
"""

import os
import glob
import random
import urllib.request
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
NOISE_DIR = os.path.join(DATA_DIR, "noises")
RIR_DIR = os.path.join(DATA_DIR, "rirs")

TARGET_SR = 16000

# ESC-50 noise samples for background acoustic diversity
ESC50_NOISE_URLS = [
    ("rain_1.wav", "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-17367-A-10.wav"),
    ("rain_2.wav", "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-21189-A-10.wav"),
    ("wind_1.wav", "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-137296-A-16.wav"),
    ("wind_2.wav", "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-29532-A-16.wav"),
    ("sea_waves_1.wav", "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-28135-A-11.wav"),
    ("vacuum_1.wav", "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-100210-A-36.wav"),
    ("clock_tick_1.wav", "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-21934-A-38.wav"),
    ("fire_1.wav", "https://raw.githubusercontent.com/karolpiczak/ESC-50/master/audio/1-17150-A-12.wav"),
]

def generate_synthetic_rir(sr=16000, room_dim=(5.0, 4.0, 3.0),
                           source_pos=(2.0, 2.0, 1.5), mic_pos=(3.0, 2.5, 1.5),
                           rt60=0.3, max_order=6) -> np.ndarray:
    """
    Simulate Room Impulse Response via Allen & Berkley Image Source Method.
    rt60: reverberation time in seconds [0.2 to 0.5s]
    """
    c = 343.0  # speed of sound m/s
    Lx, Ly, Lz = room_dim
    V = Lx * Ly * Lz
    S = 2 * (Lx * Ly + Ly * Lz + Lx * Lz)

    # Sabine formula absorption
    alpha = min(0.95, max(0.05, 24 * np.log(10) * V / (c * S * max(rt60, 0.05))))
    ref = np.sqrt(1.0 - alpha)

    rir_len = int(rt60 * sr)
    rir = np.zeros(rir_len, dtype=np.float32)

    xs, ys, zs = source_pos
    xr, yr, zr = mic_pos

    for nx in range(-max_order, max_order + 1):
        for ny in range(-max_order, max_order + 1):
            for nz in range(-max_order, max_order + 1):
                order = abs(nx) + abs(ny) + abs(nz)
                if order > max_order:
                    continue
                for sx in [xs, -xs]:
                    for sy in [ys, -ys]:
                        for sz in [zs, -zs]:
                            img_x = 2 * nx * Lx + sx
                            img_y = 2 * ny * Ly + sy
                            img_z = 2 * nz * Lz + sz
                            dist = np.sqrt((img_x - xr)**2 + (img_y - yr)**2 + (img_z - zr)**2)
                            delay = int(dist / c * sr)
                            if 0 < delay < rir_len and dist > 0.05:
                                amp = (ref ** order) / (4.0 * np.pi * dist)
                                rir[delay] += float(amp)

    # Direct path pulse at sample 0 if missing
    if rir[0] == 0:
        rir[0] = 1.0

    # Exponential decay shaping to simulate diffuse field
    decay = np.exp(-3.0 * np.linspace(0, rt60, rir_len) / rt60).astype(np.float32)
    rir = rir * decay
    peak = np.max(np.abs(rir))
    if peak > 0:
        rir = rir / peak
    return rir

def setup_augmentation_assets():
    """
    Download background noises and synthesize RIR profiles if not present.
    """
    os.makedirs(NOISE_DIR, exist_ok=True)
    os.makedirs(RIR_DIR, exist_ok=True)

    # 1. Download ESC-50 noises
    for name, url in ESC50_NOISE_URLS:
        out_path = os.path.join(NOISE_DIR, name)
        if not os.path.exists(out_path):
            try:
                tmp_path = out_path + ".tmp"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    with open(tmp_path, "wb") as f:
                        f.write(resp.read())
                # Resample to 16kHz mono and peak-normalize
                y, _ = librosa.load(tmp_path, sr=TARGET_SR, mono=True)
                y = y / (np.max(np.abs(y)) + 1e-9)
                sf.write(out_path, y.astype(np.float32), TARGET_SR)
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                print(f"  [Downloaded noise] {name} ({len(y)/TARGET_SR:.1f}s)")
            except Exception as e:
                print(f"  [WARNING] Could not download {name}: {e}")

    # Fallback synthetic pink/ambient noise if network failed
    downloaded_noises = glob.glob(os.path.join(NOISE_DIR, "*.wav"))
    if not downloaded_noises:
        print("  Generating fallback synthetic ambient noises...")
        for i, color in enumerate(["pink", "brown", "white"]):
            duration = 10.0
            n_samples = int(duration * TARGET_SR)
            white = np.random.randn(n_samples)
            if color == "pink":
                # 1/f noise filter approximation
                b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
                a = [1, -2.494956002, 2.017265875, -0.522189400]
                noise = signal.lfilter(b, a, white)
            elif color == "brown":
                noise = np.cumsum(white)
            else:
                noise = white
            noise = noise / (np.max(np.abs(noise)) + 1e-9)
            sf.write(os.path.join(NOISE_DIR, f"{color}_noise.wav"), noise.astype(np.float32), TARGET_SR)

    # 2. Synthesize RIR profiles
    rir_files = glob.glob(os.path.join(RIR_DIR, "*.wav"))
    if len(rir_files) < 6:
        print("  Synthesizing room impulse response (RIR) profiles...")
        configs = [
            ("small_office.wav", (4.0, 3.5, 2.8), 0.22),
            ("meeting_room.wav", (6.0, 5.0, 3.0), 0.30),
            ("living_room.wav", (5.5, 4.5, 2.9), 0.35),
            ("classroom.wav", (8.0, 7.0, 3.2), 0.42),
            ("reflective_hall.wav", (9.0, 8.0, 3.5), 0.48),
            ("compact_booth.wav", (3.0, 2.5, 2.5), 0.18),
        ]
        for name, dim, rt in configs:
            out_p = os.path.join(RIR_DIR, name)
            if not os.path.exists(out_p):
                rir = generate_synthetic_rir(sr=TARGET_SR, room_dim=dim, rt60=rt)
                sf.write(out_p, rir, TARGET_SR)
                print(f"  [Synthesized RIR] {name} (RT60={rt}s)")

# Augmentation Operations
def add_background_noise(audio: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add background noise to audio at target SNR in dB.
    """
    if len(noise) < len(audio):
        tile_count = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, tile_count)
    
    # Pick random offset within noise
    max_offset = max(0, len(noise) - len(audio))
    offset = random.randint(0, max_offset) if max_offset > 0 else 0
    noise_slice = noise[offset:offset + len(audio)]

    p_clean = np.mean(audio ** 2) + 1e-12
    p_noise = np.mean(noise_slice ** 2) + 1e-12

    target_noise_p = p_clean / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_p / p_noise)

    noisy_audio = audio + scale * noise_slice
    # Safe clip
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    return noisy_audio.astype(np.float32)

def pitch_shift_audio(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Pitch shift audio by semitones in [-2, -1, 1, 2].
    """
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
    shifted = np.clip(shifted, -1.0, 1.0)
    return shifted.astype(np.float32)

def apply_reverberation(audio: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """
    Apply acoustic room reverberation via fast FFT convolution with RIR.
    """
    reverbed = signal.fftconvolve(audio, rir, mode="full")[:len(audio)]
    peak = np.max(np.abs(reverbed))
    if peak > 0:
        # Match original RMS energy
        rms_orig = np.sqrt(np.mean(audio ** 2) + 1e-12)
        rms_rev = np.sqrt(np.mean(reverbed ** 2) + 1e-12)
        reverbed = reverbed * (rms_orig / rms_rev)
    reverbed = np.clip(reverbed, -1.0, 1.0)
    return reverbed.astype(np.float32)

class AudioAugmentor:
    """
    Manager for audio augmentations: selects random transformations or composite.
    """
    def __init__(self, seed=42):
        setup_augmentation_assets()
        self.noise_files = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))
        self.rir_files = sorted(glob.glob(os.path.join(RIR_DIR, "*.wav")))
        self.rng = random.Random(seed)

        # Cache noises and RIRs in memory (only ~10MB total)
        self.noises = [sf.read(f)[0].astype(np.float32) for f in self.noise_files]
        self.rirs = [sf.read(f)[0].astype(np.float32) for f in self.rir_files]
        print(f"AudioAugmentor initialized with {len(self.noises)} noise sources & {len(self.rirs)} RIR profiles.")

    def augment(self, audio: np.ndarray, sr=16000, mode="random") -> np.ndarray:
        """
        mode:
          - 'noise': inject noise at SNR 10-25 dB
          - 'pitch': pitch shift ±1 or ±2 semitones
          - 'reverb': convolve with room impulse response
          - 'random': pick one of noise, pitch, reverb, or composite
        """
        if mode == "random":
            mode = self.rng.choice(["noise", "pitch", "reverb", "composite"])

        out = np.copy(audio)

        if mode == "noise":
            noise = self.rng.choice(self.noises)
            snr = self.rng.uniform(10.0, 25.0)
            out = add_background_noise(out, noise, snr)

        elif mode == "pitch":
            steps = self.rng.choice([-2, -1, 1, 2])
            out = pitch_shift_audio(out, sr, steps)

        elif mode == "reverb":
            rir = self.rng.choice(self.rirs)
            out = apply_reverberation(out, rir)

        elif mode == "composite":
            # Pitch shift + Noise or Reverb + Noise
            if self.rng.random() > 0.5:
                steps = self.rng.choice([-1.5, -1, 1, 1.5])
                out = pitch_shift_audio(out, sr, steps)
            else:
                rir = self.rng.choice(self.rirs)
                out = apply_reverberation(out, rir)
            
            noise = self.rng.choice(self.noises)
            snr = self.rng.uniform(12.0, 25.0)
            out = add_background_noise(out, noise, snr)

        # Final peak normalization
        peak = np.max(np.abs(out))
        if peak > 0:
            out = out / peak
        return out.astype(np.float32)

if __name__ == "__main__":
    print("Testing Audio Augmentation Module...")
    augmentor = AudioAugmentor(seed=42)
    # Generate test sine sweep
    t = np.linspace(0, 2.0, 32000, endpoint=False)
    test_wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    for mode in ["noise", "pitch", "reverb", "composite"]:
        aug = augmentor.augment(test_wave, sr=TARGET_SR, mode=mode)
        print(f"Mode {mode:9s}: shape={aug.shape}, peak={np.max(np.abs(aug)):.4f}, RMS={np.sqrt(np.mean(aug**2)):.4f}")
    print("All augmentation operations successfully verified.")
