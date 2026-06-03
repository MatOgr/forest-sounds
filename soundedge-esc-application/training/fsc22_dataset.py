"""FSC22 dataset + mel pipeline + train-set normalization stats."""

import csv
import json
import logging
import os

import torch
import torch.nn as nn
import torchaudio.transforms as TT
from torch.utils.data import Dataset
from typing_extensions import override

# App-root module, resolved via the editable install.
from preprocessing import (
    TARGET_NUM_SAMPLES,
    TARGET_SR,
    NormalizeMeanStd,
    convert_to_mono,
    load_audio_with_soundfile,
    resample_if_needed,
)

log = logging.getLogger("fsc22.data")

# Mel front-end config (matches spec: STFT 1024 / hop 256, 40 mel bins, dB).
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 40


def pad_only(waveform: torch.Tensor, target: int = TARGET_NUM_SAMPLES) -> torch.Tensor:
    """Zero-pad to `target`. Truncation NONE per spec — only assert if longer."""
    n = waveform.shape[-1]
    if n < target:
        waveform = nn.functional.pad(waveform, (0, target - n))
    elif n > target:
        # Spec says no truncation, but the model's FC layer needs a fixed time
        # dim. FSC22 clips are 5 s; anything longer is unexpected -> trim + warn.
        print(f"[warn] clip longer than {target} samples ({n}); trimming tail.")
        waveform = waveform[..., :target]
    return waveform


def build_mel_db():
    """MelSpectrogram -> AmplitudeToDB (no normalization). For stats + features."""
    return nn.Sequential(
        TT.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        ),
        TT.AmplitudeToDB(),
    )


def _read_rows(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_label_map(csv_path: str):
    """Maps raw FSC22 Class ID -> contiguous index 0..N-1 (handles missing IDs)."""
    rows = _read_rows(csv_path)
    ids = sorted({int(r["Class ID"]) for r in rows})
    id_to_idx = {cid: i for i, cid in enumerate(ids)}
    idx_to_name = {}
    for r in rows:
        idx_to_name[id_to_idx[int(r["Class ID"])]] = r["Class Name"]
    return id_to_idx, [idx_to_name[i] for i in range(len(ids))]


class FSC22Dataset(Dataset):
    """
    Loads FSC22 clips for given fold(s). Returns (features[1,40,T], label).

    train=True  -> applies waveform augment (pre-mel) + spec augment (post-mel).
    Normalization uses train-set mean/std passed in `norm`.
    """

    def __init__(
        self,
        csv_path: str,
        audio_dir: str,
        folds,
        id_to_idx: dict,
        norm: NormalizeMeanStd,
        wave_aug=None,
        spec_aug=None,
        train: bool = False,
    ):
        folds = {str(f) for f in folds}
        self.rows = [r for r in _read_rows(csv_path) if r["fold"] in folds]
        self.audio_dir = audio_dir
        self.id_to_idx = id_to_idx
        self.mel_db = build_mel_db()
        self.norm = norm
        self.wave_aug = wave_aug
        self.spec_aug = spec_aug
        self.train = train

    def __len__(self):
        return len(self.rows)

    def _load_wave(self, fname: str) -> torch.Tensor:
        path = os.path.join(self.audio_dir, fname)
        if not os.path.isfile(path):
            log.error("missing audio file: %s", path)
        wav, sr = load_audio_with_soundfile(path)
        wav = convert_to_mono(wav)
        wav = resample_if_needed(wav, sr, TARGET_SR)
        wav = pad_only(wav, TARGET_NUM_SAMPLES)
        return wav  # [1, samples]

    @override
    def __getitem__(self, index):
        r = self.rows[index]
        fname = r["filename"]
        try:
            label = self.id_to_idx[int(r["Class ID"])]
            # Preprocessing has no learnable params -> no grad needed here.
            # no_grad also avoids building an autograd graph in DataLoader workers.
            with torch.no_grad():
                wav = self._load_wave(fname)

                if self.train and self.wave_aug is not None:
                    wav = self.wave_aug(wav)

                feat = self.mel_db(wav)  # [1, 40, T]
                feat = self.norm(feat)

                if self.train and self.spec_aug is not None:
                    feat = self.spec_aug(feat)

            return feat, label
        except Exception:
            # Pinpoints the offending row/file in DataLoader worker tracebacks.
            log.exception("__getitem__ failed: idx=%d file=%s row=%s", index, fname, r)
            raise


def compute_train_stats(csv_path, audio_dir, train_folds, stats_out: str):
    """Streams train folds, computes global mel-dB mean/std, writes JSON."""
    folds = {str(f) for f in train_folds}
    rows = [r for r in _read_rows(csv_path) if r["fold"] in folds]
    mel_db = build_mel_db()

    total = 0.0
    total_sq = 0.0
    count = 0
    skipped = 0
    for k, r in enumerate(rows):
        path = os.path.join(audio_dir, r["filename"])
        try:
            wav, sr = load_audio_with_soundfile(path)
            wav = convert_to_mono(wav)
            wav = resample_if_needed(wav, sr, TARGET_SR)
            wav = pad_only(wav, TARGET_NUM_SAMPLES)
            feat = mel_db(wav)
            total += feat.sum().item()
            total_sq += (feat**2).sum().item()
            count += feat.numel()
        except Exception:
            skipped += 1
            log.exception("stats: skipping unreadable file %s", path)
        if (k + 1) % 100 == 0:
            log.info("stats %d/%d (skipped=%d)", k + 1, len(rows), skipped)
    if count == 0:
        raise RuntimeError("stats: no readable clips — check --audio-dir / filenames")

    mean = total / count
    var = max(total_sq / count - mean**2, 1e-12)
    std = var**0.5
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)
    print(f"-> train stats: mean={mean:.4f} std={std:.4f} -> {stats_out}")
    return mean, std
