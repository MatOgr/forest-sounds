import glob
import os

import librosa
import numpy as np
import pandas as pd
import torch

SR = 22050
DUR = 4.0
AUDIO_EXTS = (".wav", ".flac", ".ogg", ".mp3", ".m4a")


def list_audio(root, exts=AUDIO_EXTS):
    print(f"Walking through: {root}")
    paths = []
    for ext in exts:
        paths.extend(
            found_paths := glob.glob(os.path.join(f"{root}/*{ext}"), recursive=True)
        )
        print(f"Found {len(found_paths)} {ext} files")
    return sorted(paths)


def build_metadata(root, label_from="parent"):
    """Scan dir → DataFrame(path, label, duration, sr)."""
    rows = []
    for p in list_audio(root):
        if label_from == "parent":
            label = os.path.basename(os.path.dirname(p))
        else:
            label = None
        try:
            dur = librosa.get_duration(path=p)
            sr = librosa.get_samplerate(p)
        except librosa.ParameterError:
            dur, sr = np.nan, np.nan
        rows.append({"path": p, "label": label, "duration": dur, "sr": sr})
    return pd.DataFrame(rows)


def load_fixed(path, sr=SR, dur=DUR, offset=0.0):
    """Load audio → fixed-length np.array (pad or trim)."""
    y, _ = librosa.load(path, sr=sr, offset=offset, duration=dur)
    target = int(sr * dur)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    return y[:target].astype(np.float32)


def normalize(y, eps=1e-8):
    return y / (np.max(np.abs(y)) + eps)


def squeeze_stereo_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Ensures the input waveform is 1D (mono).
    If it is stereo/multichannel, it averages the channels.

    Shape input:  (channels, samples) or (samples,)
    Shape output: (samples,)
    """
    # If it's already a flat 1D tensor, it's already mono
    if waveform.dim() == 1:
        return waveform

    # If shape is (channels, samples)
    if waveform.shape[0] > 1:
        # Average across the channel dimension (dim 0)
        waveform = torch.mean(waveform, dim=0)
    else:
        # It's (1, samples), just remove the unneeded channel dim
        waveform = waveform.squeeze(0)

    return waveform
