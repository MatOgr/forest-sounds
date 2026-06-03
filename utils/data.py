import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing_extensions import override

try:
    import torch
    from torch.utils.data import Dataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    Dataset = object

from .augment import random_augment, spec_augment
from .features import log_mel, normalize_spec
from .io import load_fixed


def encode_labels(labels):
    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, le


def make_splits(paths, labels, test_size=0.2, val_size=0.1, seed=42):
    """Stratified train/val/test split."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        paths, labels, test_size=test_size, stratify=labels, random_state=seed
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=seed
    )
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


class AudioDataset(Dataset):
    """PyTorch dataset → log-mel tensor + label."""

    def __init__(
        self, paths, labels, sr=22050, dur=4.0, augment=False, return_waveform=False
    ):
        if not _HAS_TORCH:
            raise ImportError("torch required for AudioDataset")
        self.paths = list(paths)
        self.labels = np.asarray(labels)
        self.sr = sr
        self.dur = dur
        self.augment = augment
        self.return_waveform = return_waveform

    def __len__(self):
        return len(self.paths)

    @override
    def __getitem__(self, index):
        y = load_fixed(self.paths[index], sr=self.sr, dur=self.dur)
        if self.augment:
            y = random_augment(y, sr=self.sr)
        if self.return_waveform:
            x = torch.from_numpy(y).float()
        else:
            mel = log_mel(y, sr=self.sr)
            if self.augment:
                mel = spec_augment(mel)
            mel = normalize_spec(mel)
            x = torch.from_numpy(mel).unsqueeze(0).float()  # (1, F, T)
        return x, int(self.labels[index])
