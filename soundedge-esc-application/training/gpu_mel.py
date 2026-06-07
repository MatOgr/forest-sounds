"""GPU-side mel front-end.

Moves the per-clip mel pipeline (MelSpectrogram -> AmplitudeToDB -> norm ->
derivatives -> SpecAugment) off the CPU DataLoader workers and onto the GPU,
applied once per *batch*. Workers then only decode/resample/pad the waveform,
which removes the CPU mel bottleneck that starves the GPU.

Two pieces:
  - FSC22WaveformDataset: returns clean padded waveform [1, N] + label.
  - GPUMelFrontend:        [B, 1, N] waveform batch -> [B, C, F, T] features.

The feature math mirrors FSC22Dataset.__getitem__ exactly (same branches for
1-chan / _DDD / channel_norm / specaug_order), just vectorized over the batch
and run on-device. Augmentation randomness is per-batch here (not per-sample),
consistent with how mixup already operates.
"""

import logging

import torch
import torch.nn as nn
import torchaudio.functional as AF
from typing_extensions import override

from .fsc22_dataset import (
    FSC22Dataset,
    MelConfig,
    build_mel_db,
    per_channel_standardize,
)

log = logging.getLogger("fsc22.data")


class FSC22WaveformDataset(FSC22Dataset):
    """Waveform-only variant for the GPU mel path: workers do load + mono +
    resample + pad, then hand the raw waveform to GPUMelFrontend. No mel, no
    augment, no mel cache here (the GPU front-end owns all of that)."""

    @override
    def __getitem__(self, index):
        r = self.rows[index]
        fname = r["filename"]
        try:
            label = self.id_to_idx[int(r["Class ID"])]
            with torch.no_grad():
                wav = self._load_wave(fname)  # [1, num_samples], clean
            return wav, label
        except Exception:
            log.exception("__getitem__ failed: idx=%d file=%s row=%s", index, fname, r)
            raise


class GPUMelFrontend(nn.Module):
    """Batched waveform -> features, on whatever device the module lives on.

    Mirrors FSC22Dataset's feature branches. `wave_aug` / `spec_aug` are applied
    only when `forward(..., train=True)`. Submodules (mel, norm, chan_norm,
    augments) move with `.to(device)`, so call `.to(device)` once after build.
    """

    def __init__(
        self,
        mel_cfg: MelConfig,
        norm: nn.Module,
        derivatives: bool = False,
        specaug_order: str = "after",
        channel_norm: str = "none",
        chan_norm: nn.Module | None = None,
        wave_aug: nn.Module | None = None,
        spec_aug: nn.Module | None = None,
    ):
        super().__init__()
        self.mel_db = build_mel_db(mel_cfg)
        self.norm = norm
        self.derivatives = derivatives
        self.specaug_order = specaug_order
        self.channel_norm = channel_norm
        self.chan_norm = chan_norm
        self.wave_aug = wave_aug
        self.spec_aug = spec_aug

    @staticmethod
    def _add_derivatives(feat: torch.Tensor, win_length: int = 5) -> torch.Tensor:
        """Batched delta + delta-delta stacked on the channel axis.
        [B, 1, F, T] -> [B, 3, F, T]. compute_deltas acts on the time axis,
        treating the leading dims as batch."""
        delta = AF.compute_deltas(feat, win_length=win_length)
        delta_delta = AF.compute_deltas(delta, win_length=win_length)
        return torch.cat([feat, delta, delta_delta], dim=1)

    def forward(self, wav: torch.Tensor, train: bool = False) -> torch.Tensor:
        # wav: [B, 1, num_samples]
        if train and self.wave_aug is not None:
            wav = self.wave_aug(wav)

        feat = self.mel_db(wav)  # [B, 1, F, T] raw dB

        def _specaug(f):
            if train and self.spec_aug is not None:
                return self.spec_aug(f)
            return f

        if not self.derivatives:
            feat = self.norm(feat)
            feat = _specaug(feat)
        elif self.channel_norm == "dataset":
            # Derive on raw dB; normalize each channel by train stats.
            if self.specaug_order == "before":
                feat = _specaug(feat)
            feat = self._add_derivatives(feat)  # [B, 3, F, T] raw
            assert self.chan_norm is not None
            feat = self.chan_norm(feat)
            if self.specaug_order == "after":
                feat = _specaug(feat)
        else:
            # Legacy: base norm, deltas of normalized mel.
            feat = self.norm(feat)
            if self.specaug_order == "before":
                feat = _specaug(feat)
            feat = self._add_derivatives(feat)  # [B, 3, F, T]
            if self.channel_norm == "instance":
                feat = per_channel_standardize(feat)
            if self.specaug_order == "after":
                feat = _specaug(feat)

        return feat
