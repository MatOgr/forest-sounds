"""Waveform- and spectrogram-level augmentations per FSC22 training spec."""

import torch
import torch.nn as nn
import torchaudio.transforms as T


# ---------------------------------------------------------------------
# WAVEFORM-LEVEL: additive Gaussian noise, time-stretch, pitch shift, gain
# ---------------------------------------------------------------------
class WaveformAugment(nn.Module):
    """
    Stochastic waveform perturbations. Each applied independently with prob `p`.
    Operates on mono waveform [1, samples] at `sample_rate`. Output length is
    re-padded/trimmed to the original length (so downstream mel shape is fixed).
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        p: float = 0.5,
        noise_std_range=(0.001, 0.015),
        gain_db_range=(-6.0, 6.0),
        pitch_semitone_range=(-2, 2),
        stretch_range=(0.9, 1.1),
    ):
        super().__init__()
        self.sr = sample_rate
        self.p = p
        self.noise_std_range = noise_std_range
        self.gain_db_range = gain_db_range

        # Discretize pitch + stretch so transform kernels can be built ONCE and
        # cached. Building PitchShift/Resample per-sample allocates large FFT /
        # sinc buffers every call -> the RAM spike. Cache eliminates the churn.
        lo, hi = pitch_semitone_range
        self._pitch_shifters = nn.ModuleDict(
            {
                str(s): T.PitchShift(sample_rate, n_steps=s)
                for s in range(int(lo), int(hi) + 1)
                if s != 0
            }
        )
        slo, shi = stretch_range
        rates = [
            round(r, 2) for r in (slo, (slo + shi) / 2, shi) if abs(r - 1.0) > 1e-3
        ]
        # ModuleDict keys can't contain "." -> use index keys "0","1",...
        self._resamplers = nn.ModuleDict(
            {
                str(i): T.Resample(sample_rate, int(sample_rate * r))
                for i, r in enumerate(rates)
            }
        )

    def _roll(self) -> bool:
        return torch.rand(1).item() < self.p

    def _rand(self, lo, hi) -> float:
        return lo + (hi - lo) * torch.rand(1).item()

    def _choice(self, seq):
        return seq[int(torch.randint(len(seq), (1,)).item())]

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[-1]

        # Additive Gaussian noise.
        if self._roll():
            std = self._rand(*self.noise_std_range)
            wav = wav + torch.randn_like(wav) * std

        # Random gain (dB -> linear).
        if self._roll():
            gain = 10.0 ** (self._rand(*self.gain_db_range) / 20.0)
            wav = wav * gain

        # Pitch shift — pick a cached shifter (no per-call kernel build).
        if self._pitch_shifters and self._roll():
            key = self._choice(list(self._pitch_shifters.keys()))
            wav = self._pitch_shifters[key](wav)

        # Time-stretch via cached resampler (note: also shifts pitch slightly;
        # acceptable as augmentation, paired with explicit pitch shift).
        if self._resamplers and self._roll():
            key = self._choice(list(self._resamplers.keys()))
            wav = self._resamplers[key](wav)

        # Re-fix length to original (zero-pad / trim tail).
        if wav.shape[-1] < n:
            wav = torch.nn.functional.pad(wav, (0, n - wav.shape[-1]))
        elif wav.shape[-1] > n:
            wav = wav[..., :n]

        return wav


# ---------------------------------------------------------------------
# SPECTROGRAM-LEVEL: time & frequency masking (SpecAugment)
# ---------------------------------------------------------------------
class SpecAugment(nn.Module):
    """Time + frequency masking on a [.., n_mels, time] log-mel tensor."""

    def __init__(self, freq_mask=8, time_mask=40, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_masks = nn.ModuleList(
            [T.FrequencyMasking(freq_mask) for _ in range(n_freq_masks)]
        )
        self.time_masks = nn.ModuleList(
            [T.TimeMasking(time_mask) for _ in range(n_time_masks)]
        )

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        for m in self.freq_masks:
            spec = m(spec)
        for m in self.time_masks:
            spec = m(spec)
        return spec


# ---------------------------------------------------------------------
# MIXUP (on-the-fly, batch level)
# ---------------------------------------------------------------------
def mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """
    Returns (mixed_x, y_a, y_b, lam). Loss = lam*CE(out, y_a) + (1-lam)*CE(out, y_b).
    Beta(alpha, alpha) sampled via two Gammas (torch has no direct Beta sampler arg here).
    """
    if alpha <= 0:
        return x, y, y, 1.0
    g1 = torch.distributions.Gamma(alpha, 1.0).sample()
    g2 = torch.distributions.Gamma(alpha, 1.0).sample()
    lam = float(g1 / (g1 + g2))

    perm = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)
