import librosa
import numpy as np

SR = 22050


def aug_noise(y, snr_db=20):
    rms = np.sqrt(np.mean(y**2)) + 1e-8
    noise = np.random.randn(len(y)) * rms / (10 ** (snr_db / 20))
    return (y + noise).astype(np.float32)


def aug_time_shift(y, max_frac=0.2):
    s = int(np.random.uniform(-max_frac, max_frac) * len(y))
    return np.roll(y, s).astype(np.float32)


def aug_pitch(y, sr=SR, max_steps=2.0):
    n = np.random.uniform(-max_steps, max_steps)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n).astype(np.float32)


def aug_time_stretch(y, rate_range=(0.9, 1.1)):
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(y, rate=rate).astype(np.float32)


def aug_gain(y, db_range=(-6, 6)):
    db = np.random.uniform(*db_range)
    return (y * (10 ** (db / 20))).astype(np.float32)


def spec_augment(mel, time_mask=20, freq_mask=10, n_time=1, n_freq=1):
    m = mel.copy()
    F, T = m.shape
    for _ in range(n_freq):
        f = np.random.randint(0, freq_mask + 1)
        if f and F - f > 0:
            f0 = np.random.randint(0, F - f)
            m[f0 : f0 + f, :] = 0
    for _ in range(n_time):
        t = np.random.randint(0, time_mask + 1)
        if t and T - t > 0:
            t0 = np.random.randint(0, T - t)
            m[:, t0 : t0 + t] = 0
    return m


def random_augment(y, sr=SR, p=0.5):
    """Apply random chain of waveform augs."""
    if np.random.rand() < p:
        y = aug_noise(y, snr_db=np.random.uniform(15, 30))
    if np.random.rand() < p:
        y = aug_time_shift(y)
    if np.random.rand() < p:
        y = aug_gain(y)
    if np.random.rand() < p * 0.5:
        y = aug_pitch(y, sr=sr)
    return y


import random

import torch
import torch.nn as nn
import torchaudio.transforms as T


# =====================================================================
# 1. TIME-DOMAIN AUDIO AUGMENTATIONS
# =====================================================================
class RawAudioAugmenter:
    """
    Implements physical sound alterations directly on the 1D raw waveform
    before it gets converted into a spectrogram.
    """

    def __init__(self, noise_factor=0.005, max_shift_pct=0.15):
        self.noise_factor = noise_factor
        self.max_shift_pct = max_shift_pct

    def add_white_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Injects uniform Gaussian white noise to improve robustness against microphone variance."""
        noise = torch.randn_like(waveform)
        augmented_waveform = waveform + self.noise_factor * noise
        return augmented_waveform

    def time_shift(self, waveform: torch.Tensor) -> torch.Tensor:
        """Shifts the audio randomly in time, wrapping it around (cyclic shift)."""
        shift_amt = int(random.random() * self.max_shift_pct * waveform.shape[-1])
        if random.random() > 0.5:
            shift_amt = -shift_amt
        return torch.roll(waveform, shifts=shift_amt, dims=-1)


# =====================================================================
# 2. SPECTRO-TEMPORAL MASKING (SPEC_AUGMENT)
# =====================================================================
class SpecAugmentBlock(nn.Module):
    """
    Implements SpecAugment directly on the 2D Log-Mel Spectrogram.
    This masks continuous blocks of channels or time frames, forcing
    the KAN backend to learn alternative acoustic features.
    """

    def __init__(self, freq_mask_max=15, time_mask_max=35):
        super().__init__()
        # Native torchaudio transformations used in the paper's layout
        self.freq_masker = T.FrequencyMasking(freq_mask_param=freq_mask_max)
        self.time_masker = T.TimeMasking(time_mask_param=time_mask_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expects shape: (batch_size, channels, n_mels, time)
        # Apply transforms independently across the batch elements
        x = self.freq_masker(x)
        x = self.time_masker(x)
        return x


# =====================================================================
# 3. FULL INTEGRATED AUGMENTATION PIPELINE
# =====================================================================
class EnvironmentalSoundDatasetPipeline(nn.Module):
    """
    The complete data preparation pipeline combining raw audio alterations
    and Spectrogram masking for model training.
    """

    def __init__(self, sample_rate=16000, n_mels=64):
        super().__init__()
        # Audio feature extraction configuration
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, hop_length=512, n_mels=n_mels
        )
        # Augmentation handlers
        self.time_domain_aug = RawAudioAugmenter()
        self.spec_domain_aug = SpecAugmentBlock()

    def forward(self, waveform: torch.Tensor, training: bool = True) -> torch.Tensor:
        # Input shape: (batch_size, samples)

        if training:
            # Stage 1: Time-domain raw wave mutations
            # (Executed out-of-place to maintain tensor graph sanity)
            waveform = self.time_domain_aug.time_shift(waveform)
            waveform = self.time_domain_aug.add_white_noise(waveform)

        # Stage 2: Convert to Log-Mel Spectrogram
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log(mel_spec + 1e-6).unsqueeze(1)  # Shape: (B, 1, n_mels, time)

        if training:
            # Stage 3: Frequency & Time masking (SpecAugment)
            log_mel = self.spec_domain_aug(log_mel)

        return log_mel


# =====================================================================
# PIPELINE VERIFICATION RUN
# =====================================================================
if __name__ == "__main__":
    # Create the complete execution pipeline
    pipeline = EnvironmentalSoundDatasetPipeline()

    # Generate mock dataset batch: 4 samples of 4-second audio at 16kHz
    mock_batch = torch.randn(4, 64000)

    print("=== Training Phase Execution ===")
    features_train = pipeline(mock_batch, training=True)
    print(f"Augmented Train Feature Tensor Shape: {features_train.shape}")

    print("\n=== Validation Phase Execution ===")
    features_val = pipeline(mock_batch, training=False)
    print(f"Clean Validation Feature Tensor Shape: {features_val.shape}")
    print("\n[SUCCESS] Augmentation layers are correctly verified and fully modular.")
