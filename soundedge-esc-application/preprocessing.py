import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import soundfile as sf

TARGET_SR = 44100
TARGET_DURATION = 5
TARGET_NUM_SAMPLES = TARGET_SR * TARGET_DURATION


class NormalizeMeanStd(nn.Module):
    def __init__(self, mean: float, std: float, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


def mel_transform_from_stats(
    stats_path: str,
    sample_rate: int = 44100,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 40,
):
    with open(stats_path, "r", encoding="utf-8") as f:
        s = json.load(f)

    return nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        ),
        torchaudio.transforms.AmplitudeToDB(),
        NormalizeMeanStd(s["mean"], s["std"]),
    )


def load_audio_with_soundfile(file_path: str):
    waveform, sr = sf.read(file_path, always_2d=True)   # [samples, channels]
    waveform = torch.tensor(waveform, dtype=torch.float32).transpose(0, 1)  # [channels, samples]
    return waveform, sr


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def resample_if_needed(waveform: torch.Tensor, orig_sr: int, target_sr: int = TARGET_SR) -> torch.Tensor:
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform


def pad_or_trim(waveform: torch.Tensor, target_num_samples: int = TARGET_NUM_SAMPLES) -> torch.Tensor:
    num_samples = waveform.shape[1]

    if num_samples > target_num_samples:
        waveform = waveform[:, :target_num_samples]
    elif num_samples < target_num_samples:
        pad_amount = target_num_samples - num_samples
        waveform = F.pad(waveform, (0, pad_amount))

    return waveform


def preprocess_audio(file_path: str, stats_path: str) -> torch.Tensor:
    waveform, sr = load_audio_with_soundfile(file_path)

    waveform = convert_to_mono(waveform)
    waveform = resample_if_needed(waveform, sr, TARGET_SR)
    waveform = pad_or_trim(waveform, TARGET_NUM_SAMPLES)

    mel_transform = mel_transform_from_stats(stats_path=stats_path)
    features = mel_transform(waveform)   # [1, 40, time]

    features = features.unsqueeze(0)     # [1, 1, 40, time]
    return features