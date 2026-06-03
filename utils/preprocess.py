from pathlib import Path

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from typing_extensions import override


def precalculate_mel_dataset(audio_dir, output_dir, target_sr=16000, n_mels=64):
    """
    Scans a directory of audio files, converts them to Log-Mel Spectrograms,
    and writes raw PyTorch tensors to disk.
    """
    audio_path = Path(audio_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Define our static frontend transformation
    mel_transform = T.MelSpectrogram(
        sample_rate=target_sr, n_fft=1024, hop_length=512, n_mels=n_mels
    )

    print(f"--- Starting Pre-calculation from {audio_path} ---")

    # Supported audio extensions
    extensions = {".wav", ".mp3", ".flac", ".ogg"}

    for file in audio_path.rglob("*"):
        if file.suffix.lower() in extensions:
            # 1. Load file safely
            waveform, sr = torchaudio.load(str(file))

            # 2. Downmix Stereo to Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 3. Resample to standard target sample rate if needed
            if sr != target_sr:
                resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)

            # 4. Extract Log-Mel Features
            with torch.no_grad():
                mel_spec = mel_transform(waveform)
                log_mel = torch.log(mel_spec + 1e-6).squeeze(0)  # Shape: (n_mels, time)

            # 5. Save directly as a PyTorch Binary Tensor
            relative_path = file.relative_to(audio_path)
            save_file = out_path / relative_path.with_suffix(".pt")
            save_file.parent.mkdir(parents=True, exist_ok=True)

            torch.save(log_mel, save_file)

    print(f"--- Pre-calculation Complete! Features stored in: {out_path} ---")


# Example Usage:
# precalculate_mel_dataset("data/raw_wavs", "data/precomputed_mels")


class OnTheFlyAugmentedDataset(Dataset):
    """
    Loads pre-calculated Log-Mel tensors from disk and applies
    dynamic frequency and time masking in real-time.
    """

    def __init__(self, precomputed_dir, file_list, labels, is_training=True):
        self.precomputed_dir = Path(precomputed_dir)
        self.file_list = file_list  # List of relative paths to the .pt files
        self.labels = labels  # Integers matching the classes
        self.is_training = is_training

        # On-the-fly spectro-temporal maskers (Highly CPU efficient)
        self.freq_masker = T.FrequencyMasking(freq_mask_param=15)
        self.time_masker = T.TimeMasking(time_mask_param=35)

    def __len__(self):
        return len(self.file_list)

    @override
    def __getitem__(self, index):
        # 1. High-speed binary tensor load from disk
        tensor_path = self.precomputed_dir / self.file_list[index]
        log_mel = torch.load(tensor_path)  # Shape: (n_mels, time)

        # Add a channel dimension required for CNN inputs -> (1, n_mels, time)
        log_mel = log_mel.unsqueeze(0)

        # 2. Dynamic Augmentations (Only applied during the training phase)
        if self.is_training:
            log_mel = self.freq_masker(log_mel)
            log_mel = self.time_masker(log_mel)

        label = torch.tensor(self.labels[index], dtype=torch.long)

        return log_mel, label


# =====================================================================
# VERIFICATION PIPELINE RUN
# =====================================================================
if __name__ == "__main__":
    import tempfile

    # Setup mock files for validation testing
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create 4 dummy pre-calculated spectrograms (64 mel bands, 125 frames)
        mock_files = ["sample_0.pt", "sample_1.pt", "sample_2.pt", "sample_3.pt"]
        mock_labels = [12, 45, 3, 21]

        for name in mock_files:
            torch.save(torch.randn(64, 125), tmp_path / name)

        # Instantiate the Dataset in Training Mode
        train_dataset = OnTheFlyAugmentedDataset(
            precomputed_dir=tmp_path,
            file_list=mock_files,
            labels=mock_labels,
            is_training=True,
        )

        # Create standard training dataloader
        train_loader = DataLoader(
            train_dataset, batch_size=2, shuffle=True, num_workers=2
        )

        print("=== Simulating Training Loop Batches ===")
        for batch_idx, (features, targets) in enumerate(train_loader):
            # Shape output will be: (BatchSize, Channels=1, n_mels=64, Time=125)
            print(
                f"Batch {batch_idx} -> Feature Matrix Shape: {features.shape} | Targets: {targets}"
            )

        print("\n[SUCCESS] The hybrid structure is verified and ready for integration.")
