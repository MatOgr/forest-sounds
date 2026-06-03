import numpy as np
import librosa

SR = 22050
N_FFT = 1024
HOP = 512
N_MELS = 128
N_MFCC = 20


def mel_spectrogram(y, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
    return librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels
    )


def log_mel(y, sr=SR, **kw):
    mel = mel_spectrogram(y, sr=sr, **kw)
    return librosa.power_to_db(mel, ref=np.max)


def mfcc(y, sr=SR, n_mfcc=N_MFCC):
    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)


def extract_handcrafted(y, sr=SR, n_mfcc=N_MFCC):
    """MFCC + chroma + ZCR + spectral centroid stats → 1-D vector."""
    m = mfcc(y, sr=sr, n_mfcc=n_mfcc)
    zcr = librosa.feature.zero_crossing_rate(y)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.concatenate([
        m.mean(1), m.std(1),
        [zcr.mean(), zcr.std()],
        [cent.mean(), cent.std()],
        [bw.mean(), bw.std()],
        [roll.mean(), roll.std()],
        chroma.mean(1), chroma.std(1),
    ]).astype(np.float32)


def batch_features(audios, sr=SR, fn=extract_handcrafted):
    return np.stack([fn(y, sr=sr) for y in audios])


def normalize_spec(spec, mean=None, std=None, eps=1e-6):
    if mean is None: mean = spec.mean()
    if std is None:  std = spec.std()
    return (spec - mean) / (std + eps)
