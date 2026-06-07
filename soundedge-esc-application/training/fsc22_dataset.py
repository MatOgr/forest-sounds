"""FSC22 dataset + mel pipeline + train-set normalization stats."""

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torchaudio.functional as AF
import torchaudio.transforms as TT

# App-root module, resolved via the editable install.
from preprocessing import (
    TARGET_DURATION,
    TARGET_NUM_SAMPLES,
    TARGET_SR,
    NormalizeMeanStd,
    NormalizePerChannel,
    convert_to_mono,
    load_audio_with_soundfile,
    resample_if_needed,
)
from torch.utils.data import Dataset
from typing_extensions import override

log = logging.getLogger("fsc22.data")

# Mel front-end defaults (spec: STFT 1024 / hop 512, 40 mel bins, dB).
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40


@dataclass(frozen=True)
class MelConfig:
    """Mel front-end + clip-length config, threaded from CLI to dataset/stats.
    Defaults reproduce the original module-constant behavior. `num_samples`
    follows sample_rate so clip duration stays fixed at TARGET_DURATION s."""

    sample_rate: int = TARGET_SR
    n_fft: int = N_FFT
    hop_length: int = HOP_LENGTH
    n_mels: int = N_MELS
    num_samples: int = TARGET_NUM_SAMPLES

    @classmethod
    def from_args(cls, sample_rate, n_fft, hop_length, n_mels) -> "MelConfig":
        return cls(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            num_samples=sample_rate * TARGET_DURATION,
        )

    def tag(self) -> str:
        """Filename-safe tag for stats caching (distinct config != shared cache)."""
        return f"sr{self.sample_rate}_fft{self.n_fft}_hop{self.hop_length}_mel{self.n_mels}"


def add_derivatives(feat: torch.Tensor, win_length: int = 5) -> torch.Tensor:
    """Per-sample delta + delta-delta stacked as channels.
    feat [1, F, T] -> [3, F, T]. Deltas computed along the time axis."""
    delta = AF.compute_deltas(feat, win_length=win_length)
    delta_delta = AF.compute_deltas(delta, win_length=win_length)
    return torch.cat([feat, delta, delta_delta], dim=0)


def per_channel_standardize(feat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero-mean / unit-std per channel over (F, T). Deltas have a different
    scale than the base mel; this puts all 3 channels on a comparable range."""
    mean = feat.mean(dim=(-2, -1), keepdim=True)
    std = feat.std(dim=(-2, -1), keepdim=True)
    return (feat - mean) / (std + eps)


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


def build_mel_db(cfg: MelConfig = MelConfig()):
    """MelSpectrogram -> AmplitudeToDB (no normalization). For stats + features."""
    return nn.Sequential(
        TT.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
        ),
        TT.AmplitudeToDB(),
    )


def _read_rows(csv_path: str):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- #
# Raw-dB mel disk cache (reused across runs with the same MelConfig)
# --------------------------------------------------------------------------- #
def mel_cache_subdir(cache_dir: str, cfg: MelConfig) -> str:
    """Per-config cache subdir: cache_dir/<sr..mel tag>/. Distinct mel specs
    land in distinct dirs, so a cache is only ever reused by a run whose
    front-end (sample_rate/n_fft/hop/n_mels) matches bit-for-bit."""
    return os.path.join(cache_dir, cfg.tag())


def ensure_mel_cache(cache_dir: str, cfg: MelConfig) -> str:
    """Create the per-config cache dir and write/validate its meta stamp.
    Guards against a tag collision silently serving incompatible features."""
    sub = mel_cache_subdir(cache_dir, cfg)
    os.makedirs(sub, exist_ok=True)
    meta_path = os.path.join(sub, "_mel_meta.json")
    meta = asdict(cfg)
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            existing = json.load(f)
        if existing != meta:
            raise RuntimeError(
                f"mel cache config mismatch in {sub}: cached={existing} "
                f"requested={meta}. Delete the dir or use a fresh --mel-cache-dir."
            )
    else:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return sub


def _cache_path(cache_subdir: str, fname: str, variant: int = 0) -> str:
    # fname is a bare clip name (e.g. "1_10101.wav"); flatten any separators.
    # variant 0 = clean mel; variant k>0 = the k-th offline wave-aug copy.
    key = fname.replace(os.sep, "__")
    if variant:
        key += f"__aug{variant}"
    return os.path.join(cache_subdir, key + ".pt")


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
        derivatives: bool = False,
        specaug_order: str = "after",
        channel_norm: str = "none",
        chan_norm: NormalizePerChannel | None = None,
        mel_cfg: MelConfig = MelConfig(),
        cache_dir: str | None = None,
        aug_variants: int = 1,
    ):
        folds = {str(f) for f in folds}
        self.rows = [r for r in _read_rows(csv_path) if r["fold"] in folds]
        self.audio_dir = audio_dir
        self.id_to_idx = id_to_idx
        self.mel_cfg = mel_cfg
        self.mel_db = build_mel_db(mel_cfg)
        self.norm = norm
        self.wave_aug = wave_aug
        # Raw-dB mel disk cache. When set, the expensive load+resample+mel is
        # done once per clip and reused every epoch / across runs. Caching is
        # pre-norm/pre-specaug, so those stay dynamic.
        #
        # wave_aug is pre-mel, so a single cached mel can't carry a per-epoch
        # random waveform. Two modes:
        #   aug_variants <= 1 -> cache the CLEAN mel only; wave_aug bypassed.
        #   aug_variants  = K -> cache K frozen mels per clip (variant 0 clean +
        #                        K-1 offline wave-aug copies). Train picks one at
        #                        random per access -> K-way wave-aug diversity
        #                        with zero runtime aug cost. spec_aug stays live.
        self.aug_variants = max(1, int(aug_variants))
        self.cache_subdir = ensure_mel_cache(cache_dir, mel_cfg) if cache_dir else None
        if self.cache_subdir and train and wave_aug is not None and self.aug_variants <= 1:
            log.warning(
                "mel cache ON (aug_variants=1) -> waveform augment BYPASSED "
                "(pre-mel, incompatible with cached mel). SpecAugment still active. "
                "Set --aug-variants K>1 to keep wave-aug as K frozen copies."
            )
        self.spec_aug = spec_aug
        self.train = train
        # 3-channel (mel + delta + delta-delta) mode for CNN_..._KAN_DDD.
        self.derivatives = derivatives
        # "after"  -> mask the stacked 3-chan tensor (same mask all channels)
        # "before" -> mask base mel, then derive (deltas see the masked input)
        self.specaug_order = specaug_order
        # "none"     -> legacy: base norm, deltas of normalized mel, no per-chan
        # "instance" -> per-sample per-channel zero-mean/unit-std
        # "dataset"  -> derive on raw dB, normalize each chan by train stats
        self.channel_norm = channel_norm
        self.chan_norm = chan_norm  # NormalizePerChannel, used iff "dataset"

    def __len__(self):
        return len(self.rows)

    def _load_wave(self, fname: str) -> torch.Tensor:
        path = os.path.join(self.audio_dir, fname)
        if not os.path.isfile(path):
            log.error("missing audio file: %s", path)
        wav, sr = load_audio_with_soundfile(path)
        wav = convert_to_mono(wav)
        wav = resample_if_needed(wav, sr, self.mel_cfg.sample_rate)
        wav = pad_only(wav, self.mel_cfg.num_samples)
        return wav  # [1, samples]

    def _mel_raw(self, fname: str) -> torch.Tensor:
        """Raw-dB mel [1, n_mels, T]. Uses the disk cache when enabled, else
        computes (and populates the cache).

        Cache, aug_variants=1 : variant 0 = CLEAN mel (deterministic, reusable).
        Cache, aug_variants=K : train picks variant k in [0,K) per access. k=0 is
            clean; k>0 is a frozen wave-aug copy (computed + cached lazily on miss
            with a fresh wave_aug pass). Eval (train=False) always reads k=0."""
        if self.cache_subdir is not None:
            k = 0
            if self.train and self.aug_variants > 1:
                k = int(torch.randint(self.aug_variants, (1,)).item())
            cpath = _cache_path(self.cache_subdir, fname, k)
            if os.path.isfile(cpath):
                return torch.load(cpath)
            wav = self._load_wave(fname)
            if k > 0 and self.wave_aug is not None:
                wav = self.wave_aug(wav)  # frozen aug copy
            feat = self.mel_db(wav)
            # Atomic write: tmp + replace so a killed worker can't leave a
            # truncated file that a later run would load as valid.
            tmp = cpath + f".tmp{os.getpid()}"
            torch.save(feat, tmp)
            os.replace(tmp, cpath)
            return feat

        wav = self._load_wave(fname)
        if self.train and self.wave_aug is not None:
            wav = self.wave_aug(wav)
        return self.mel_db(wav)

    @override
    def __getitem__(self, index):
        r = self.rows[index]
        fname = r["filename"]
        try:
            label = self.id_to_idx[int(r["Class ID"])]
            # Preprocessing has no learnable params -> no grad needed here.
            # no_grad also avoids building an autograd graph in DataLoader workers.
            with torch.no_grad():
                feat = self._mel_raw(fname)  # [1, n_mels, T] raw dB (cached)

                def _specaug(f):
                    if self.train and self.spec_aug is not None:
                        return self.spec_aug(f)
                    return f

                if not self.derivatives:
                    feat = self.norm(feat)
                    feat = _specaug(feat)
                elif self.channel_norm == "dataset":
                    # Derive on raw dB; normalize each channel by train stats.
                    if self.specaug_order == "before":
                        feat = _specaug(feat)
                    feat = add_derivatives(feat)  # [3, 40, T] raw
                    feat = self.chan_norm(feat)
                    if self.specaug_order == "after":
                        feat = _specaug(feat)
                else:
                    # Legacy: base norm, deltas of normalized mel.
                    feat = self.norm(feat)
                    if self.specaug_order == "before":
                        feat = _specaug(feat)
                    feat = add_derivatives(feat)  # [3, 40, T]
                    if self.channel_norm == "instance":
                        feat = per_channel_standardize(feat)
                    if self.specaug_order == "after":
                        feat = _specaug(feat)

            return feat, label
        except Exception:
            # Pinpoints the offending row/file in DataLoader worker tracebacks.
            log.exception("__getitem__ failed: idx=%d file=%s row=%s", index, fname, r)
            raise


def precompute_mel_cache(
    csv_path,
    audio_dir,
    cache_dir: str,
    mel_cfg: MelConfig = MelConfig(),
    folds=None,
    wave_aug=None,
    aug_variants: int = 1,
    aug_folds=None,
) -> int:
    """Warm the raw-dB mel cache for every clip (all folds unless `folds` given),
    single pass. Skips clips already cached, so it's resumable and idempotent.

    Every clip gets variant 0 (clean mel). Clips whose fold is in `aug_folds`
    additionally get variants 1..aug_variants-1, each a fresh `wave_aug` pass
    (offline static augmentation). `aug_folds` should be the TRAIN folds only —
    val/test must stay clean. No-op extra variants if wave_aug is None.

    Returns the number of mel tensors written this call (clean + aug)."""
    sub = ensure_mel_cache(cache_dir, mel_cfg)
    mel_db = build_mel_db(mel_cfg)
    rows = _read_rows(csv_path)
    if folds is not None:
        keep = {str(f) for f in folds}
        rows = [r for r in rows if r["fold"] in keep]
    aug_keep = {str(f) for f in aug_folds} if aug_folds is not None else None
    n_aug = max(1, int(aug_variants)) if wave_aug is not None else 1

    written = 0
    for k, r in enumerate(rows):
        fname = r["filename"]
        # variant 0 always; aug variants only for clips in aug_folds.
        do_aug = aug_keep is None or r["fold"] in aug_keep
        nvar = n_aug if do_aug else 1
        # Load + base mel once; reuse the clean waveform for each aug variant.
        clean_wav = None
        for v in range(nvar):
            cpath = _cache_path(sub, fname, v)
            if os.path.isfile(cpath):
                continue
            path = os.path.join(audio_dir, fname)
            try:
                if clean_wav is None:
                    wav, sr = load_audio_with_soundfile(path)
                    wav = convert_to_mono(wav)
                    wav = resample_if_needed(wav, sr, mel_cfg.sample_rate)
                    clean_wav = pad_only(wav, mel_cfg.num_samples)
                src = clean_wav if v == 0 else wave_aug(clean_wav)
                feat = mel_db(src)
                tmp = cpath + f".tmp{os.getpid()}"
                torch.save(feat, tmp)
                os.replace(tmp, cpath)
                written += 1
            except Exception:
                log.exception("mel cache: skipping unreadable file %s", path)
                break
        if (k + 1) % 200 == 0:
            log.info("mel cache %d/%d (written=%d)", k + 1, len(rows), written)
    log.info(
        "mel cache ready: %s (wrote %d tensors, aug_variants=%d)", sub, written, n_aug
    )
    return written


def compute_train_stats(
    csv_path,
    audio_dir,
    train_folds,
    stats_out: str,
    derivatives=False,
    mel_cfg: MelConfig = MelConfig(),
):
    """Streams train folds, computes mel-dB mean/std, writes JSON.

    derivatives=True -> also accumulate per-channel stats for the raw-dB
    delta and delta-delta channels (matching the "dataset" channel_norm path),
    written as `means`/`stds` length-3 arrays alongside the base `mean`/`std`."""
    folds = {str(f) for f in train_folds}
    rows = [r for r in _read_rows(csv_path) if r["fold"] in folds]
    mel_db = build_mel_db(mel_cfg)

    nch = 3 if derivatives else 1
    total = [0.0] * nch
    total_sq = [0.0] * nch
    count = 0
    skipped = 0
    for k, r in enumerate(rows):
        path = os.path.join(audio_dir, r["filename"])
        try:
            wav, sr = load_audio_with_soundfile(path)
            wav = convert_to_mono(wav)
            wav = resample_if_needed(wav, sr, mel_cfg.sample_rate)
            wav = pad_only(wav, mel_cfg.num_samples)
            feat = mel_db(wav)  # [1, F, T]
            if derivatives:
                feat = add_derivatives(feat)  # [3, F, T] raw dB
            for c in range(nch):
                ch = feat[c]
                total[c] += ch.sum().item()
                total_sq[c] += (ch**2).sum().item()
            count += feat[0].numel()
        except Exception:
            skipped += 1
            log.exception("stats: skipping unreadable file %s", path)
        if (k + 1) % 100 == 0:
            log.info("stats %d/%d (skipped=%d)", k + 1, len(rows), skipped)
    if count == 0:
        raise RuntimeError("stats: no readable clips — check --audio-dir / filenames")

    means = [t / count for t in total]
    stds = [max(tsq / count - m**2, 1e-12) ** 0.5 for tsq, m in zip(total_sq, means)]
    out = {"mean": means[0], "std": stds[0]}
    if derivatives:
        out["means"] = means
        out["stds"] = stds
    with open(stats_out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if derivatives:
        print(f"-> train stats (3-chan): means={means} stds={stds} -> {stats_out}")
    else:
        print(f"-> train stats: mean={means[0]:.4f} std={stds[0]:.4f} -> {stats_out}")
    return means[0], stds[0]
