import argparse
import typing
from dataclasses import MISSING, dataclass, field, fields
from typing import Literal

import torch

MODEL_VARIANTS = Literal[
    "CNN_PCAw_SSRPMS_KAN",
    "CNN_PCAw_SSRPMS_KAN_DDD",
]

OPTIMIZERS = Literal[
    "adamw",  # default; lr 1e-3
    "sgd",  # Nesterov momentum; lr ~1e-2
    "lbfgs",  # 2nd-order, closure-based; lr ~1e-1, no AMP/accum/mixup-determinism
    "sophia",  # SophiaG; needs `pip install sophia-optimizer`; lr ~2e-4
]


class DataclassArgs:
    """Mixin: build an ArgumentParser from a dataclass's fields and parse into
    an instance. Each `field_name` -> `--field-name`. Per-field `metadata` keys:
        help        -> argparse help text
        type        -> override the parse type (e.g. for Optional fields)
        nargs       -> argparse nargs (e.g. "+", argparse.REMAINDER)
        positional  -> True for a positional arg (no `--` flag)
    bool fields become `store_true`; their dataclass default is the flag-off value.
    Literal-typed fields become a str arg with argparse `choices` = the Literal
    values (Literal itself isn't a callable, so it can't be an argparse `type`).
    """

    @classmethod
    def parse_args(cls, argv=None):
        p = argparse.ArgumentParser()
        for f in fields(cls):
            kw: dict = {}
            if "help" in f.metadata:
                kw["help"] = f.metadata["help"]

            if f.metadata.get("positional"):
                name = f.name
                if "nargs" in f.metadata:
                    kw["nargs"] = f.metadata["nargs"]
                p.add_argument(name, **kw)
                continue

            flag = "--" + f.name.replace("_", "-")

            # bool fields -> store_true; the dataclass default supplies the flag-off value.
            if f.type is bool:
                p.add_argument(flag, action="store_true", **kw)
                continue

            ftype = f.metadata.get("type", f.type)
            # Literal -> str arg constrained by choices (Literal isn't callable).
            if typing.get_origin(ftype) is Literal:
                kw["choices"] = list(typing.get_args(ftype))
                ftype = str
            kw["type"] = ftype
            if "nargs" in f.metadata:
                kw["nargs"] = f.metadata["nargs"]
            if f.default is not MISSING:
                kw["default"] = f.default
            elif f.default_factory is not MISSING:  # type: ignore[misc]
                kw["default"] = f.default_factory()
            else:
                kw["required"] = True
            p.add_argument(flag, **kw)

        return cls(**vars(p.parse_args(argv)))


@dataclass
class TrainArgs(DataclassArgs):
    """Typed training config. CLI flags are derived from these fields:
    each `field_name` -> `--field-name`. `store_true`/`type` overrides and
    `help` live in per-field metadata. `args.stats` is mutated post-parse
    (see resolve_stats_path), so this stays non-frozen."""

    csv: str
    audio_dir: str = field(metadata={"help": "dir holding FSC22 .wav files"})
    val_fold: int = 5
    test_fold: int = field(
        default=0,
        metadata={
            "help": "held-out fold evaluated ONCE after training (0=disabled). "
            "Excluded from train AND val to keep the test estimate unbiased."
        },
    )
    epochs: int = 300
    seed: int = field(default=42, metadata={"help": "RNG seed (torch/numpy/random)"})
    model: MODEL_VARIANTS = field(
        default="CNN_PCAw_SSRPMS_KAN",
        metadata={
            "help": "CNN_PCAw_SSRPMS_KAN (1-chan) or CNN_PCAw_SSRPMS_KAN_DDD "
            "(3-chan mel + delta + delta-delta). _DDD auto-enables derivatives."
        },
    )
    specaug_order: str = field(
        default="after",
        metadata={
            "help": "_DDD only: 'after' masks the stacked 3-chan tensor, "
            "'before' masks base mel pre-derivation."
        },
    )
    channel_norm: str = field(
        default="none",
        metadata={
            "help": "_DDD only per-channel normalization: 'none' (deltas of "
            "normalized mel), 'instance' (per-sample per-chan z-score), "
            "'dataset' (per-chan train stats; computed + cached in --stats)."
        },
    )
    sample_rate: int = field(
        default=44100,
        metadata={
            "help": "target SR; clips resampled to this. num_samples "
            "= sample_rate * 5 s, so input length tracks SR automatically."
        },
    )
    n_fft: int = field(default=1024, metadata={"help": "STFT window size"})
    hop_length: int = field(default=512, metadata={"help": "STFT hop"})
    n_mels: int = field(default=40, metadata={"help": "mel bins (freq dim)"})
    mel_cache_dir: str = field(
        default="",
        metadata={
            "help": "dir for raw-dB mel disk cache (empty=off). Reused "
            "across runs with matching mel specs; subdir keyed by sr/n_fft/hop/"
            "n_mels. NOTE: enabling it bypasses waveform augment (pre-mel)."
        },
    )
    precompute_mel_cache: bool = field(
        default=False,
        metadata={
            "help": "warm the mel cache (all folds) before training, then "
            "train. Needs --mel-cache-dir."
        },
    )
    aug_variants: int = field(
        default=1,
        metadata={
            "help": "offline static wave-aug: cache K frozen mels/clip "
            "(variant 0 clean + K-1 wave-aug copies); train picks one at random per "
            "access. Keeps wave-aug under --mel-cache-dir at K-way diversity instead "
            "of bypassed. 1=off. Needs --mel-cache-dir."
        },
    )
    batch_size: int = 4  # PCAw SVD is VRAM-heavy
    accum_steps: int = 8  # 4*8 = effective batch 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: OPTIMIZERS = field(
        default="adamw",
        metadata={
            "help": "optimizer: adamw (lr~1e-3), sgd (Nesterov, lr~1e-2), "
            "lbfgs (2nd-order closure; forces AMP/accum off, lr~1e-1), sophia "
            "(SophiaG, needs sophia-optimizer pkg, lr~2e-4). Tune --lr per choice."
        },
    )
    kan_lr_scale: float = field(
        default=1.0,
        metadata={
            "help": "lr multiplier for the WavKAN head's param group "
            "(adamw/sgd only). 1.0=single group (no split). <1 trains the kan "
            "slower than conv+fc; kan also gets weight_decay=0. Sweep this before "
            "committing to a true two-optimizer split."
        },
    )
    momentum: float = field(default=0.9, metadata={"help": "SGD momentum"})
    nesterov: bool = field(
        default=False, metadata={"help": "SGD: enable Nesterov momentum"}
    )
    sophia_rho: float = field(
        default=0.01, metadata={"help": "SophiaG rho (Hessian clip threshold)"}
    )
    lbfgs_max_iter: int = field(
        default=20, metadata={"help": "LBFGS max iterations per .step()"}
    )
    lbfgs_history: int = field(default=100, metadata={"help": "LBFGS history size"})
    mixup_alpha: float = 0.2
    patience: int = 100
    amp: bool = field(default=False, metadata={"help": "mixed precision (cuda only)"})
    num_workers: int = 8
    out: str = "weights/fsc22_model.pth"
    stats: str = field(
        default="",
        metadata={
            "help": "mel mean/std cache. Empty -> auto path: per-split when "
            "--test-fold set (no cross-split leakage), else "
            "stats/fsc22_mel_stats.json."
        },
    )
    recompute_stats: bool = False
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    wandb: bool = field(default=False, metadata={"help": "log to Weights & Biases"})
    wandb_project: str = "fsc22-esc"
    wandb_run: str | None = field(
        default=None, metadata={"type": str, "help": "run name (default: auto)"}
    )
    wandb_entity: str | None = field(default=None, metadata={"type": str})


@dataclass
class SplitArgs(DataclassArgs):
    csv: str
    audio_dir: str
    out_dir: str = field(
        default="weights/cv", metadata={"help": "per-fold checkpoints + metrics"}
    )
    epochs: int = 300
    seed: int = field(default=42, metadata={"help": "RNG seed forwarded to train.py"})
    folds: list = field(
        default_factory=lambda: [1, 2, 3, 4, 5],
        metadata={
            "type": int,
            "nargs": "+",
            "help": "which folds to use as the test fold (default: all 5)",
        },
    )
    wandb: bool = field(
        default=False,
        metadata={"help": "log each fold to W&B as a separate run (testfold{N})"},
    )
    wandb_project: str = "fsc22-esc-cv"
    # Everything after a literal `--` is forwarded verbatim to train.py.
    forward: list = field(
        default_factory=list,
        metadata={"positional": True, "nargs": argparse.REMAINDER},
    )
