import argparse
from dataclasses import MISSING, dataclass, field, fields

import torch


class DataclassArgs:
    """Mixin: build an ArgumentParser from a dataclass's fields and parse into
    an instance. Each `field_name` -> `--field-name`. Per-field `metadata` keys:
        help        -> argparse help text
        type        -> override the parse type (e.g. for Optional fields)
        nargs       -> argparse nargs (e.g. "+", argparse.REMAINDER)
        positional  -> True for a positional arg (no `--` flag)
    bool fields become `store_true`; their dataclass default is the flag-off value.
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

            kw["type"] = f.metadata.get("type", f.type)
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
    batch_size: int = 4  # PCAw SVD is VRAM-heavy
    accum_steps: int = 8  # 4*8 = effective batch 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
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
