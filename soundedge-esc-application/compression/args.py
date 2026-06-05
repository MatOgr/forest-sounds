import argparse
from dataclasses import dataclass


@dataclass
class OptimizationArgs:
    csv: str | None
    audio_dir: str | None
    weights: str
    stats: str
    recompute_stats: bool
    val_fold: int
    test_fold: int
    prune_amount: float
    paper_mode: bool
    prune_step: float
    prune_rounds: int
    target_params: int
    rewind_epochs: int
    kd_epochs: int
    qat_epochs: int
    lr: float
    qat_lr: float
    temperature: float
    alpha: float
    batch_size: int
    num_workers: int
    qat_backend: str
    out_dir: str
    smoke: bool
    wandb: bool
    wandb_project: str
    wandb_run: str | None
    wandb_entity: str | None


def parse_args() -> OptimizationArgs:
    p = argparse.ArgumentParser()
    p.add_argument("--csv")
    p.add_argument("--audio-dir")
    p.add_argument("--weights", default="weights/fsc22_model.pth")
    p.add_argument("--stats", default="../stats/fsc22_mel_stats.json")
    p.add_argument("--recompute-stats", action="store_true")
    p.add_argument("--val-fold", type=int, default=5)
    p.add_argument(
        "--test-fold",
        type=int,
        default=0,
        help="held-out fold for an unbiased final estimate (0=disabled). Excluded "
        "from train AND val/KD/QAT selection; every stage is scored on it once.",
    )
    p.add_argument(
        "--prune-amount",
        type=float,
        default=0.5,
        help="one-shot prune fraction (ignored in --paper-mode)",
    )
    p.add_argument(
        "--paper-mode",
        action="store_true",
        help="iterative asymmetric prune (conv1/conv2 only, conv3+KAN locked)",
    )
    p.add_argument(
        "--prune-step",
        type=float,
        default=0.1,
        help="paper-mode: fraction of CURRENT channels pruned per round",
    )
    p.add_argument(
        "--prune-rounds",
        type=int,
        default=8,
        help="paper-mode: max iterative prune+rewind rounds",
    )
    p.add_argument(
        "--target-params",
        type=int,
        default=0,
        help="paper-mode: stop early once params <= this (0=disabled)",
    )
    p.add_argument(
        "--rewind-epochs",
        type=int,
        default=2,
        help="paper-mode: KD fine-tune epochs per prune round",
    )
    p.add_argument("--kd-epochs", type=int, default=50)
    p.add_argument("--qat-epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--qat-lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=0.4, help="CE weight in KD loss")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--qat-backend", default="fbgemm")
    p.add_argument("--out-dir", default="weights")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="tiny random-data run to verify wiring (no audio needed)",
    )
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    p.add_argument("--wandb-project", default="fsc22-optimize")
    p.add_argument("--wandb-run", default=None, help="run name (default: auto)")
    p.add_argument("--wandb-entity", default=None)
    return OptimizationArgs(**vars(p.parse_args()))
