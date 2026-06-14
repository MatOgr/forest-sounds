"""
FSC22 training entry point for CNN_PCAw_SSRPMS_KAN.

Spec: 44.1 kHz / 5 s / zero-pad; waveform + spectrogram augment + mixup;
AdamW (lr 1e-3, wd 1e-4), batch 32, CE loss, cosine annealing, early stop (patience 100).

Run (from soundedge-esc-application/):
    python -m training.train \
        --csv ../../data/fsc22/5-fold.csv \
        --audio-dir /path/to/fsc22/audio \
        --val-fold 5 --epochs 300 --out weights/fsc22_model.pth
"""

import json
import logging
import os
import random
import resource  # peak-RSS diagnostics (Unix only)

import numpy as np
import torch
from model import CNN_PCAw_SSRPMS_KAN, CNN_PCAw_SSRPMS_KAN_DDD
from preprocessing import NormalizeMeanStd, NormalizePerChannel
from torch import nn
from torch.utils.data import DataLoader

from .args import TrainArgs

# Siblings: relative (this is the `training` package). App-root modules
# (model, preprocessing): absolute, resolved via the editable install.
from .augment import SpecAugment, WaveformAugment, mixup_batch, mixup_criterion
from .cross_validate import Splits
from .fsc22_dataset import (
    FSC22Dataset,
    MelConfig,
    build_label_map,
    compute_train_stats,
    precompute_mel_cache,
)
from .wnb import WandbLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(funcName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fsc22.train")


MODELS = {
    "CNN_PCAw_SSRPMS_KAN": CNN_PCAw_SSRPMS_KAN,
    "CNN_PCAw_SSRPMS_KAN_DDD": CNN_PCAw_SSRPMS_KAN_DDD,
}


def seed_everything(seed: int) -> torch.Generator:
    """Seed python/numpy/torch global RNGs. Returns a torch.Generator seeded
    the same way for the DataLoader's shuffle (so epoch order is reproducible).
    Per-worker augment RNG is reseeded in `_worker_init` from this base seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def _worker_init(worker_id: int) -> None:
    """Give each DataLoader worker a distinct-but-deterministic seed so the
    in-Dataset augmentations (torch global RNG) don't collide across workers
    yet stay reproducible run-to-run."""
    base = torch.initial_seed() % 2**32  # base+worker_id, set by DataLoader
    random.seed(base)
    np.random.seed(base)


def resolve_stats_path(
    stats: str, val_fold: int, test_fold: int | None, model: str, mel_cfg: MelConfig
) -> str:
    """Per-split, per-model, per-mel-config stats. Test-fold split keeps CV folds
    from reusing a mean/std computed over another fold's train data; the model
    tag keeps 1-chan and 3-chan (_DDD) stats apart; the mel tag keeps different
    sample_rate/n_fft/hop/n_mels stats from clobbering each other (the dB
    distribution changes with the mel front-end)."""
    if stats:
        return stats
    # val_fold is always in the tag: it's excluded from train, so different
    # val folds yield different train stats and must not share a cache file.
    split = f"_val{val_fold}" + (f"_test{test_fold}" if test_fold else "")
    return f"stats/fsc22_mel_stats_{model}_{mel_cfg.tag()}{split}.json"


# --------------------------------------------------------------------------- #
# Train / eval loop
# --------------------------------------------------------------------------- #
def _log_batch_diag(phase, x, y, device, use_amp) -> None:
    """One-time per-phase memory/shape sanity check on the first batch."""
    log.info(
        "[%s] first batch x=%s y=%s amp=%s",
        phase,
        tuple(x.shape),
        tuple(y.shape),
        use_amp,
    )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MB (Linux)
    log.info("[%s] peak RSS: %.0f MB", phase, rss)
    if device.type == "cuda":
        log.info(
            "[%s] VRAM allocated: %.0f MB",
            phase,
            torch.cuda.memory_allocated(device) / 1e6,
        )


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    mixup_alpha,
    train,
    scaler: torch.amp.GradScaler,
    accum_steps=1,
) -> tuple[float, float]:
    phase = "train" if train else "val"
    # LBFGS reevaluates the loss via a closure; AMP/GradScaler and grad
    # accumulation don't apply, so it takes a dedicated step path.
    needs_closure = isinstance(optimizer, torch.optim.LBFGS)
    use_amp = scaler.is_enabled() and not needs_closure
    if train and needs_closure and accum_steps != 1:
        log.warning("LBFGS: --accum-steps ignored (closure steps per batch)")
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    torch.set_grad_enabled(train)
    if train:
        optimizer.zero_grad()

    for bi, (x, y) in enumerate(loader):
        try:
            x, y = x.to(device), y.to(device)
            if bi == 0:
                _log_batch_diag(phase, x, y, device, use_amp)

            if train and needs_closure:
                # Fix the (possibly mixed) batch once so every closure
                # reevaluation optimizes the same objective.
                if mixup_alpha > 0:
                    xb, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
                else:
                    xb, y_a, y_b, lam = x, y, None, None
                cap: dict = {}

                def closure(xb=xb, y=y, y_a=y_a, y_b=y_b, lam=lam, cap=cap):
                    optimizer.zero_grad()
                    out = model(xb)
                    if y_b is None:
                        l = criterion(out, y)
                    else:
                        l = mixup_criterion(criterion, out, y_a, y_b, lam)
                    l.backward()
                    cap["out"], cap["loss"] = out, l
                    return l

                loss = optimizer.step(closure)
                logits = cap["out"]
            else:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    if train and mixup_alpha > 0:
                        x, y_a, y_b, lam = mixup_batch(x, y, mixup_alpha)
                        logits = model(x)
                        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                    else:
                        logits = model(x)
                        loss = criterion(logits, y)

                if train:
                    # Scale so accumulated grads ~= one big-batch step.
                    scaler.scale(loss / accum_steps).backward()
                    if (bi + 1) % accum_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
        except Exception:
            log.exception("[%s] FAILED at batch %d (x=%s)", phase, bi, tuple(x.shape))
            raise

    # Flush trailing grads if the last accumulation window was partial.
    if train and not needs_closure and n > 0 and len(loader) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    torch.set_grad_enabled(True)
    if n == 0:
        log.error("[%s] no samples processed (empty loader?)", phase)
        return 0.0, 0.0
    return total_loss / n, correct / n


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #
def load_or_compute_norm(
    args, train_folds, mel_cfg: MelConfig
) -> tuple[NormalizeMeanStd, NormalizePerChannel | None]:
    log.info("STAGE: normalization stats")
    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)
    # Per-channel (3-chan) stats are needed only for the "dataset" channel_norm.
    need_chan = args.model.endswith("_DDD") and args.channel_norm == "dataset"

    def _cache_ok() -> bool:
        if args.recompute_stats or not os.path.exists(args.stats):
            return False
        if need_chan:  # cached base-only stats can't serve a 3-chan request
            with open(args.stats, encoding="utf-8") as f:
                return "means" in json.load(f)
        return True

    if not _cache_ok():
        log.info("computing train stats (one-time, streams all train clips)...")
        compute_train_stats(
            args.csv,
            args.audio_dir,
            train_folds,
            args.stats,
            derivatives=need_chan,
            mel_cfg=mel_cfg,
        )
    else:
        log.info("reusing cached stats: %s", args.stats)

    with open(args.stats, encoding="utf-8") as f:
        s = json.load(f)
    log.info("stats mean=%.4f std=%.4f", s["mean"], s["std"])
    norm = NormalizeMeanStd(s["mean"], s["std"])
    chan_norm = None
    if need_chan:
        chan_norm = NormalizePerChannel(s["means"], s["stds"])
        log.info("per-channel stats means=%s stds=%s", s["means"], s["stds"])
    return norm, chan_norm


def build_loaders(
    args, splits, id_to_idx, norm, device_type, mel_cfg: MelConfig, chan_norm=None
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    log.info("STAGE: build datasets / loaders")
    DS = FSC22Dataset
    # 3-channel features only for the _DDD model; flags are inert otherwise.
    ddd_kw = {
        "derivatives": args.model.endswith("_DDD"),
        "specaug_order": args.specaug_order,
        "channel_norm": args.channel_norm,
        "chan_norm": chan_norm,
        "mel_cfg": mel_cfg,
        "cache_dir": args.mel_cache_dir or None,
    }
    # Augment lives in the dataset (CPU path). wave_aug is pre-mel; spec_aug
    # post-mel. Under --mel-cache-dir, wave_aug is bypassed unless --aug-variants.
    train_aug = {
        "wave_aug": WaveformAugment(sample_rate=mel_cfg.sample_rate),
        "spec_aug": SpecAugment(),
    }
    train_ds = DS(
        args.csv,
        args.audio_dir,
        splits.train_folds,
        id_to_idx,
        norm,
        train=True,
        # K frozen wave-aug mel copies per clip (cache path); 1 = clean only.
        # val/test omit it -> always read the clean variant.
        aug_variants=args.aug_variants,
        **train_aug,
        **ddd_kw,
    )
    val_ds = DS(
        args.csv,
        args.audio_dir,
        splits.val_folds,
        id_to_idx,
        norm,
        train=False,
        **ddd_kw,
    )

    loader_kw = {
        "num_workers": args.num_workers,
        "pin_memory": (device_type == "cuda"),  # faster host->GPU copies
        "persistent_workers": args.num_workers > 0,  # don't respawn workers each epoch
        "prefetch_factor": 4 if args.num_workers > 0 else None,
        "worker_init_fn": _worker_init if args.num_workers > 0 else None,
    }
    # Seeded generator -> reproducible shuffle order each epoch.
    shuffle_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=shuffle_gen,
        **loader_kw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw
    )

    test_loader = None
    if splits.test_fold:
        test_ds = DS(
            args.csv,
            args.audio_dir,
            [splits.test_fold],
            id_to_idx,
            norm,
            train=False,
            **ddd_kw,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw
        )

    log.info(
        "train_ds=%d val_ds=%d batches/epoch≈%d",
        len(train_ds),
        len(val_ds),
        len(train_loader),
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        log.error("empty dataset — check audio-dir/filenames/fold split")
    return train_loader, val_loader, test_loader


def _load_sophiag():
    """Return SophiaG from the `sophia-optimizer` pkg. Its `__init__.py` imports
    a nonexistent `sophiag` symbol and crashes on `import sophia`, so load the
    self-contained `sophia/sophia.py` module file directly, bypassing __init__."""
    import importlib.util

    spec = importlib.util.find_spec("sophia")  # locate pkg without executing __init__
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("sophia not installed: pip install sophia-optimizer")
    mod_path = os.path.join(spec.submodule_search_locations[0], "sophia.py")
    mspec = importlib.util.spec_from_file_location("_sophia_impl", mod_path)
    module = importlib.util.module_from_spec(mspec)
    mspec.loader.exec_module(module)
    return module.SophiaG


def _param_groups(args, model) -> list[dict]:
    """Param groups for the --kan-lr-scale split. scale==1.0 -> a single group
    (identical to model.parameters(), no behavior change). scale!=1.0 -> the
    WavKAN head gets lr*scale and weight_decay=0 (wavelet coeffs shouldn't be
    decayed toward 0), everything else (conv+fc) keeps base lr/wd. This is the
    cheap precursor to a true two-optimizer split: tune the head's lr in one
    optimizer before paying for separate .step() machinery."""
    if args.kan_lr_scale == 1.0:
        return [{"params": list(model.parameters())}]
    kan_ids = {id(p) for p in model.kan.parameters()}
    rest = [p for p in model.parameters() if id(p) not in kan_ids]
    log.info(
        "kan split: lr=%g (scale %g, wd=0) | rest lr=%g wd=%g",
        args.lr * args.kan_lr_scale,
        args.kan_lr_scale,
        args.lr,
        args.weight_decay,
    )
    return [
        {"params": rest},  # inherits optimizer-level lr / weight_decay
        {
            "params": list(model.kan.parameters()),
            "lr": args.lr * args.kan_lr_scale,
            "weight_decay": 0.0,
        },
    ]


def build_optimizer(args, model) -> torch.optim.Optimizer:
    """Select optimizer from --optimizer. AdamW/SGD/Sophia step normally; LBFGS
    is second-order and drives a closure in run_epoch (AMP/accum forced off).
    AdamW/SGD honor --kan-lr-scale (per-group lr for the WavKAN head)."""
    name = args.optimizer.lower()
    if name in ("lbfgs", "sophia") and args.kan_lr_scale != 1.0:
        log.warning("--kan-lr-scale ignored: only adamw/sgd use param groups")
    if name == "adamw":
        return torch.optim.AdamW(
            _param_groups(args, model), lr=args.lr, weight_decay=args.weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            _param_groups(args, model),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay,
        )
    params = model.parameters()
    if name == "lbfgs":
        return torch.optim.LBFGS(
            params,
            lr=args.lr,
            max_iter=args.lbfgs_max_iter,
            history_size=args.lbfgs_history,
        )
    if name == "sophia":  # TODO: reconsider this one
        SophiaG = _load_sophiag()
        return SophiaG(
            params,
            lr=args.lr,
            betas=(0.965, 0.99),
            rho=args.sophia_rho,
            weight_decay=args.weight_decay,
        )
    raise SystemExit(f"unknown --optimizer {args.optimizer!r}")


def build_training(
    args, num_classes, device, device_type
) -> tuple[
    nn.Module,
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.amp.GradScaler,
]:
    log.info("STAGE: build model (device=%s) model=%s", args.device, args.model)
    if args.model not in MODELS:
        raise SystemExit(f"unknown --model {args.model!r}; choices: {list(MODELS)}")
    model = MODELS[args.model](num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    log.info("optimizer=%s lr=%g", args.optimizer, args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # LBFGS reevaluates the loss inside a closure; GradScaler/AMP can't wrap it.
    lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    amp_enabled = args.amp and device_type == "cuda" and not lbfgs
    if args.amp and lbfgs:
        log.warning("--amp ignored: LBFGS uses a closure (AMP unsupported)")
    if args.amp and not amp_enabled and not lbfgs:
        log.warning("--amp ignored: only supported on cuda")
    scaler = torch.amp.GradScaler(device_type, enabled=amp_enabled)
    log.info("AMP %s", "ON" if amp_enabled else "off")
    return model, criterion, optimizer, scheduler, scaler


# --------------------------------------------------------------------------- #
# Training driver
# --------------------------------------------------------------------------- #
def train_model(
    args,
    model,
    loaders,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    class_names,
    wandb_logger,
) -> tuple[float, int]:
    train_loader, val_loader, _ = loaders
    best_acc, best_epoch, wait = 0.0, -1, 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        tr_loss, tr_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            args.mixup_alpha,
            True,
            accum_steps=args.accum_steps,
            scaler=scaler,
        )
        val_loss, val_acc = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            0.0,
            False,
            scaler=scaler,
        )
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(
            f"epoch {epoch + 1}/{args.epochs}  "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f}  "
            f"val_loss={val_loss:.4f} acc={val_acc:.3f}  lr={lr:.2e}"
        )
        wandb_logger.log(
            {
                "epoch": epoch + 1,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "lr": lr,
                "best_val_acc": best_acc,
            }
        )

        if val_acc > best_acc:
            best_acc, best_epoch, wait = val_acc, epoch, 0
            torch.save(model.state_dict(), args.out)
            with open(args.out + ".classes.json", "w", encoding="utf-8") as f:
                json.dump(class_names, f, indent=2)
            print(f"  * new best val_acc={best_acc:.3f} -> {args.out}")
        else:
            wait += 1
            if wait >= args.patience:
                print(f"early stop (no val gain {args.patience} epochs).")
                break

    print(f"done. best val_acc={best_acc:.3f} @ epoch {best_epoch + 1}")
    return best_acc, best_epoch


def evaluate_test(
    args,
    model,
    test_loader,
    criterion,
    optimizer,
    scaler,
    device,
    test_fold,
) -> float | None:
    """One-shot held-out test eval (unbiased: test fold never trained/selected on)."""
    if test_loader is None:
        return None
    if os.path.exists(args.out):
        model.load_state_dict(torch.load(args.out, map_location=device))
        log.info("reloaded best checkpoint for test eval: %s", args.out)
    else:
        log.warning("no checkpoint at %s; testing last-epoch weights", args.out)
    _, test_acc = run_epoch(
        model,
        test_loader,
        criterion,
        optimizer,
        device,
        0.0,
        False,
        scaler=scaler,
    )
    print(f"TEST fold={test_fold}  test_acc={test_acc:.3f}")
    return test_acc


def write_metrics(args, splits, best_acc, best_epoch, test_acc, num_classes) -> dict:
    metrics = {
        "val_fold": args.val_fold,
        "test_fold": splits.test_fold,
        "train_folds": splits.train_folds,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch + 1,
        "test_acc": test_acc,
        "num_classes": num_classes,
    }
    with open(args.out + ".metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    log.info("metrics -> %s", args.out + ".metrics.json")
    return metrics


# --------------------------------------------------------------------------- #
def main() -> None:
    args = TrainArgs.parse_args()
    log.info("args: %s", vars(args))
    seed_everything(args.seed)
    log.info("seed=%d", args.seed)

    device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    device = torch.device(args.device)
    if device_type == "cuda":
        # Autotune conv algos for fixed input shapes (5 s clips -> constant size).
        torch.backends.cudnn.benchmark = True
    if not os.path.isdir(args.audio_dir):
        log.error("audio-dir not found: %s", args.audio_dir)
    if not os.path.isfile(args.csv):
        log.error("csv not found: %s", args.csv)

    mel_cfg = MelConfig.from_args(
        args.sample_rate, args.n_fft, args.hop_length, args.n_mels
    )
    log.info("mel cfg: %s", mel_cfg)

    splits = Splits.resolve(args.val_fold, args.test_fold)
    args.stats = resolve_stats_path(
        args.stats, args.val_fold, splits.test_fold, args.model, mel_cfg
    )
    log.info("stats path: %s", args.stats)

    log.info("STAGE: build label map")
    id_to_idx, class_names = build_label_map(args.csv)
    num_classes = len(class_names)
    log.info(
        "classes=%d train_folds=%s val_fold=%s test_fold=%s",
        num_classes,
        splits.train_folds,
        args.val_fold,
        splits.test_fold,
    )

    if args.aug_variants > 1 and not args.mel_cache_dir:
        raise SystemExit("--aug-variants>1 needs --mel-cache-dir (offline cache)")

    if args.precompute_mel_cache:
        if not args.mel_cache_dir:
            raise SystemExit("--precompute-mel-cache needs --mel-cache-dir")
        log.info(
            "STAGE: precompute mel cache (clean all folds; %d-way wave-aug on "
            "train folds %s)",
            args.aug_variants,
            splits.train_folds,
        )
        precompute_mel_cache(
            args.csv,
            args.audio_dir,
            args.mel_cache_dir,
            mel_cfg,
            wave_aug=(
                WaveformAugment(sample_rate=mel_cfg.sample_rate)
                if args.aug_variants > 1
                else None
            ),
            aug_variants=args.aug_variants,
            aug_folds=splits.train_folds,
        )

    wandb_logger = WandbLogger(args)
    norm, chan_norm = load_or_compute_norm(args, splits.train_folds, mel_cfg)
    loaders = build_loaders(
        args, splits, id_to_idx, norm, device_type, mel_cfg, chan_norm=chan_norm
    )
    model, criterion, optimizer, scheduler, scaler = build_training(
        args, num_classes, device, device_type
    )

    best_acc, best_epoch = train_model(
        args,
        model,
        loaders,
        criterion,
        optimizer,
        scheduler,
        scaler,
        device,
        class_names,
        wandb_logger,
    )
    test_acc = evaluate_test(
        args,
        model,
        loaders[2],
        criterion,
        optimizer,
        scaler,
        device,
        splits.test_fold,
    )
    write_metrics(args, splits, best_acc, best_epoch, test_acc, num_classes)

    wandb_logger.summarize(
        best_val_acc=best_acc, best_epoch=best_epoch + 1, test_acc=test_acc
    )
    wandb_logger.finish()


if __name__ == "__main__":
    main()
