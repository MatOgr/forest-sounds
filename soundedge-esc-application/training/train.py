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
import resource  # peak-RSS diagnostics (Unix only)

import torch
import torch.nn as nn
from model import CNN_PCAw_SSRPMS_KAN
from preprocessing import NormalizeMeanStd
from torch.utils.data import DataLoader

from .args import TrainArgs

# Siblings: relative (this is the `training` package). App-root modules
# (model, preprocessing): absolute, resolved via the editable install.
from .augment import SpecAugment, WaveformAugment, mixup_batch, mixup_criterion
from .cross_validate import Splits
from .fsc22_dataset import FSC22Dataset, build_label_map, compute_train_stats
from .wnb import WandbLogger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(funcName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fsc22.train")


def resolve_stats_path(stats: str, val_fold: int, test_fold: int | None) -> str:
    """Per-split stats when a test fold is held out, so CV folds never reuse a
    mean/std computed over data that lands in another fold's train set."""
    if stats:
        return stats
    if test_fold:
        return f"stats/fsc22_mel_stats_val{val_fold}_test{test_fold}.json"
    return "stats/fsc22_mel_stats.json"


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
    use_amp = scaler.is_enabled()
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
    if train and n > 0 and len(loader) % accum_steps != 0:
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
def load_or_compute_norm(args, train_folds) -> NormalizeMeanStd:
    log.info("STAGE: normalization stats")
    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)
    if args.recompute_stats or not os.path.exists(args.stats):
        log.info("computing train stats (one-time, streams all train clips)...")
        compute_train_stats(args.csv, args.audio_dir, train_folds, args.stats)
    else:
        log.info("reusing cached stats: %s", args.stats)
    with open(args.stats, encoding="utf-8") as f:
        s = json.load(f)
    log.info("stats mean=%.4f std=%.4f", s["mean"], s["std"])
    return NormalizeMeanStd(s["mean"], s["std"])


def build_loaders(
    args, splits, id_to_idx, norm, device_type
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    log.info("STAGE: build datasets / loaders")
    train_ds = FSC22Dataset(
        args.csv,
        args.audio_dir,
        splits.train_folds,
        id_to_idx,
        norm,
        wave_aug=WaveformAugment(sample_rate=44100),
        spec_aug=SpecAugment(),
        train=True,
    )
    val_ds = FSC22Dataset(
        args.csv, args.audio_dir, splits.val_folds, id_to_idx, norm, train=False
    )

    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),  # faster host->GPU copies
        persistent_workers=args.num_workers > 0,  # don't respawn workers each epoch
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **loader_kw
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw
    )

    test_loader = None
    if splits.test_fold:
        test_ds = FSC22Dataset(
            args.csv, args.audio_dir, [splits.test_fold], id_to_idx, norm, train=False
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


def build_training(
    args, num_classes, device, device_type
) -> tuple[
    nn.Module,
    nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.amp.GradScaler,
]:
    log.info("STAGE: build model (device=%s)", args.device)
    model = CNN_PCAw_SSRPMS_KAN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    amp_enabled = args.amp and device_type == "cuda"
    if args.amp and not amp_enabled:
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
    args, model, test_loader, criterion, optimizer, scaler, device, test_fold
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
        model, test_loader, criterion, optimizer, device, 0.0, False, scaler=scaler
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

    device_type = "cuda" if str(args.device).startswith("cuda") else "cpu"
    device = torch.device(args.device)
    if device_type == "cuda":
        # Autotune conv algos for fixed input shapes (5 s clips -> constant size).
        torch.backends.cudnn.benchmark = True
    if not os.path.isdir(args.audio_dir):
        log.error("audio-dir not found: %s", args.audio_dir)
    if not os.path.isfile(args.csv):
        log.error("csv not found: %s", args.csv)

    splits = Splits.resolve(args.val_fold, args.test_fold)
    args.stats = resolve_stats_path(args.stats, args.val_fold, splits.test_fold)
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

    wandb_logger = WandbLogger(args)
    norm = load_or_compute_norm(args, splits.train_folds)
    loaders = build_loaders(args, splits, id_to_idx, norm, device_type)
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
        args, model, loaders[2], criterion, optimizer, scaler, device, splits.test_fold
    )
    write_metrics(args, splits, best_acc, best_epoch, test_acc, num_classes)

    wandb_logger.summarize(
        best_val_acc=best_acc, best_epoch=best_epoch + 1, test_acc=test_acc
    )
    wandb_logger.finish()


if __name__ == "__main__":
    main()
