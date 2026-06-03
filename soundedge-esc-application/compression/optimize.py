"""
Post-training optimization for CNN_PCAw_SSRPMS_KAN (FSC22).

Pipeline:  prune  ->  KD recover  ->  QAT fine-tune  ->  convert int8

Why this order:
  apply_structural_pruning() rebuilds `fc` with RANDOM weights (flatten dim
  changes after channel pruning), so the pruned net is broken until retrained.
  Knowledge distillation from the full trained teacher recovers accuracy, then
  QAT fine-tunes the (conv3 + fc) int8 blocks before the final convert.

Run (from soundedge-esc-application/):
    python -m compression.optimize \
        --csv ../data/fsc22/5-fold.csv \
        --audio-dir /path/to/fsc22/audio \
        --weights weights/fsc22_model.pth \
        --val-fold 5 \
        --prune-amount 0.5 \
        --kd-epochs 50 --qat-epochs 15

Quick wiring check (no data needed):
    python -m compression.optimize --smoke
"""

import argparse
import copy
import json
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Siblings: relative (this is the `compression` package). App-root modules
# (model, preprocessing): absolute, resolved via the editable install.
from .pruning import apply_asymmetric_pruning, apply_structural_pruning
from .qat import convert_qat_model, prepare_qat_model
from model import CNN_PCAw_SSRPMS_KAN
from preprocessing import NormalizeMeanStd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(funcName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fsc22.optimize")

# Mel frontend produces (1, 40, 862) for a 5 s / 44.1 kHz clip (hop 256).
EXAMPLE_INPUT = (1, 1, 40, 862)


class KnowledgeDistillationLoss(nn.Module):
    """KL over softened logits + CE on hard targets (self-contained, no utils dep)."""

    def __init__(self, temperature=3.0, alpha=0.4):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits, teacher_logits, targets):
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        kd = self.kl_loss(soft_student, soft_teacher) * (self.temperature**2)
        ce = self.ce_loss(student_logits, targets)
        return (self.alpha * ce) + ((1.0 - self.alpha) * kd)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1e6 if os.path.exists(path) else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> float:
    model.eval()
    correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        n += y.numel()
    return correct / n if n else 0.0


def train_distill(
    student,
    teacher,
    loader,
    val_loader,
    *,
    epochs,
    lr,
    device,
    temperature,
    alpha,
    tag,
    save_path=None,
    wb=None,
):
    """KD fine-tune `student` against frozen `teacher`. Saves best-val state_dict."""
    student.to(device)
    teacher.to(device).eval()
    criterion = KnowledgeDistillationLoss(temperature=temperature, alpha=alpha)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))

    best_acc, best_state = -1.0, None
    for ep in range(epochs):
        student.train()
        teacher.eval()
        run = seen = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            opt.zero_grad()
            s_logits = student(x)
            loss = criterion(s_logits, t_logits, y)
            loss.backward()
            opt.step()
            run += loss.item() * x.size(0)
            seen += x.size(0)
        sched.step()
        acc = evaluate(student, val_loader, device)
        log.info(
            "[%s] epoch %d/%d  loss=%.4f  val_acc=%.3f",
            tag,
            ep + 1,
            epochs,
            run / max(seen, 1),
            acc,
        )
        if wb is not None:
            wb.log({
                f"{tag}/loss": run / max(seen, 1),
                f"{tag}/val_acc": acc,
                f"{tag}/epoch": ep + 1,
            })
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(student.state_dict())
            if save_path:
                torch.save(best_state, save_path)
    if best_state is not None:
        student.load_state_dict(best_state)
    log.info("[%s] best val_acc=%.3f", tag, best_acc)
    return student, best_acc


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def build_loaders(args):
    from augment import SpecAugment, WaveformAugment
    from fsc22_dataset import (  # local import: pulls torchaudio only when needed
        FSC22Dataset,
        build_label_map,
        compute_train_stats,
    )

    all_folds = {1, 2, 3, 4, 5}
    test_fold = args.test_fold or None
    if test_fold is not None and test_fold == args.val_fold:
        raise SystemExit(
            f"--test-fold ({test_fold}) must differ from --val-fold ({args.val_fold})"
        )
    excluded = {args.val_fold} | ({test_fold} if test_fold else set())
    train_folds = sorted(all_folds - excluded)

    id_to_idx, class_names = build_label_map(args.csv)
    num_classes = len(class_names)

    os.makedirs(os.path.dirname(args.stats) or ".", exist_ok=True)
    if args.recompute_stats or not os.path.exists(args.stats):
        compute_train_stats(args.csv, args.audio_dir, train_folds, args.stats)
    with open(args.stats, encoding="utf-8") as f:
        s = json.load(f)
    norm = NormalizeMeanStd(s["mean"], s["std"])

    train_ds = FSC22Dataset(
        args.csv,
        args.audio_dir,
        train_folds,
        id_to_idx,
        norm,
        wave_aug=WaveformAugment(sample_rate=44100),
        spec_aug=SpecAugment(),
        train=True,
    )
    val_ds = FSC22Dataset(
        args.csv,
        args.audio_dir,
        [args.val_fold],
        id_to_idx,
        norm,
        train=False,
    )
    kw = dict(
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, **kw
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **kw)

    test_loader = None
    if test_fold:
        test_ds = FSC22Dataset(
            args.csv,
            args.audio_dir,
            [test_fold],
            id_to_idx,
            norm,
            train=False,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, **kw
        )
    return train_loader, val_loader, test_loader, num_classes


def build_smoke_loaders(num_classes=27, n=16, batch=4):
    """Random spectrogram batches for wiring verification (no audio/torchaudio)."""
    from torch.utils.data import TensorDataset

    x = torch.randn(n, *EXAMPLE_INPUT[1:])  # (n, 1, 40, 862)
    y = torch.randint(0, num_classes, (n,))
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch)
    return dl, dl, dl, num_classes  # train, val, test all share the random set


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
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
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    log.info("device=%s  smoke=%s", device, args.smoke)

    wb = None
    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run,
            config=vars(args),
        )
        wb = wandb

    # ---- data ----
    if args.smoke:
        args.kd_epochs = min(args.kd_epochs, 1)
        args.qat_epochs = min(args.qat_epochs, 1)
        train_loader, val_loader, test_loader, num_classes = build_smoke_loaders()
    else:
        if not (args.csv and args.audio_dir):
            raise SystemExit("--csv and --audio-dir required (or use --smoke)")
        train_loader, val_loader, test_loader, num_classes = build_loaders(args)
    log.info("num_classes=%d  test_fold=%s", num_classes, args.test_fold or None)

    cpu = torch.device("cpu")

    def test_acc(model, dev=device):
        """Held-out test accuracy (None if no test fold)."""
        return evaluate(model, test_loader, dev) if test_loader is not None else None

    def fmt(a):
        return f"{a:.3f}" if a is not None else "n/a"

    # ---- teacher (frozen, full-precision trained net) ----
    teacher = CNN_PCAw_SSRPMS_KAN(num_classes=num_classes)
    if os.path.exists(args.weights) and not args.smoke:
        teacher.load_state_dict(torch.load(args.weights, map_location="cpu"))
        log.info("loaded teacher weights: %s", args.weights)
    else:
        log.warning("teacher weights not loaded (smoke or missing): %s", args.weights)
    teacher.to(device)
    base_acc = evaluate(teacher, val_loader, device)
    log.info("TEACHER  params=%d  val_acc=%.3f", count_params(teacher), base_acc)

    # ---- STAGE 1: structured channel pruning ----
    student = copy.deepcopy(teacher)
    if args.paper_mode:
        # Iterative magnitude schedule: small asymmetric cut + brief KD rewind,
        # repeated. conv3 out + fc + KAN stay fixed (KAN frontier constraint).
        log.info(
            "PAPER-MODE: iterative asymmetric prune (step=%.2f, rounds<=%d, "
            "target=%s, rewind=%dep)",
            args.prune_step,
            args.prune_rounds,
            args.target_params or "off",
            args.rewind_epochs,
        )
        for r in range(args.prune_rounds):
            if args.target_params and count_params(student) <= args.target_params:
                log.info(
                    "reached target params (%d) -> stop pruning", args.target_params
                )
                break
            student = apply_asymmetric_pruning(student, amount=args.prune_step).to(
                device
            )
            log.info("  round %d: params=%d", r + 1, count_params(student))
            student, _ = train_distill(
                student,
                teacher,
                train_loader,
                val_loader,
                epochs=args.rewind_epochs,
                lr=args.lr,
                device=device,
                temperature=args.temperature,
                alpha=args.alpha,
                tag=f"rewind{r + 1}",
                save_path=None,
                wb=wb,
            )
        log.info(
            "PRUNED   params=%d  (asymmetric, conv3/fc/KAN intact)",
            count_params(student),
        )
    else:
        example_input = torch.randn(*EXAMPLE_INPUT)
        student = apply_structural_pruning(
            student, amount=args.prune_amount, example_input=example_input
        ).to(device)
        log.info(
            "PRUNED   params=%d  (one-shot amount=%.2f, fc reset -> needs KD)",
            count_params(student),
            args.prune_amount,
        )

    # ---- STAGE 2: KD recovery ----
    pruned_fp32_path = os.path.join(args.out_dir, "fsc22_pruned_distilled.pth")
    student, kd_acc = train_distill(
        student,
        teacher,
        train_loader,
        val_loader,
        epochs=args.kd_epochs,
        lr=args.lr,
        device=device,
        temperature=args.temperature,
        alpha=args.alpha,
        tag="KD",
        save_path=pruned_fp32_path,
        wb=wb,
    )
    log.info(
        "KD pruned fp32 saved -> %s (%.2f MB)",
        pruned_fp32_path,
        file_size_mb(pruned_fp32_path),
    )

    # ---- STAGE 3: QAT fine-tune (teacher-guided) ----
    qat_model = prepare_qat_model(student, backend=args.qat_backend).to(device)
    qat_model, qat_acc = train_distill(
        qat_model,
        teacher,
        train_loader,
        val_loader,
        epochs=args.qat_epochs,
        lr=args.qat_lr,
        device=device,
        temperature=args.temperature,
        alpha=args.alpha,
        tag="QAT",
        save_path=None,
        wb=wb,
    )

    # ---- STAGE 4: convert -> int8 (CPU) ----
    int8_model = convert_qat_model(qat_model, backend=args.qat_backend)  # -> cpu
    int8_acc = evaluate(int8_model, val_loader, cpu)
    int8_path = os.path.join(args.out_dir, "fsc22_model_optimized_int8.pth")
    torch.save(int8_model, int8_path)  # full object: quantized modules need the graph
    try:
        ts = torch.jit.script(int8_model)
        ts.save(int8_path.replace(".pth", "_scripted.pt"))
    except Exception as e:  # noqa: BLE001
        log.warning("torchscript export skipped: %s", e)

    # ---- held-out test eval (each stage scored ONCE on the untouched test fold) ----
    teacher_test = test_acc(teacher)
    kd_test = test_acc(student)
    int8_test = test_acc(int8_model, cpu)

    # ---- summary ----
    log.info("=" * 72)
    log.info("SUMMARY  (test_fold=%s)", args.test_fold or "none")
    log.info(
        "  teacher        params=%-8d  val_acc=%.3f  test=%s  size=%.2fMB",
        count_params(teacher),
        base_acc,
        fmt(teacher_test),
        file_size_mb(args.weights),
    )
    log.info(
        "  pruned+KD fp32 params=%-8d  val_acc=%.3f  test=%s  size=%.2fMB",
        count_params(student),
        kd_acc,
        fmt(kd_test),
        file_size_mb(pruned_fp32_path),
    )
    log.info("  +QAT (fake-q)               val_acc=%.3f", qat_acc)
    log.info(
        "  int8 deployed               val_acc=%.3f  test=%s  size=%.2fMB -> %s",
        int8_acc,
        fmt(int8_test),
        file_size_mb(int8_path),
        int8_path,
    )
    log.info("=" * 72)

    with open(
        os.path.join(args.out_dir, "optimize_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "val_fold": args.val_fold,
                "test_fold": args.test_fold or None,
                "paper_mode": args.paper_mode,
                "teacher": {
                    "params": count_params(teacher),
                    "val_acc": base_acc,
                    "test_acc": teacher_test,
                },
                "pruned_kd_fp32": {
                    "params": count_params(student),
                    "val_acc": kd_acc,
                    "test_acc": kd_test,
                },
                "qat_val_acc": qat_acc,
                "int8": {
                    "val_acc": int8_acc,
                    "test_acc": int8_test,
                    "size_mb": file_size_mb(int8_path),
                },
            },
            f,
            indent=2,
        )

    if wb is not None:
        wb.run.summary.update({
            "teacher/params": count_params(teacher),
            "teacher/val_acc": base_acc, "teacher/test_acc": teacher_test,
            "pruned_kd/params": count_params(student),
            "pruned_kd/val_acc": kd_acc, "pruned_kd/test_acc": kd_test,
            "qat/val_acc": qat_acc,
            "int8/val_acc": int8_acc, "int8/test_acc": int8_test,
            "int8/size_mb": file_size_mb(int8_path),
        })
        wb.finish()


if __name__ == "__main__":
    main()
