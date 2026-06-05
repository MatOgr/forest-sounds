"""
Proper 5-fold cross-validation driver for CNN_PCAw_SSRPMS_KAN on FSC22.

Each of the 5 folds serves as the held-out TEST fold exactly once. For each
test fold a neighbouring fold is used as the VAL fold (early-stop + model
selection); the remaining 3 folds train. The 5 unbiased test accuracies are
averaged -> mean ± std.

    test_fold:  1  2  3  4  5
    val_fold:   2  3  4  5  1   (rotating neighbour)
    train:      others (3 folds)

Each fold trains in a FRESH subprocess (clean CUDA state, no leakage between
folds). Per-fold metrics come from `<out>.metrics.json` written by train.py.

Run (from soundedge-esc-application/):
    python -m training.cross_validate \
        --csv ../data/fsc22/5-fold.csv \
        --audio-dir /path/to/FSC22/Audio/ \
        --epochs 300 --out-dir weights/cv \
        -- --batch-size 32 --num-workers 8        # args after `--` are forwarded to train.py
"""

import json
import os
import subprocess
import sys
from dataclasses import dataclass

import torch

from .args import SplitArgs

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.dirname(_HERE)  # soundedge-esc-application/ (cwd for `-m training.*`)

ALL_FOLDS = [1, 2, 3, 4, 5]


# --------------------------------------------------------------------------- #
# Fold / split resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Splits:
    train_folds: list
    val_folds: list
    test_fold: int | None  # None when disabled

    @classmethod
    def resolve(cls, val_fold, test_fold):
        test_fold = test_fold or None
        if test_fold is not None and test_fold == val_fold:
            raise SystemExit(
                f"--test-fold ({test_fold}) must differ from --val-fold ({val_fold})"
            )
        excluded = {val_fold} | ({test_fold} if test_fold else set())
        return cls(
            train_folds=sorted(set(ALL_FOLDS) - excluded),
            val_folds=[val_fold],
            test_fold=test_fold,
        )


def val_for(test_fold: int) -> int:
    """Rotating neighbour: 1->2, 2->3, ..., 5->1."""
    return ALL_FOLDS[test_fold % len(ALL_FOLDS)]


def main():
    args = SplitArgs.parse_args()
    os.makedirs(os.path.join(_APP, args.out_dir), exist_ok=True)
    # Strip a leading '--' separator if argparse left it in REMAINDER.
    forward = args.forward[1:] if args.forward[:1] == ["--"] else args.forward

    results = []
    for test_fold in args.folds:
        val_fold = val_for(test_fold)
        out = os.path.join(args.out_dir, f"fsc22_testfold{test_fold}.pth")
        cmd = [
            sys.executable,
            "-m",
            "training.train",
            "--csv",
            args.csv,
            "--audio-dir",
            args.audio_dir,
            "--val-fold",
            str(val_fold),
            "--test-fold",
            str(test_fold),
            "--epochs",
            str(args.epochs),
            "--out",
            out,
            *forward,
        ]
        if args.wandb:
            # Distinct run name per fold so the 5 runs don't collide.
            cmd += [
                "--wandb",
                "--wandb-project",
                args.wandb_project,
                "--wandb-run",
                f"testfold{test_fold}",
            ]
        print(
            f"\n{'=' * 70}\n[CV] test_fold={test_fold} val_fold={val_fold} -> {out}\n{'=' * 70}"
        )
        subprocess.run(cmd, cwd=_APP, check=True)

        metrics_path = os.path.join(_APP, out + ".metrics.json")
        with open(metrics_path, encoding="utf-8") as f:
            m = json.load(f)
        results.append(m)
        print(
            f"[CV] fold {test_fold}: test_acc={m['test_acc']:.3f} "
            f"(val_acc={m['best_val_acc']:.3f})"
        )

    # ---- aggregate ----
    test_accs = torch.tensor([r["test_acc"] for r in results], dtype=torch.float64)
    mean, std = test_accs.mean().item(), test_accs.std(unbiased=False).item()

    print(f"\n{'=' * 70}\n5-FOLD CV SUMMARY\n{'=' * 70}")
    for r in results:
        print(
            f"  test_fold={r['test_fold']} (val={r['val_fold']}): "
            f"test_acc={r['test_acc']:.3f}  val_acc={r['best_val_acc']:.3f}"
        )
    print(f"  ----\n  mean test_acc = {mean:.3f} ± {std:.3f}  (n={len(results)})")

    summary_path = os.path.join(_APP, args.out_dir, "cv_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {"folds": results, "mean_test_acc": mean, "std_test_acc": std},
            f,
            indent=2,
        )
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()
