"""
Full-pipeline 5-fold cross-validation: train teacher -> compress -> test, per fold.

Mirrors the paper's experimentation loop exactly:

  for each of the 5 folds (each is the held-out TEST fold exactly once):
    1. split: 4 folds train  /  1 fold test (untouched)
    2. train the full teacher to convergence on the train folds
    3. compress on train data only:  L1 prune -> KD recovery -> QAT -> INT8
    4. evaluate the finished INT8 model on the untouched test fold
  report mean ± std of the 5 INT8 test accuracies.

A val fold (rotating neighbour, drawn from the 4 train folds) is used only for
early-stopping / best-checkpoint selection; the TEST fold is never seen during
train OR compression. Each fold runs train.py then optimize.py in fresh
subprocesses (clean CUDA state, no leakage between folds).

Run (from soundedge-esc-application/):
    python -m compression.cv_optimize \
        --csv ../data/fsc22/5-fold.csv \
        --audio-dir /path/to/FSC22/Audio/ \
        --epochs 300 --paper-mode \
        --out-dir weights/cv_opt
"""

import argparse
import json
import os
import subprocess
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.dirname(_HERE)  # soundedge-esc-application/ (cwd for `-m ...`)
ALL_FOLDS = [1, 2, 3, 4, 5]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--epochs", type=int, default=300, help="teacher training epochs")
    p.add_argument("--out-dir", default="weights/cv_opt")
    p.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=ALL_FOLDS,
        help="which folds to use as the test fold (default: all 5)",
    )
    p.add_argument(
        "--skip-train",
        action="store_true",
        help="reuse existing per-fold teacher checkpoints (only compress)",
    )
    # shared loader knobs (forwarded to both stages)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    # teacher knobs
    p.add_argument("--patience", type=int, default=100)
    # compression knobs (forwarded to optimize.py)
    p.add_argument("--paper-mode", action="store_true")
    p.add_argument("--prune-amount", type=float, default=0.5)
    p.add_argument("--prune-step", type=float, default=0.1)
    p.add_argument("--prune-rounds", type=int, default=8)
    p.add_argument("--rewind-epochs", type=int, default=2)
    p.add_argument("--target-params", type=int, default=0)
    p.add_argument("--kd-epochs", type=int, default=50)
    p.add_argument("--qat-epochs", type=int, default=15)
    # wandb (per-fold runs)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="fsc22-cv-opt")
    return p.parse_args()


def val_for(test_fold: int) -> int:
    """Rotating neighbour drawn from the train folds: 1->2, ..., 5->1."""
    return ALL_FOLDS[test_fold % len(ALL_FOLDS)]


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=_APP, check=True)


def main():
    args = parse_args()
    py = sys.executable
    os.makedirs(os.path.join(_APP, args.out_dir), exist_ok=True)

    results = []
    for test_fold in args.folds:
        val_fold = val_for(test_fold)
        fold_dir = os.path.join(args.out_dir, f"fold{test_fold}")
        os.makedirs(os.path.join(_APP, fold_dir), exist_ok=True)
        teacher = os.path.join(fold_dir, "teacher.pth")

        print(
            f"\n{'#' * 72}\n# FOLD test={test_fold} val={val_fold} train={sorted(set(ALL_FOLDS) - {test_fold})}"
            f"\n{'#' * 72}"
        )

        # ---- step 2: train teacher (test fold held out) ----
        if not (args.skip_train and os.path.exists(os.path.join(_APP, teacher))):
            cmd = [
                py,
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
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--out",
                teacher,
            ]
            if args.wandb:
                cmd += [
                    "--wandb",
                    "--wandb-project",
                    args.wandb_project,
                    "--wandb-run",
                    f"fold{test_fold}-teacher",
                ]
            run(cmd)
        else:
            print(f"[skip-train] reusing {teacher}")

        # ---- steps 3-4: compress on train data, eval on test fold ----
        cmd = [
            py,
            "-m",
            "compression.optimize",
            "--csv",
            args.csv,
            "--audio-dir",
            args.audio_dir,
            "--weights",
            teacher,
            "--val-fold",
            str(val_fold),
            "--test-fold",
            str(test_fold),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--kd-epochs",
            str(args.kd_epochs),
            "--qat-epochs",
            str(args.qat_epochs),
            "--out-dir",
            fold_dir,
        ]
        if args.paper_mode:
            cmd += [
                "--paper-mode",
                "--prune-step",
                str(args.prune_step),
                "--prune-rounds",
                str(args.prune_rounds),
                "--rewind-epochs",
                str(args.rewind_epochs),
                "--target-params",
                str(args.target_params),
            ]
        else:
            cmd += ["--prune-amount", str(args.prune_amount)]
        if args.wandb:
            cmd += [
                "--wandb",
                "--wandb-project",
                args.wandb_project,
                "--wandb-run",
                f"fold{test_fold}-opt",
            ]
        run(cmd)

        # ---- collect this fold's metrics ----
        with open(
            os.path.join(_APP, fold_dir, "optimize_metrics.json"), encoding="utf-8"
        ) as f:
            m = json.load(f)
        m["test_fold"], m["val_fold"] = test_fold, val_fold
        results.append(m)
        print(
            f"[fold {test_fold}] teacher test={m['teacher']['test_acc']}  "
            f"int8 test={m['int8']['test_acc']}"
        )

    # ---- step 5: aggregate (average the untouched-test INT8 accuracy) ----
    def mean_std(vals):
        t = torch.tensor([v for v in vals if v is not None], dtype=torch.float64)
        return (
            (t.mean().item(), t.std(unbiased=False).item()) if len(t) else (None, None)
        )

    teach_m, teach_s = mean_std([r["teacher"]["test_acc"] for r in results])
    kd_m, kd_s = mean_std([r["pruned_kd_fp32"]["test_acc"] for r in results])
    int8_m, int8_s = mean_std([r["int8"]["test_acc"] for r in results])

    print(
        f"\n{'=' * 72}\n5-FOLD FULL-PIPELINE CV SUMMARY  (mean test acc over untouched folds)\n{'=' * 72}"
    )
    for r in results:
        print(
            f"  fold {r['test_fold']} (val={r['val_fold']}): "
            f"teacher={r['teacher']['test_acc']}  "
            f"pruned+KD={r['pruned_kd_fp32']['test_acc']}  "
            f"int8={r['int8']['test_acc']}"
        )
    print("  ----")
    print(f"  teacher    test_acc = {teach_m:.3f} ± {teach_s:.3f}")
    print(f"  pruned+KD  test_acc = {kd_m:.3f} ± {kd_s:.3f}")
    print(
        f"  INT8       test_acc = {int8_m:.3f} ± {int8_s:.3f}   <-- final reported number"
    )

    summary_path = os.path.join(_APP, args.out_dir, "cv_opt_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "folds": results,
                "mean": {"teacher": teach_m, "pruned_kd": kd_m, "int8": int8_m},
                "std": {"teacher": teach_s, "pruned_kd": kd_s, "int8": int8_s},
            },
            f,
            indent=2,
        )
    print(f"  summary -> {summary_path}")


if __name__ == "__main__":
    main()
