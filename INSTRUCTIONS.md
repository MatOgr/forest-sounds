# FSC22 Acoustic Classifier — Training & Optimization Instructions

End-to-end guide for training the `CNN_PCAw_SSRPMS_KAN` model on FSC22 and
compressing it (prune → knowledge distillation → quantization-aware training →
INT8) for edge deployment.

---

## 1. Environment

- Python virtualenv: `.venv/` at repo root.
- Key deps: `torch 2.12 (+cu130)`, `torchaudio 2.11`, `soundfile`. CUDA available.
- Quantization backends present: `fbgemm`, `x86`, `qnnpack`, `onednn`.

Always invoke with the venv interpreter:

```bash
../.venv/bin/python ...        # when cwd = soundedge-esc-application/
./.venv/bin/python ...         # when cwd = repo root
```

---

## 2. Repository layout (relevant parts)

```
forest-sounds/
├── data/fsc22/5-fold.csv               # fold split + labels (Class ID, Class Name, filename, fold)
├── stats/fsc22_mel_stats.json          # cached train-set mel mean/std (reused across runs)
├── utils/
│   ├── pipeline.py                     # generic demo net + KnowledgeDistillationLoss (only the LOSS is reused)
│   └── knowledge_distillation.py       # generic KD demo loop (NOT used by optimize.py)
└── soundedge-esc-application/
    ├── model.py                        # CNN_PCAw_SSRPMS_KAN (the real model)
    ├── preprocessing.py                # audio load / mono / resample / pad / mel-dB / normalize
    ├── custom_layers/
    │   ├── PCAw_Pool.py                # PCA-weighted pooling (SVD/eigh, runs fp32 — NOT quantizable)
    │   ├── SSRP_MS.py                  # multi-scale temporal pooling (collapses time -> (B,C,F))
    │   └── WavKAN.py                    # WavKANLinear classifier head (wavelet KAN + BN)
    ├── training/
    │   ├── args.py                     # TrainArgs/SplitArgs dataclasses + DataclassArgs CLI parser
    │   ├── train.py                    # training entry point
    │   ├── cross_validate.py           # proper 5-fold CV driver (fresh subprocess per fold)
    │   ├── fsc22_dataset.py            # FSC22Dataset, label map, stats computation
    │   └── augment.py                  # waveform + spec augment + mixup
    ├── compression/
    │   ├── pruning.py                  # structured L1 channel pruning (two variants)
    │   ├── qat.py                      # QAT wrap / prepare / convert (conv3 + fc only)
    │   ├── optimize.py                 # ORCHESTRATOR: prune -> KD -> QAT -> INT8 (single fold)
    │   └── cv_optimize.py              # 5-fold CV: train+compress+test per fold, averaged
    └── weights/
        ├── fsc22_model.pth             # trained teacher (FP32)
        ├── fsc22_model.pth.classes.json
        ├── fsc22_pruned_distilled.pth  # (produced) pruned + KD-recovered FP32
        └── fsc22_model_optimized_int8.pth  # (produced) final INT8 deploy model
```

---

## 3. Model architecture

`CNN_PCAw_SSRPMS_KAN(num_classes)`:

```
input (B,1,40,T)                      # 40 mel bins, T≈862 for 5 s @ 44.1 kHz, hop 256
  conv1 = ZeroPad → Conv2d(1→64,k3) → BN → ReLU → PCAw_Pool(3×3, stride 3)
  conv2 = ZeroPad → Conv2d(64→128,k3) → BN → ReLU → AvgPool(2×2)
  conv3 = Conv2d(128→256,k3) → BN → ReLU
  ssrp_ms (SSRP_MS)   → (B,256,4)      # collapses time
  flatten             → 1024
  fc = Linear(1024→128)
  kan = WavKANLinear(128→num_classes)  # ← KAN frontier: 128-dim input is the fixed interface
```

Param hotspots: `fc` (1024×128 ≈ 131k) and `kan` dominate; conv1/conv2 are a
smaller share. This matters for the compression ceiling (see §6.4).

---

## 4. Audio / feature pipeline (fixed contract)

- Sample rate **44.1 kHz**, clip length **5 s** (zero-pad shorter, trim longer).
- Mono downmix, resample if needed.
- Mel: `n_fft=1024`, `hop=256`, `n_mels=40`, then `AmplitudeToDB`.
- Normalize with **train-set** mean/std from `stats/fsc22_mel_stats.json`.
- Output feature tensor per clip: `[1, 40, T]` → batched `[B, 1, 40, T]`.

> ⚠️ `utils/pipeline.py`'s `ProductionAudioFrontend` uses **16 kHz / 64 mels** —
> that is a generic demo and does **not** match this model. Ignore it.

---

## 5. Training

### 5.1 Spec
44.1 kHz / 5 s / zero-pad; waveform + spec augment + mixup; **AdamW** (lr 1e-3,
wd 1e-4); CE loss; cosine annealing; early stop (patience 100). Gradient
accumulation supported for VRAM-heavy PCAw SVD. Runs are **reproducible**:
`--seed` (default 42) seeds python/numpy/torch + the DataLoader shuffle
generator and reseeds workers (so augment RNG is deterministic).

### 5.2 Command (from `soundedge-esc-application/`)

```bash
../.venv/bin/python -m training.train \
    --csv ../data/fsc22/5-fold.csv \
    --audio-dir /path/to/FSC22/Audio/ \
    --val-fold 5 --epochs 300 --out weights/fsc22_model.pth \
    --batch-size 32 --accum-steps 1 --num-workers 8 --seed 42
```

### 5.3 Outputs
- `weights/fsc22_model.pth` — best-val state_dict.
- `weights/fsc22_model.pth.classes.json` — ordered class names.
- `stats/fsc22_mel_stats.json` — computed once, reused (pass `--recompute-stats` to force).

### 5.4 Data splits — train / val / test

By default `train.py` uses only **2 splits**: train (4 folds) + val (1 fold),
and the val fold is used for *both* early-stopping and best-model selection — so
that `val_acc` is optimistically biased, not a clean test estimate.

- **3-way split**: pass `--test-fold N`. That fold is excluded from *both* train
  and val, and evaluated **exactly once** after training (best checkpoint
  reloaded). Must differ from `--val-fold`. Writes `<out>.metrics.json` with
  `best_val_acc`, `test_acc`, and the fold assignment.

  ```bash
  ../.venv/bin/python -m training.train \
      --csv ../data/fsc22/5-fold.csv --audio-dir /path/to/FSC22/Audio/ \
      --val-fold 4 --test-fold 5 --epochs 300 --out weights/fsc22_model.pth
  ```

- **Proper 5-fold CV** (`training/cross_validate.py`): each fold is the held-out
  test fold exactly once (val = rotating neighbour, train = other 3). Each fold
  runs in a **fresh subprocess** (clean CUDA state). Reports per-fold test acc +
  **mean ± std**, saved to `weights/cv/cv_summary.json`. Args after `--` are
  forwarded to `train.py`.

  ```bash
  ../.venv/bin/python -m training.cross_validate \
      --csv ../data/fsc22/5-fold.csv --audio-dir /path/to/FSC22/Audio/ \
      --epochs 300 --out-dir weights/cv \
      -- --batch-size 32 --num-workers 8
  ```

  Fold assignment: test 1→val 2, 2→3, 3→4, 4→5, 5→1. `--seed` (default 42) is
  forwarded to every fold's `train.py` subprocess for reproducibility.

`optimize.py` accepts the same `--test-fold N` (excluded from train + KD/QAT
selection); every stage — teacher, pruned+KD, INT8 — is then scored once on that
fold, written to `weights/optimize_metrics.json`.

### 5.5 Experiment tracking (Weights & Biases)

Optional, off by default; `wandb` is a lazy import (no dependency unless `--wandb`).

```bash
pip install wandb && wandb login            # once
```

- **`train.py`**: `--wandb [--wandb-project P] [--wandb-run NAME] [--wandb-entity E]`.
  Logs per-epoch `train/`·`val/` loss+acc, `lr`, `best_val_acc`; final summary gets
  `best_val_acc`, `best_epoch`, `test_acc`. Default project `fsc22-esc`.
- **`cross_validate.py`**: `--wandb [--wandb-project P]` → each fold is a **separate
  run** named `testfold{N}` (default project `fsc22-esc-cv`). Do *not* also pass
  `--wandb-run` after `--` (it would clash with the per-fold names).
- **`optimize.py`**: `--wandb [...]` → per-epoch curves tagged by stage
  (`rewind{r}/`, `KD/`, `QAT/`); summary holds teacher / pruned_kd / qat / int8
  val+test acc, params, int8 size. Default project `fsc22-optimize`.

Offline (no login): `export WANDB_MODE=offline`.

### 5.6 Be aware
- `--batch-size`/`--accum-steps`: PCAw_Pool's per-patch SVD/eigh is VRAM- and
  compute-heavy. Effective batch = `batch_size × accum_steps`.
- `--amp` only takes effect on CUDA. PCAw linalg always forces fp32 internally
  (SVD/eigh unstable under fp16).
- Stats are cached by path; if you change the dataset, use `--recompute-stats`.
- `--seed` makes a run reproducible (RNG + shuffle + worker reseed). Note
  `cudnn.benchmark=True` is left on for speed, so CUDA conv kernels may still
  introduce tiny nondeterminism; CPU runs are fully deterministic.
- `--stats` defaults to empty → auto path: **per-split**
  (`stats/fsc22_mel_stats_val{V}_test{T}.json`) whenever `--test-fold` is set, so
  CV folds never share a mean/std computed over another fold's data. Without a
  test fold it stays the canonical `stats/fsc22_mel_stats.json` (reproduces the
  existing teacher). `optimize.py` keeps using the teacher's stats file (its
  normalization must match the teacher) — pass `--stats` to override.

---

## 6. Optimization (prune → KD → QAT → INT8)

Single orchestrator: `compression/optimize.py`. Pipeline order is **fixed and
intentional** (see §6.3). Run from `soundedge-esc-application/`.

### 6.1 Two pruning modes

**(a) Default — one-shot symmetric** (`apply_structural_pruning`):
Prunes conv1, conv2 **and conv3** output channels at `--prune-amount`, then
**rebuilds `fc` with random weights** (flatten dim changes). Larger, faster size
cut; relies on KD to recover the discarded `fc`. *Not* the paper's method.

```bash
../.venv/bin/python -m compression.optimize \
    --csv ../data/fsc22/5-fold.csv --audio-dir /path/to/FSC22/Audio/ \
    --weights weights/fsc22_model.pth --val-fold 5 \
    --prune-amount 0.5 --kd-epochs 50 --qat-epochs 15 --batch-size 32
```

**(b) `--paper-mode` — iterative asymmetric** (paper-faithful, see `PRUNING.md`):
- L1 magnitude, **structured** (whole channels).
- **Asymmetric**: prunes conv1 + conv2 only; **conv3 output is LOCKED at 256**
  → flatten stays 1024 → `fc` and `kan` (and its spline grids) are **never
  reset**. Respects the KAN-frontier constraint.
- **Iterative**: small cut per round (`--prune-step`) + short KD "rewind"
  (`--rewind-epochs`), repeated `--prune-rounds` times or until `--target-params`.

```bash
../.venv/bin/python -m compression.optimize --paper-mode \
    --csv ../data/fsc22/5-fold.csv --audio-dir /path/to/FSC22/Audio/ \
    --weights weights/fsc22_model.pth --val-fold 5 \
    --prune-step 0.1 --prune-rounds 8 --rewind-epochs 2 \
    --target-params 50000 --kd-epochs 30 --qat-epochs 15 --batch-size 32
```

### 6.2 Stages (both modes)
1. **Prune** the student (deep copy of teacher).
2. **KD recovery** — distill frozen full teacher → pruned student
   (`KnowledgeDistillationLoss`, AdamW, cosine, best-val saved to
   `fsc22_pruned_distilled.pth`).
3. **QAT fine-tune** — `prepare_qat_model` inserts fake-quant on **conv3 + fc
   only**; fine-tune teacher-guided (KD loss again). conv1/conv2/PCAw/SSRP/KAN
   stay float.
4. **Convert INT8** — `convert_qat_model` → CPU INT8 model. Saved as a full
   object to `fsc22_model_optimized_int8.pth` (+ TorchScript `.pt` if scriptable).

### 6.3 Why this order
`apply_structural_pruning` (default mode) rebuilds `fc` randomly, so the pruned
net is broken until retrained — KD **must** follow pruning. QAT must follow KD so
fake-quant fine-tunes already-recovered weights. Convert is last (irreversible).

### 6.4 Be aware
- **INT8 model is CPU-only** (fbgemm/x86). Do not move it to CUDA.
- Only **conv3 + fc** are quantized. PCAw_Pool (SVD/eigh), SSRP, KAN, conv1,
  conv2 remain FP32 by design — quantizing the linalg ops is unsupported/unstable.
- **`--paper-mode` cannot easily reach ~50k params.** `fc`+`kan`+conv3 are
  locked by the KAN-frontier rule, and they hold most parameters. Expect a
  plateau well above 50k. Reaching the paper's headline number needs an extra
  lever (e.g. pruning conv3 out + re-fitting `fc` while keeping kan's 128-dim
  input) — not yet implemented. Ask before adding `--prune-fc`.
- **Quantized model loading**: it is saved as a pickled object (`torch.save(model)`),
  not just a state_dict. Reload with `torch.load(path)` (full object). A bare
  `state_dict` reload requires re-running `prepare_qat_model` + `convert` first.
- TorchScript export of the INT8 model may be skipped (PCAw_Pool `.norm` isn't
  scriptable) — this is logged and non-fatal; the `.pth` object is the source of truth.
- Stats: defaults to `../stats/fsc22_mel_stats.json` (repo-root copy). Use the
  same stats the teacher trained with, or accuracy will drift.

### 6.5 Wiring check (no data / audio needed)
Runs the full pipeline on random tensors with tiny epochs to verify imports,
shapes, prune→KD→QAT→convert, and INT8 export:

```bash
../.venv/bin/python -m compression.optimize --smoke
../.venv/bin/python -m compression.optimize --smoke --paper-mode --prune-rounds 3
```
(Accuracy will read 0.0 — random teacher/data; that's expected.)

### 6.6 Full-pipeline 5-fold CV (`compression/cv_optimize.py`)

The paper's complete experimentation loop, automated end to end: for each of the
5 folds (each held out as TEST exactly once) it **trains the teacher → prunes →
KD → QAT → INT8 → evaluates the untouched test fold**, then averages. Each stage
runs in a fresh subprocess (clean CUDA state, no leakage); the test fold is never
seen during training or compression, so the averaged INT8 accuracy is unbiased.
A val fold (rotating neighbour from the 4 train folds) is used only for
early-stop / KD-selection.

```bash
../.venv/bin/python -m compression.cv_optimize \
    --csv ../data/fsc22/5-fold.csv --audio-dir /path/to/FSC22/Audio/ \
    --epochs 300 --paper-mode \
    --prune-step 0.1 --prune-rounds 8 --rewind-epochs 2 \
    --kd-epochs 30 --qat-epochs 15 --batch-size 32 \
    --out-dir weights/cv_opt
```

- Reports `teacher` / `pruned+KD` / `INT8` test acc as **mean ± std**; the INT8
  line is the final reported number. Saved to `weights/cv_opt/cv_opt_summary.json`;
  per-fold artifacts in `weights/cv_opt/fold{N}/`.
- `--wandb` → per-fold runs `fold{N}-teacher` + `fold{N}-opt` (project `fsc22-cv-opt`).
- `--skip-train` → reuse existing per-fold teachers, recompress only.
- `--folds 1 3` → run a subset; drop `--paper-mode` + add `--prune-amount` for one-shot.
- **Heavy**: 5× (full training + compression). Plan compute accordingly.

> Use `cv_optimize.py` when you want the headline compressed-model number.
> Use `optimize.py` (single fold) for iterating on one split; `train.py` /
> `cross_validate.py` for the uncompressed teacher only.

---

## 7. Key CLI arguments (`optimize.py`)

| Arg | Default | Meaning |
|-----|---------|---------|
| `--csv`, `--audio-dir` | — | dataset (required unless `--smoke`) |
| `--weights` | `weights/fsc22_model.pth` | trained teacher |
| `--stats` | `../stats/fsc22_mel_stats.json` | mel normalization stats |
| `--val-fold` | 5 | held-out fold |
| `--prune-amount` | 0.5 | one-shot prune fraction (default mode) |
| `--paper-mode` | off | iterative asymmetric prune |
| `--prune-step` | 0.1 | per-round fraction (paper-mode) |
| `--prune-rounds` | 8 | max prune+rewind rounds (paper-mode) |
| `--target-params` | 0 (off) | early stop on param count (paper-mode) |
| `--rewind-epochs` | 2 | KD epochs per round (paper-mode) |
| `--kd-epochs` | 50 | full KD recovery epochs |
| `--qat-epochs` | 15 | QAT fine-tune epochs |
| `--lr` / `--qat-lr` | 1e-3 / 1e-4 | learning rates |
| `--temperature` / `--alpha` | 3.0 / 0.4 | KD softmax temp / CE weight |
| `--qat-backend` | fbgemm | quantization engine |
| `--smoke` | off | random-data wiring test |

---

## 8. Produced artifacts

| File | Stage | Notes |
|------|-------|-------|
| `weights/fsc22_pruned_distilled.pth` | after KD | pruned FP32 state_dict (best val) |
| `weights/fsc22_model_optimized_int8.pth` | after convert | INT8 full object, CPU-only |
| `weights/fsc22_model_optimized_int8_scripted.pt` | after convert | TorchScript, if scriptable |

---

## 9. Quick reference

```bash
# Train
cd soundedge-esc-application
../.venv/bin/python -m training.train --csv ../data/fsc22/5-fold.csv \
    --audio-dir /path/to/FSC22/Audio/ --val-fold 5 --epochs 300 \
    --out weights/fsc22_model.pth --batch-size 32 --num-workers 8 --seed 42

# Optimize (paper-faithful)
../.venv/bin/python -m compression.optimize --paper-mode \
    --csv ../data/fsc22/5-fold.csv --audio-dir /path/to/FSC22/Audio/ \
    --weights weights/fsc22_model.pth --val-fold 5

# Verify wiring only
../.venv/bin/python -m compression.optimize --smoke --paper-mode
```
