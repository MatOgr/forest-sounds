### 1. SGD with Nesterov Momentum (Best for Small Datasets)

Forces uniform weight transitions and adds heavy regularization, preventing the model from overfitting to the small number of clips in FSC22.

```python
import torch.optim as optim

# Ideal baseline setup for small audio datasets
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True,
    weight_decay=1e-4
)

```

### 2. LBFGS (Best for KAN-based Layers)

A second-order optimizer that tracks loss landscape curvature. Highly recommended for optimizing the complex spline activation functions inside the **CNN-PSK** architecture.

```python
import torch.optim as optim

# Note: Requires a closure function inside your training loop
optimizer = optim.LBFGS(
    model.parameters(),
    lr=0.1,
    max_iter=20,
    history_size=100
)

# Standard training loop implementation step:
def closure():
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    return loss

optimizer.step(closure)

```

### 3. Sophia-G (Best Fast Hybrid Option)

Provides second-order curvature stability (essential for KAN components) while keeping the fast, per-epoch execution speed of AdamW.

> [!warning]
> Requires revisiting the implementation, not fully functioning

```python
# Requires the 'sophia-optimizer' package (pip install sophia-optimizer)
from sophia import SophiaG

optimizer = SophiaG(
    model.parameters(),
    lr=2e-4,
    betas=(0.965, 0.99),
    rho=0.01,
    weight_decay=1e-1
)

```

---

## Using them in `training.train`

All four are wired through the `--optimizer` flag (default `adamw`). The optimizer
is built by `build_optimizer()` and the training loop (`run_epoch`) handles the
LBFGS closure path automatically.

### CLI

```bash
# AdamW (default, paper baseline)
python -m training.train --csv ... --audio-dir ...

# SGD + Nesterov
python -m training.train --csv ... --audio-dir ... \
    --optimizer sgd --lr 1e-2 --momentum 0.9 --nesterov --weight-decay 1e-4

# LBFGS (forces AMP + accum-steps off)
python -m training.train --csv ... --audio-dir ... \
    --optimizer lbfgs --lr 1e-1 --lbfgs-max-iter 20 --lbfgs-history 100

# Sophia-G  (pip install sophia-optimizer)
python -m training.train --csv ... --audio-dir ... \
    --optimizer sophia --lr 2e-4 --sophia-rho 0.01 --weight-decay 1e-1
```

### Flags

| Flag               | Applies to       | Default | Notes                                      |
| ------------------ | ---------------- | ------- | ------------------------------------------ |
| `--optimizer`      | all              | `adamw` | `adamw` \| `sgd` \| `lbfgs` \| `sophia`    |
| `--lr`             | all              | `1e-3`  | **Retune per optimizer** — see table below |
| `--weight-decay`   | adamw/sgd/sophia | `1e-4`  | LBFGS has none; Sophia wants `1e-1`        |
| `--momentum`       | sgd              | `0.9`   |                                            |
| `--nesterov`       | sgd              | off     | store-true; pass to enable                 |
| `--sophia-rho`     | sophia           | `0.01`  | Hessian clip threshold                     |
| `--lbfgs-max-iter` | lbfgs            | `20`    | inner iters per `.step()`                  |
| `--lbfgs-history`  | lbfgs            | `100`   | curvature history size                     |

## (Hypothetical) Best practices & caveats

- **Cosine schedule applies to all.** `CosineAnnealingLR(T_max=epochs)` wraps
  whichever optimizer you pick — no extra flag.
- **AdamW** — keep as the default/reproducible baseline; compare others against it.
- **SGD + Nesterov** — strongest small-data regularizer here, but slowest to
  converge: budget more epochs and pair with _warmup_ low start or a longer
  cosine.
- **LBFGS** — second-order, **full-batch by design**.  
  Mini-batch + mixup makes the objective noisy across closure evals; the loop fixes one (mixed) batch per step to keep each closure consistent, but one should expect instability. Each step does `max_iter` forward/backward passes (≈5–20× slower/step) and **AMP + grad accumulation are disabled**.  
  Try only on small/quasi-full batches – if it diverges, drop `--lr` and `--lbfgs-max-iter` (e.g. to `1e-2` , `5` respectively).
- **Sophia-G** — needs `pip install sophia-optimizer`. Current integration runs
  the **gradient step only** (no periodic Hessian/`update_hessian()` call), so it
  behaves close to AdamW with curvature-clipped updates rather than full Sophia.
  Use AMP-friendly; good middle ground if AdamW plateaus.
- **Compare fairly** — hold `--seed`, folds, epochs, and mel config fixed; only
  swap `--optimizer` + its LR. Log each as a separate W&B run via `--wandb-run`.
