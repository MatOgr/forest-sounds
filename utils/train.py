import time

import numpy as np
from torch import nn

from utils.pipeline import ProductionAudioFrontend

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.stop = False

    def step(self, value):
        if self.best is None:
            self.best = value
            return False
        improved = (
            (value < self.best - self.min_delta)
            if self.mode == "min"
            else (value > self.best + self.min_delta)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad() if _HAS_TORCH else (lambda f: f)
def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def fit(
    model,
    train_loader,
    val_loader,
    opt,
    loss_fn,
    device,
    epochs=30,
    patience=7,
    scheduler=None,
    ckpt_path=None,
    verbose=True,
):
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    es = EarlyStopping(patience=patience, mode="min")
    best_val = float("inf")
    for ep in range(epochs):
        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, opt, loss_fn, device)
        vl, va = eval_epoch(model, val_loader, loss_fn, device)
        if scheduler is not None:
            scheduler.step(vl) if hasattr(
                scheduler, "step"
            ) and "ReduceLROnPlateau" in type(scheduler).__name__ else scheduler.step()
        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)
        if vl < best_val:
            best_val = vl
            if ckpt_path:
                torch.save(model.state_dict(), ckpt_path)
        if verbose:
            print(
                f"ep {ep + 1:02d} | {time.time() - t0:5.1f}s | "
                f"tr_loss {tl:.4f} acc {ta:.3f} | val_loss {vl:.4f} acc {va:.3f}"
            )
        if es.step(vl):
            if verbose:
                print(f"early stop at epoch {ep + 1}")
            break
    return history


def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x).argmax(1).cpu().numpy()
            ps.append(out)
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# =====================================================================
# PARAMETRIZED TRAINING (RECORDINGS OR SPECTROGRAMS INPUT)
# =====================================================================
def train_model(
    model: nn.Module,
    dataloader,
    input_type: str = "recordings",
    frontend: ProductionAudioFrontend | None = None,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
):
    """
    Trains `model` on either raw audio recordings or precomputed spectrograms.

    input_type:
        "recordings"   -> batch x = (B, samples); `frontend` applied to make log-mel.
                        Requires `frontend` (defaults to a fresh ProductionAudioFrontend).
        "spectrograms" -> batch x = (B, 1, n_mels, time); fed straight to model.

    dataloader yields (x, targets).
    """
    if input_type not in ("recordings", "spectrograms"):
        raise ValueError(
            f"input_type must be 'recordings' or 'spectrograms', got {input_type!r}"
        )

    if input_type == "recordings" and frontend is None:
        frontend = ProductionAudioFrontend()

    model = model.to(device)
    if frontend is not None:
        frontend = frontend.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for x, targets in dataloader:
            x, targets = x.to(device), targets.to(device)

            # Recordings need frontend transform; spectrograms are model-ready.
            if input_type == "recordings":
                x = frontend(x)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        avg = running_loss / len(dataloader.dataset)
        print(f"[{input_type}] epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    return model
