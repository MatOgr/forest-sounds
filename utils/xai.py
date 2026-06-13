"""Layer-activation capture + visualization for explainability (XAI).

Model-agnostic: works on any ``nn.Module`` given layer names resolved via
``model.named_modules()``. Forward hooks record each tapped layer's output so you
can inspect what consecutive stages produce on a given input.
"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


class LayerActivations:
    """Capture outputs of named layers via forward hooks.

    Parameters
    ----------
    model : nn.Module
    layers : list[str] | None
        Dotted module names (e.g. ``"conv1"``, nested ``"conv1.1"``). ``None``
        defaults to the model's direct named children (top-level stages).

    Usage
    -----
    >>> acts = LayerActivations(model)(x)        # OrderedDict name -> tensor
    >>> with LayerActivations(model, ["conv1.1"]) as cap:
    ...     acts = cap.capture(x)
    """

    def __init__(self, model, layers=None):
        self.model = model
        modules = dict(model.named_modules())

        if layers is None:
            layers = [name for name, _ in model.named_children()]

        missing = [n for n in layers if n not in modules]
        if missing:
            raise KeyError(
                f"layers not found in model: {missing}. "
                f"available: {sorted(n for n in modules if n)}"
            )

        self.layers = list(layers)
        self._acts = OrderedDict()
        self._handles = []
        for name in self.layers:
            handle = modules[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            self._acts[name] = out.detach().cpu()

        return hook

    @property
    def layer_names(self):
        return list(self.layers)

    @torch.no_grad()
    def capture(self, x):
        """Run one forward pass; return ordered {name: activation} dict."""
        self._acts = OrderedDict()
        was_training = self.model.training
        self.model.eval()
        self.model(x)
        if was_training:
            self.model.train()
        # Re-order to match requested layer order (hooks fire in forward order).
        return OrderedDict((n, self._acts[n]) for n in self.layers if n in self._acts)

    __call__ = capture

    def remove(self):
        """Detach all forward hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


def plot_activations(acts, sample=0, max_channels=16, cmap="viridis"):
    """
    Render captured activations -> list of (name, fig).

    Dispatch by tensor rank (batch item ``sample`` selected):
    - 4-D (B,C,H,W): grid of first ``max_channels`` channel heatmaps + a
    mean-over-channels overview panel.
    - 3-D (B,C,F): single C x F heatmap.
    - 2-D (B,N): line/bar plot of the vector.
    """
    figs = []
    for name, t in acts.items():
        a = t[sample].numpy()
        if a.ndim == 3:  # (C, H, W)
            figs.append((name, _plot_maps(a, name, max_channels, cmap)))
        elif a.ndim == 2:  # (C, F)
            figs.append((name, _plot_heatmap(a, name, cmap)))
        elif a.ndim == 1:  # (N,)
            figs.append((name, _plot_vector(a, name)))
        else:
            raise ValueError(f"{name}: unsupported activation rank {a.ndim}")
    return figs


def _plot_maps(a, name, max_channels, cmap):
    c = a.shape[0]
    n = min(max_channels, c)
    cols = int(np.ceil(np.sqrt(n + 1)))
    rows = int(np.ceil((n + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    axes = np.atleast_1d(axes).ravel()

    axes[0].imshow(a.mean(0), aspect="auto", origin="lower", cmap=cmap)
    axes[0].set_title("mean", fontsize=8)
    for i in range(n):
        axes[i + 1].imshow(a[i], aspect="auto", origin="lower", cmap=cmap)
        axes[i + 1].set_title(f"ch{i}", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n + 1 :]:
        ax.axis("off")

    fig.suptitle(f"{name}  (C={c}, showing {n})")
    fig.tight_layout()
    return fig


def _plot_heatmap(a, name, cmap):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(a, cmap=cmap, ax=ax, cbar=True)
    ax.set_title(f"{name}  {a.shape}")
    ax.set_xlabel("F")
    ax.set_ylabel("C")
    fig.tight_layout()
    return fig


def _plot_vector(a, name):
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.plot(a, linewidth=0.8)
    ax.set_title(f"{name}  (N={a.shape[0]})")
    ax.set_xlabel("unit")
    ax.set_ylabel("activation")
    ax.margins(x=0)
    fig.tight_layout()
    return fig
