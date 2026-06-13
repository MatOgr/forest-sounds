# XAI — Layer Activation Visualization

`utils/xai.py` — capture and visualize what each stage/layer of a model produces
on a given input. Model-agnostic (works on any `nn.Module` via forward hooks).

## API

### `LayerActivations(model, layers=None)`
Registers forward hooks on named layers; one forward pass returns an ordered
`{name: tensor}` dict (detached, on CPU).

- `layers=None` → model's top-level stages (direct children).
- `layers=list[str]` → dotted names, incl. nested (e.g. `"conv1.1"`).
- Call it (or `.capture(x)`) to run a pass. Use as a context manager, or call
  `.remove()`, to detach hooks. `.layer_names` lists tapped layers.

### `plot_activations(acts, sample=0, max_channels=16, cmap='viridis')`
Renders captured dict → `list[(name, fig)]`, dispatched by tensor rank:
- 4-D `(B,C,H,W)` → grid of channel heatmaps + mean overview.
- 3-D `(B,C,F)` → single heatmap.
- 2-D `(B,N)` → vector line plot.

## Usage

```python
import torch
from utils import LayerActivations, plot_activations
from model import CNN_PCAw_SSRPMS_KAN

model = CNN_PCAw_SSRPMS_KAN(num_classes=27)
x = torch.randn(1, 1, 40, 200)          # (B, 1, n_mels, time) mel-dB

# Default: top stages
acts = LayerActivations(model)(x)
for name, t in acts.items():
    print(name, tuple(t.shape))
# conv1 (1,64,13,66)  conv2 (1,128,6,32)  conv3 (1,256,4,30)
# ssrp_ms (1,256,4)  freq_pool (1,256,4)
# flatten (1,1024)  fc (1,128)  kan (1,27)

# Nested taps (auto-cleans hooks)
with LayerActivations(model, ["conv1.1", "conv3.0"]) as cap:
    acts = cap.capture(x)

# Plot
for name, fig in plot_activations(acts):
    fig.savefig(f"{name}.png")
```

## Note — import path
Root `utils/` package name collides with `soundedge-esc-application/utils.py`.
Run from the repo root and append the app dir last so root `utils/` wins:

```python
import sys; sys.path.append("soundedge-esc-application")
from utils import LayerActivations   # root package
from model import CNN_PCAw_SSRPMS_KAN # app dir
```
