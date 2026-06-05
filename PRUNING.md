In the paper, the authors perform **Structured Channel Pruning** using an **Iterative Magnitude-Based Strategy**, combined with an architecture-specific constraint tailored to their hybrid Convolution-KAN setup.

Instead of removing individual random weights scattered across the network (unstructured pruning), they permanently remove **entire convolutional filters**.

---

### 1. Calculate the Magnitude (Importance Score) of Each Filter

The pruning algorithm evaluates the importance of each convolutional filter by calculating its **$L_1$-norm** (the sum of the absolute values of all the weights within that specific filter matrix).

Mathematically, for a filter $W_i$, the importance score is:

$$Score_i = \sum |W_i|$$

**The core assumption** – filters with very small weight magnitudes contribute negligible feature maps to the downstream layers **VS** large magnitudes contain critical acoustic patterns (transient edges, pitch indicators, etc.).

---

### 2. The Iterative Pruning Schedule (Not a One-Shot Cut)

A major reason compressed models fail or lose massive amounts of accuracy is "pruning shock" — cutting too many filters at once. The authors avoided this by using an **iterative schedule**:

1. Train the baseline model to maximum convergence first.
2. Prune a small percentage (e.g., 10%) of the lowest-magnitude filters globally.
3. Run a brief "rewind" / fine-tuning phase (with Knowledge Distillation loss) for 1–2 epochs to let the remaining filters adapt / compensate for the missing channels.
4. Repeat steps 2 and 3 until reaching the target structural compression limit (which dropped the model down to its final ~50k parameter blueprint).

---

### 3. The KAN-Frontier Constraint (Why it's unique)

Back-end of this network uses a **Kolmogorov-Arnold Network (KAN)** (not standard MLP). The pruning algorithm had to respect a hard physical constraint at the boundary where the CNN meets the KAN.

```
[Conv Layers] ── (Pruned Channels) ──> [SSRP Aggregator] ── (Fixed Interface) ──> [KAN Classifier Head]

```

- **Standard MLPs** are incredibly forgiving; changing the number of incoming features → you easily alter the weight matrix dimensions.
- **KANs** map variables via complex, explicit 1D B-spline curves on edges. Dynamically changing the input dimensions of a KAN graph mid-training can break mathematical structure / force a complete re-initialization of the spline grids.

**To bypass**: the paper applied **asymmetric channel pruning** – heavily pruned internal channels of `Conv1` and `Conv2`, keeping the output feature map size of the final block (`Conv3`) aligned with KAN's (classification head) fixed input width.

### Summary

> _"We are implementing Magnitude-Based Structured L1 pruning iteratively. By targeting full channels instead of unstructured weights, we achieve true hardware-level acceleration. Furthermore, we are locking the output channel boundary of the final convolutional block to match the KAN classifier interface, preserving the spline grid layouts of the classification head while aggressively stripping redundancy from the early feature-extraction layers."_
