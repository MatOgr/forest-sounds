import torch
import torch.nn as nn

def _make_pruned_conv(old_conv: nn.Conv2d, keep_out_idx: torch.Tensor) -> nn.Conv2d:
    """
    Create a new Conv2d with fewer output channels (keep_out_idx),
    copying weights/bias from old_conv.
    """
    device = old_conv.weight.device
    dtype = old_conv.weight.dtype

    new_out = keep_out_idx.numel()
    new_conv = nn.Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=new_out,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=(old_conv.bias is not None),
        padding_mode=old_conv.padding_mode,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        new_conv.weight.copy_(old_conv.weight.data[keep_out_idx].contiguous())
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias.data[keep_out_idx].contiguous())

    return new_conv


def _make_pruned_bn(old_bn: nn.BatchNorm2d, keep_idx: torch.Tensor) -> nn.BatchNorm2d:
    """
    Create a new BatchNorm2d with fewer channels, copying params + running stats.
    """
    device = old_bn.weight.device
    dtype = old_bn.weight.dtype

    new_nf = keep_idx.numel()
    new_bn = nn.BatchNorm2d(
        num_features=new_nf,
        eps=old_bn.eps,
        momentum=old_bn.momentum,
        affine=old_bn.affine,
        track_running_stats=old_bn.track_running_stats,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        if old_bn.affine:
            new_bn.weight.copy_(old_bn.weight.data[keep_idx].contiguous())
            new_bn.bias.copy_(old_bn.bias.data[keep_idx].contiguous())

        if old_bn.track_running_stats:
            new_bn.running_mean.copy_(old_bn.running_mean.data[keep_idx].contiguous())
            new_bn.running_var.copy_(old_bn.running_var.data[keep_idx].contiguous())
            new_bn.num_batches_tracked.copy_(old_bn.num_batches_tracked)

    return new_bn


def prune_conv_bn_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d, amount: float = 0.3):
    """
    Structurally prune Conv2d output channels using L1 norm.
    Returns: (new_conv, new_bn, keep_out_idx)
    """
    if not (0.0 <= amount < 1.0):
        raise ValueError("amount must be in [0, 1).")

    W = conv.weight.data  # (out, in, kH, kW)
    out_ch = W.shape[0]
    num_prune = int(round(amount * out_ch))

    if num_prune <= 0:
        keep_idx = torch.arange(out_ch, device=W.device)
        return conv, bn, keep_idx

    # L1 norm per output channel
    channel_l1 = W.abs().sum(dim=(1, 2, 3))  # (out,)

    # Keep the highest-L1 channels
    sorted_idx = torch.argsort(channel_l1, descending=True)
    keep_idx = sorted_idx[num_prune:]  # (kept,)

    # Keep indices sorted for nicer determinism
    keep_idx, _ = torch.sort(keep_idx)

    new_conv = _make_pruned_conv(conv, keep_idx)
    new_bn = _make_pruned_bn(bn, keep_idx)

    return new_conv, new_bn, keep_idx


def prune_conv_input_channels(conv: nn.Conv2d, keep_in_idx: torch.Tensor) -> nn.Conv2d:
    """
    Prune Conv2d input channels by selecting keep_in_idx on dim=1 of weight.
    Returns a new conv with in_channels = len(keep_in_idx).
    """
    device = conv.weight.device
    dtype = conv.weight.dtype

    new_in = keep_in_idx.numel()
    new_conv = nn.Conv2d(
        in_channels=new_in,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,  # assumes groups-compatible; your model uses groups=1
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        # weight shape: (out, in, kH, kW)
        new_conv.weight.copy_(conv.weight.data[:, keep_in_idx].contiguous())
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias.data.contiguous())

    return new_conv


def rebuild_fc_after_pruning(model: nn.Module, example_input: torch.Tensor) -> None:
    """
    Rebuild model.fc input dim based on the current conv/ssrp path.
    Assumes model has attributes: conv1, conv2, conv3, ssrp_ms, flatten, fc.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = example_input.to(device)
        feats = model.flatten(model.ssrp_ms(model.conv3(model.conv2(model.conv1(x)))))
        in_dim = feats.shape[1]

    old_out = model.fc.out_features
    model.fc = nn.Linear(in_dim, old_out).to(device)
    model.train()

def apply_structural_pruning(model: nn.Module, amount: float = 0.8, example_input: torch.Tensor = torch.randn(1, 1, 40, 862)) -> nn.Module:
    """
    Structural channel pruning for your CNN_PCAw_SSRPMS_KAN conv blocks.

    - Prunes conv1 out channels + bn1, updates conv2 input channels accordingly
    - Prunes conv2 out channels + bn2, updates conv3 input channels accordingly
    - Prunes conv3 out channels + bn3
    - Optionally rebuilds fc using example_input

    example_input should be shaped like your model input, e.g. (1, 1, F, T)
    """
    device = next(model.parameters()).device

    # ---- conv1 prune (conv1[1] is Conv2d, conv1[2] is BN) ----
    conv1_old = model.conv1[1]
    bn1_old = model.conv1[2]
    conv1_new, bn1_new, keep1 = prune_conv_bn_pair(conv1_old, bn1_old, amount)

    model.conv1[1] = conv1_new
    model.conv1[2] = bn1_new

    # ---- conv2 input prune to match conv1 kept outputs ----
    model.conv2[1] = prune_conv_input_channels(model.conv2[1], keep1)

    # ---- conv2 prune ----
    conv2_old = model.conv2[1]
    bn2_old = model.conv2[2]
    conv2_new, bn2_new, keep2 = prune_conv_bn_pair(conv2_old, bn2_old, amount)

    model.conv2[1] = conv2_new
    model.conv2[2] = bn2_new

    # ---- conv3 input prune to match conv2 kept outputs ----
    model.conv3[0] = prune_conv_input_channels(model.conv3[0], keep2)

    # ---- conv3 prune ----
    conv3_old = model.conv3[0]
    bn3_old = model.conv3[1]
    conv3_new, bn3_new, keep3 = prune_conv_bn_pair(conv3_old, bn3_old, amount)

    model.conv3[0] = conv3_new
    model.conv3[1] = bn3_new

    # Ensure the whole model stays on the same device
    model.to(device)

    # ---- fc rebuild (needed because flatten dim changes) ----
    if example_input is not None:
        rebuild_fc_after_pruning(model, example_input)

    return model
