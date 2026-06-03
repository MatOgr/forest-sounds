import torch
from torch import nn
import torch.nn.functional as F

class SSRP_MS(nn.Module):
    """
    Multi-Scale Sparse Shift-Invariant Representation Pooling (SSRP-MS)

    Given feature maps x with shape (B, C, F, T):
      - For L scales with window sizes W_l = l * W0 (l = 1..L),
        compute moving means along time with stride=1 (no padding).
      - Take max over time at each scale (one max per frequency band).
      - Average these L maxima to obtain the pooled descriptor z_c(f).

    Output: (B, C, F)
    """
    def __init__(self, base_window: int, num_levels: int):
        super().__init__()
        assert base_window >= 1 and num_levels >= 1
        self.base_window = base_window
        self.num_levels  = num_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, F, T)
        returns: (B, C, F)
        """
        B, C, Freq, T = x.shape
        x_1d = x.reshape(B * C * Freq, 1, T)  # (BCF, 1, T)

        level_maxes = []
        # W_l = l * W0 (clamped to T to avoid empty pooling)
        for l in range(1, self.num_levels + 1):
            k = min(l * self.base_window, T)
            # avg over time windows of length k (stride 1)
            local_means = F.avg_pool1d(x_1d, kernel_size=k, stride=1)  # (BCF, 1, T - k + 1)
            # max over time (Eq. 5's inner max_t)
            max_over_t = local_means.max(dim=2)[0].squeeze(1)          # (BCF,)
            level_maxes.append(max_over_t)

        # Average across levels (1/L * sum_l ...)
        z = torch.stack(level_maxes, dim=0).mean(dim=0)                # (BCF,)

        return z.view(B, C, Freq)

    def extra_repr(self) -> str:
        return f"base_window={self.base_window}, num_levels={self.num_levels}"
