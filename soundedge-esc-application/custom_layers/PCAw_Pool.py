from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .eigen import basis_optimized, basis_svd

SpectrogramTensor = Float[Tensor, "B C Fh Tw"]


class PCAw_Pool(nn.Module):
    def __init__(
        self,
        kernel_size,
        stride=(1, 1),
        eps: float = 1e-4,
        normalize_weights: bool = True,
        method: str = "optimized",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.normalize_weights = normalize_weights
        # Method to use for calculating Eigen vectors
        self.method = method

        # D = number of features per patch = F_k * T_k
        F_k, T_k = kernel_size
        D = F_k * T_k

        # Trainable weights over PCA components (columns). Shape: (D,)
        # Initialized small to avoid overpowering early training.
        self.weights = nn.Parameter(0.01 * torch.randn(D))

    def forward(self, x: SpectrogramTensor) -> torch.Tensor:
        # Fh: FreqBins (axis height), Tw: TimeFrames (axis width)
        B, C, Fh, Tw = x.shape
        F_k, T_k = self.kernel_size

        X = self._unfold_and_arrange(x)

        # Output spatial size
        H = (Fh - F_k) // self.stride[0] + 1
        W = (Tw - T_k) // self.stride[1] + 1

        # Center features
        mean = X.mean(dim=1, keepdim=True)  # (B, 1, D)
        Xc = X - mean  # (B, N, D)

        # Optional: standardize per-feature to tame scale explosions from DSConv
        var = Xc.pow(2).mean(dim=1, keepdim=True)  # (B, 1, D)
        Xc = Xc / torch.sqrt(var + 1e-6)

        scores = self._get_projection(Xc)
        return scores.view(B, C, H, W)

    def _unfold_and_arrange(self, X: torch.Tensor) -> torch.Tensor:
        B, C, Fh, Tw = X.shape
        # Unfold per-channel
        x_bc = X.view(B * C, 1, Fh, Tw)  # (B*C, 1, F, T)
        patches = F.unfold(
            x_bc,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )  # (B*C, D, NumPatches)
        D = patches.shape[1]
        NumPatches = patches.shape[2]

        # Arrange to (B, C, NumPatches, D) -> (B, C*NumPatches, D)
        patches = patches.permute(0, 2, 1).contiguous()  # (B*C, NumPatches, D)
        patches = patches.view(B, C, NumPatches, D)  # (B, C, NumPatches, D)
        arranged_patches = patches.view(
            B, C * NumPatches, D
        )  # (B, N, D) with N = C*NumPatches
        _N = arranged_patches.shape[1]
        return arranged_patches

    def _get_projection(self, Xc: torch.Tensor) -> torch.Tensor:
        # Projection basis (eigenvectors of Xc, columns = components, descending).
        eigvecs = self._eigen_vecs(Xc)

        # Weighted projection direction v = E @ w
        w = self._get_weights()

        v = torch.matmul(eigvecs, w)  # (B, D)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)  # normalize

        # Project samples onto v -> scalar per sample
        scores = torch.matmul(Xc, v.unsqueeze(-1)).squeeze(-1)  # (B, N)
        return scores

    def _get_weights(self) -> torch.Tensor:
        if self.normalize_weights:
            w = torch.softmax(self.weights, dim=0)  # (D,)
        else:
            w = self.weights
        return w

    def _eigen_vecs(self, Xc: torch.Tensor) -> torch.Tensor:
        eigvecs = self._methods[self.method](Xc)
        return eigvecs

    @property
    def _methods(self) -> dict[str, Callable]:
        return {
            "optimized": basis_optimized,
            "base": basis_svd,
        }

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"eps={self.eps}, normalize_weights={self.normalize_weights}"
        )
