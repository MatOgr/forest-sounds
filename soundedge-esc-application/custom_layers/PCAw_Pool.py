import torch
import torch.nn as nn
import torch.nn.functional as F

class PCAw_Pool(nn.Module):
    def __init__(self, kernel_size, stride=(1, 1), eps: float = 1e-4, normalize_weights: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.eps = eps
        self.normalize_weights = normalize_weights

        # D = number of features per patch = F_k * T_k
        F_k, T_k = kernel_size
        D = F_k * T_k

        # Trainable weights over PCA components (columns). Shape: (D,)
        # Initialized small to avoid overpowering early training.
        self.weights = nn.Parameter(0.01 * torch.randn(D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, Fh, Tw = x.shape
        F_k, T_k = self.kernel_size

        # Unfold per-channel
        x_bc = x.view(B * C, 1, Fh, Tw)  # (B*C, 1, F, T)
        patches = F.unfold(x_bc, kernel_size=(F_k, T_k), stride=self.stride)  # (B*C, D, NumPatches)
        D = patches.shape[1]
        NumPatches = patches.shape[2]

        # Arrange to (B, C, NumPatches, D) -> (B, C*NumPatches, D)
        patches = patches.permute(0, 2, 1).contiguous()         # (B*C, NumPatches, D)
        patches = patches.view(B, C, NumPatches, D)             # (B, C, NumPatches, D)
        X = patches.view(B, C * NumPatches, D)                  # (B, N, D) with N = C*NumPatches
        N = X.shape[1]

        # Output spatial size
        H = (Fh - F_k) // self.stride[0] + 1
        W = (Tw - T_k) // self.stride[1] + 1

        # Center features
        mean = X.mean(dim=1, keepdim=True)                     # (B, 1, D)
        Xc = X - mean                                          # (B, N, D)

        # Optional: standardize per-feature to tame scale explosions from DSConv
        var = Xc.pow(2).mean(dim=1, keepdim=True)              # (B, 1, D)
        Xc = Xc / torch.sqrt(var + 1e-6)

        # --------- Stable projection basis via SVD (more robust than eigh) ---------
        # Xc = U S V^T  -> principal components are columns of V
        # Use full_matrices=False for efficiency and stable backward
        U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)   # Vh: (B, D, D)
        eigvecs = Vh.transpose(1, 2)                           # (B, D, D)

        # Detach eigenvectors to avoid backprop through SVD
        eigvecs = eigvecs.detach()

        # Weighted projection direction v = E @ w
        if self.normalize_weights:
            w = torch.softmax(self.weights, dim=0)             # (D,)
        else:
            w = self.weights

        v = torch.matmul(eigvecs, w)                           # (B, D)
        v = v / (v.norm(dim=1, keepdim=True) + 1e-8)           # normalize

        # Project samples onto v -> scalar per sample
        scores = torch.matmul(Xc, v.unsqueeze(-1)).squeeze(-1) # (B, N)

        return scores.view(B, C, H, W)

    def extra_repr(self) -> str:
        return (f"kernel_size={self.kernel_size}, stride={self.stride}, "
                f"eps={self.eps}, normalize_weights={self.normalize_weights}")