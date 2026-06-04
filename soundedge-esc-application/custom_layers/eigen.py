import torch


def basis_svd(Xc: torch.Tensor) -> torch.Tensor:
    """
    Original path: principal components via full SVD of Xc.
    Xc = U S V^T -> components are columns of V (descending). Robust but
    materializes U (B, N, D) and runs SVD over the huge N dim -> slow.
    """
    # linalg is unstable/unsupported under fp16 -> force fp32, no autocast.
    with torch.autocast(device_type=Xc.device.type, enabled=False):
        Xcf = Xc.float()
        _U, _S, Vh = torch.linalg.svd(Xcf, full_matrices=False)  # Vh: (B, D, D)
        eigvecs = Vh.transpose(1, 2)  # cols=comps, desc.
    return eigvecs.detach().to(Xc.dtype)  # no backprop through decomposition


def basis_optimized(Xc: torch.Tensor) -> torch.Tensor:
    """
    Fast path: right singular vectors of Xc == eigenvectors of (Xc^T Xc),
    a tiny D x D matrix (D = F_k*T_k). Avoids materializing U (B, N, D) and
    avoids SVD over the huge N dim -> ~40x speedup + big memory drop.

    NOT bitwise-equal to the SVD path: eigvecs have an arbitrary per-column
    sign/order convention, so the combined direction v differs. It is still
    a valid PCA basis -> safe for training from scratch (weights adapt), but
    do NOT mix with weights trained under the SVD path.
    """
    # linalg is unstable/unsupported under fp16 -> force fp32, no autocast.
    with torch.autocast(device_type=Xc.device.type, enabled=False):
        Xcf = Xc.float()
        cov = torch.matmul(Xcf.transpose(1, 2), Xcf)  # (B, D, D)
        # eigh: ascending eigenvalues, eigenvectors as columns. Flip to
        # descending to match the SVD component ordering.
        _, eigvecs = torch.linalg.eigh(cov)  # (B, D, D)
        eigvecs = torch.flip(eigvecs, dims=[-1])  # cols=comps, desc.
    return eigvecs.detach().to(Xc.dtype)
