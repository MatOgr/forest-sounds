import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from efficient_kan import KAN


# ==========================================
# 1. AUDIO PREPROCESSING PIPELINE
# ==========================================
class LogMelFrontend(nn.Module):
    """
    Transforms raw 1D waveforms into Log-Mel Spectrogram features
    using torchaudio as described in the paper's preprocessing layout.
    """

    def __init__(self, sample_rate=16_000, n_fft=1_024, hop_length=512, n_mels=64):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )

    def forward(self, x):
        # Input shape: (batch_size, samples) or (batch_size, 1, samples)
        if x.dim() == 3:
            x = x.squeeze(1)

        mel = self.mel_transform(x)  # Shape: (batch_size, n_mels, time)
        log_mel = torch.log(mel + 1e-6)

        # Add a channel dimension for 2D CNN input -> (batch_size, 1, n_mels, time)
        return log_mel.unsqueeze(1)


# ==========================================
# 2. ADVANCED POOLING STRATEGIES
# ==========================================
class PCAPooling2D(nn.Module):
    """
    Downsamples the feature maps by computing principal components localized
    across the pooling window, retaining principal variance over traditional pooling.

    Blocker stays: PCAPooling2D.svd (line 66) has no ONNX op → export raises on SVD node. Docstring flags it. To unblock pick one:
        1. Swap PCA pool → avg/max pool (retrain, simplest)
        2. Fixed-projection approx of 1st PC (precompute, no per-batch SVD)
        3. Run torch+torchaudio directly on UGV (if board has full PyTorch)

    """

    def __init__(self, kernel_size=2):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        b, c, h, w = x.shape
        kh = kw = self.kernel_size

        # Unfold the tensor into distinct sliding/non-overlapping blocks
        # Shape: (b, c, h_out, w_out, kh, kw)
        patches = x.unfold(2, kh, kh).unfold(3, kw, kw)
        h_out, w_out = patches.shape[2], patches.shape[3]

        # Flatten patches into vectors for PCA computation: (b * c * h_out * w_out, kh * kw)
        flat_patches = patches.contiguous().view(-1, kh * kw)

        # Center the data matrix
        mean = flat_patches.mean(dim=-1, keepdim=True)
        centered = flat_patches - mean

        # Compute Singular Value Decomposition (SVD) to obtain the 1st Principal Component
        # flat_patches design matrix is typically small (e.g. 2x2 = 4 features), making SVD fast
        _, _, V = torch.linalg.svd(centered, full_matrices=False)
        first_pc = V[:, :, 0]  # Extract highest variance layout (b*c*h_out*w_out, 1)

        # Project the patches down to a single representative score
        pooled = torch.bmm(centered.unsqueeze(1), first_pc.unsqueeze(2)).squeeze(-1)

        # Reconstruct spatial dimensions -> (b, c, h_out, w_out)
        return pooled.view(b, c, h_out, w_out)


class SparseSalientRegionPooling(nn.Module):
    """
    SSRP isolates sparse time-frequency clusters showing top tier power profiles,
    preventing non-salient noise from masking transient indicators.
    """

    def __init__(self, sparsity_ratio=0.3):
        super().__init__()
        self.sparsity_ratio = sparsity_ratio

    def forward(self, x):
        # Input shape: (batch_size, channels, height, width)
        b, c, h, w = x.shape
        flat_features = x.view(b, c, -1)  # (b, c, h * w)

        # Determine the cutoff index based on the targeted sparsity allocation
        k = max(1, int(h * w * self.sparsity_ratio))

        # Extract the values of the top-k most salient features per channel
        topk_values, _ = torch.topk(flat_features, k=k, dim=-1, largest=True)

        # Compute the average over only these highly salient target zones
        ssrp_descriptor = topk_values.mean(dim=-1)  # Shape: (b, c)
        return ssrp_descriptor


# ==========================================
# 3. CONVOLUTION-KAN HYBRID ARCHITECTURE
# ==========================================
class HybridConvKAN(nn.Module):
    """
    CNN-PSK architecture containing a standard CNN front-end,
    Advanced custom pooling routines, and a Kolmogorov-Arnold Network back-end.
    """

    def __init__(self, num_classes=50, n_mels=64):
        super().__init__()

        # Audio extraction configuration
        self.frontend = LogMelFrontend(n_mels=n_mels)

        # Front-End CNN Architecture block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pca_pool1 = PCAPooling2D(kernel_size=2)  # 64 -> 32

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pca_pool2 = PCAPooling2D(kernel_size=2)  # 32 -> 16

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Global Feature Aggregator Replacement
        self.ssrp = SparseSalientRegionPooling(sparsity_ratio=0.25)

        # Efficient KAN Classification Back-End Head
        # Note: input features equal the channel depth of the final CNN block (128)
        self.kan_classifier = KAN(
            layers_hidden=[128, 64, num_classes], grid_size=5, spline_order=3
        )

    def forward(self, x):
        # 1. Waveform to Log-Mel Transformation
        x = self.frontend(x)

        # 2. Convolution-PCA Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pca_pool1(x)

        # 3. Convolution-PCA Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pca_pool2(x)

        # 4. Deep Feature Extraction Stage
        x = F.relu(self.bn3(self.conv3(x)))

        # 5. Sparse Salient Global Readout
        x = self.ssrp(x)  # Outputs flat representations: (batch_size, 128)

        # 6. Non-linear classification via KAN
        logits = self.kan_classifier(x)
        return logits


# ==========================================
# 4. EDGE EXPORT (waveform -> logits, mel folded into graph)
# ==========================================
def export_onnx(
    model: nn.Module,
    path: str = "model.onnx",
    num_samples: int = 64_000,
    opset: int = 17,
    dynamic_time: bool = False,
):
    """Export the full waveform->logits graph for edge deployment (UGV).

    Mel front-end is already inside the model, so the edge runtime feeds raw
    PCM and reads class logits — no torchaudio dependency on-device.

    opset>=17 is REQUIRED: that's where ONNX gained the STFT op that
    torchaudio's MelSpectrogram lowers to.

    WARNING: PCAPooling2D uses torch.linalg.svd, which has NO ONNX op.
    Export will raise on the SVD node. Replace PCA pooling with an
    ONNX-friendly pool (avg/max, or a fixed-projection approx) before
    relying on this path, or run torch/torchaudio directly on the UGV.

    dynamic_time=False -> fixed input shape, lets the edge runtime
    (ONNX Runtime / TensorRT) autotune; faster, recommended for UGV.
    """
    model.eval()
    dummy = torch.randn(1, num_samples)
    dynamic_axes = {"wav": {1: "samples"}} if dynamic_time else None
    torch.onnx.export(
        model,
        dummy,
        path,
        opset_version=opset,
        input_names=["wav"],
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
    )
    print(f"-> exported ONNX graph: {path} (opset {opset})")
    return path


def verify_parity(
    model: nn.Module, onnx_path: str, num_samples: int = 64_000, tol: float = 1e-3
) -> float:
    """Assert exported ONNX matches torch logits on a random clip.

    MANDATORY after export: silent mel/param drift = distribution shift =
    accuracy drop with no error. Returns max abs diff."""
    import onnxruntime as ort  # optional dep, only needed for verification

    model.eval()
    wav = torch.randn(1, num_samples)
    with torch.no_grad():
        ref = model(wav).cpu().numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    out = sess.run(["logits"], {"wav": wav.numpy()})[0]
    diff = float(abs(ref - out).max())
    print(f"parity max|Δ| = {diff:.2e} (tol {tol:.0e})")
    assert diff < tol, f"ONNX parity FAILED: {diff} >= {tol}"
    return diff


# ==========================================
# 5. VERIFICATION PIPELINE EXAMPLES
# ==========================================
if __name__ == "__main__":
    # Initialize the model (Configured for ESC-50 class layout)
    model = HybridConvKAN(num_classes=50)

    # Calculate parameter footprints (~500k target)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Successfully constructed CNN-PSK Network architecture.")
    print(f"Total Trainable Parameter Allocation: {total_params:,}")

    # Simulate a batch of raw environmental audio clips
    # Batch size: 4, duration: 4 seconds at a 16kHz sampling frequency (64,000 samples)
    dummy_waveforms = torch.randn(4, 64_000)

    print("\nExecuting verification forward pass...")
    output_predictions = model(dummy_waveforms)

    print(f"Input Shape:  {dummy_waveforms.shape} (Batch size, Samples)")
    print(
        f"Output Shape: {output_predictions.shape} (Batch size, Prediction Categories)"
    )
