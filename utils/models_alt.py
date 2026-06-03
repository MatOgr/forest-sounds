import torch
import torch.nn as nn
import torch.nn.functional as F
from kan import KAN

from __main__ import LogMelFrontend, PCAPooling2D, SparseSalientRegionPooling


class HybridConvPyKAN(nn.Module):
    """
    CNN-PSK architecture alternative using the official 'pykan' package. (doi.org/10.3390/s24123749)
    """

    def __init__(self, num_classes=50, n_mels=64):
        super().__init__()

        # 1. Front-End Feature Extraction
        self.frontend = LogMelFrontend(n_mels=n_mels)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pca_pool1 = PCAPooling2D(kernel_size=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pca_pool2 = PCAPooling2D(kernel_size=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 2. Global Spatial Aggregator
        self.ssrp = SparseSalientRegionPooling(sparsity_ratio=0.25)

        # 3. Official pykan Classification Head
        # pykan expects a list structure defining width layers: [input_dim, hidden_dim, output_dim]
        # grid: number of grid intervals, k: spline order
        self.kan_classifier = KAN(width=[128, 64, num_classes], grid=5, k=3)

    def forward(self, x):
        # Apply the stereo-to-mono handling natively if forgotten in dataloader
        if x.dim() == 3 and x.shape[1] > 1:
            x = torch.mean(x, dim=1)

        # CNN Feature extraction
        x = self.frontend(x)
        x = self.pca_pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pca_pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))

        # Compress space using SSRP -> (batch_size, 128)
        x = self.ssrp(x)

        # Pass to official pykan layer
        # pykan treats inputs smoothly via structural symbolic transformations
        logits = self.kan_classifier(x)
        return logits


# Verification code snippet
if __name__ == "__main__":
    # Create pseudo stereo input data batch: (Batch, Channels=2, Samples=64000)
    fake_stereo_batch = torch.randn(2, 2, 64000)

    model = HybridConvPyKAN(num_classes=50)
    out = model(fake_stereo_batch)
    print("PyKAN model forward execution completed successfully.")
    print("Output Logits Tensor Shape:", out.shape)
