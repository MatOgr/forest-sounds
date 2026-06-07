from typing import cast

import torch.nn as nn
from custom_layers.PCAw_Pool import PCAw_Pool
from custom_layers.SSRP_MS import SSRP_MS
from custom_layers.WavKAN import WavKANLinear
from typing_extensions import override


class CNN_PCAw_SSRPMS_KAN(nn.Module):
    def __init__(self, num_classes):
        super(CNN_PCAw_SSRPMS_KAN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(1, 64, kernel_size=3),
            nn.BatchNorm2d(64),  # TODO: check
            nn.ReLU(),  # TODO: check
            PCAw_Pool(kernel_size=(3, 3), stride=(3, 3)),
        )

        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),  # TODO: check
            nn.ReLU(),  # TODO: check
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.ssrp_ms = SSRP_MS(base_window=3, num_levels=5)
        # SSRP_MS yields (B, C=256, F); F varies with n_mels. Pool freq to a
        # fixed 4 so flatten is always 256*4=1024 regardless of mel bins ->
        # FC stays Linear(1024, 128) across any sample_rate/n_fft/n_mels sweep.
        # (F==4 at n_mels=40, so this is a no-op for the original config.)
        self.freq_pool = nn.AdaptiveAvgPool1d(4)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(1024, 128)
        self.kan = WavKANLinear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ssrp_ms(x)  # (B, 256, F)
        x = self.freq_pool(x)  # (B, 256, 4) -> fixed regardless of n_mels
        x = self.flatten(x)
        x = self.fc(x)
        x = self.kan(x)
        return x


class CNN_PCAw_SSRPMS_KAN_DDD(CNN_PCAw_SSRPMS_KAN):
    """CNN-PSK with 3-channel input."""

    @override
    def __init__(self, num_classes: int):
        super(CNN_PCAw_SSRPMS_KAN_DDD, self).__init__(num_classes)
        old = cast(nn.Conv2d, self.conv1[1])
        self.conv1[1] = nn.Conv2d(3, old.out_channels, old.kernel_size[0])
