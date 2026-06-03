import torch
import torch.nn as nn

from custom_layers.SSRP_MS import SSRP_MS
from custom_layers.WavKAN import WavKANLinear
from custom_layers.PCAw_Pool import PCAw_Pool


class CNN_PCAw_SSRPMS_KAN(nn.Module):
    def __init__(self, num_classes):
        super(CNN_PCAw_SSRPMS_KAN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(1, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            PCAw_Pool(kernel_size=(3, 3), stride=(3, 3)),
        )

        self.conv2 = nn.Sequential(
            nn.ZeroPad2d((0, 0, 0, 1)),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.ssrp_ms = SSRP_MS(base_window=3, num_levels=5)
        self.flatten = nn.Flatten()

        self.fc = nn.Linear(1024, 128)
        self.kan = WavKANLinear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.ssrp_ms(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.kan(x)
        return x