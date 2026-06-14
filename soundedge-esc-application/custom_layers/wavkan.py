import math
from itertools import pairwise

import torch
from torch import nn

from .wavelets import DoGW, mexican_hat, meyer, morlet, shannon, unknown


class WavKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type="dog"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        # Parameters for wavelet transformation
        self.scale = nn.Parameter(torch.ones(out_features, in_features))
        self.translation = nn.Parameter(torch.zeros(out_features, in_features))
        self.wavelet_weights = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.kaiming_uniform_(self.wavelet_weights, a=math.sqrt(5))

        # Base activation function #not used for this experiment
        self.base_activation = nn.SiLU()

        # Batch normalization
        self.bn = nn.BatchNorm1d(out_features)

    def wavelet_transform(self, x):
        if x.dim() == 2:
            x_expanded = x.unsqueeze(1)
        else:
            x_expanded = x

        translation_expanded = self.translation.unsqueeze(0).expand(x.size(0), -1, -1)
        scale_expanded = self.scale.unsqueeze(0).expand(x.size(0), -1, -1)
        x_scaled = (x_expanded - translation_expanded) / scale_expanded

        wavelet_method = self._methods.get(self.wavelet_type, unknown)
        wavelet_output = wavelet_method(x_scaled, self.wavelet_weights)

        return wavelet_output

    def forward(self, x):
        return self.bn(self.wavelet_transform(x))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"wavelet_type={self.wavelet_type}"
        )

    @property
    def _methods(self):
        return {
            "mexican_hat": mexican_hat,
            "morlet": morlet,
            "dog": DoGW,
            "meyer": meyer,
            "shannon": shannon,
        }


class WavKAN(nn.Module):
    def __init__(self, layers_hidden, wavelet_type="dog"):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.5)
        for in_features, out_features in pairwise(layers_hidden):
            self.layers.append(WavKANLinear(in_features, out_features, wavelet_type))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = self.dropout(x)
        return x
