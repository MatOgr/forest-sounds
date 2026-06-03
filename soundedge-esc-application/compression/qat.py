import copy
import torch
import torch.nn as nn
import torch.ao.quantization as tq


class CNN_PSK_QATWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.m = base_model
        self.quant_in_conv3 = tq.QuantStub()
        self.dequant_after_conv3 = tq.DeQuantStub()
        self.quant_in_fc = tq.QuantStub()
        self.dequant_after_fc = tq.DeQuantStub()

    def forward(self, x):
        x = self.m.conv1(x)          # float
        x = self.m.conv2(x)          # float

        x = self.quant_in_conv3(x)   # int8-sim
        x = self.m.conv3(x)          # quantized block
        x = self.dequant_after_conv3(x)

        x = self.m.ssrp_ms(x)        # float
        x = self.m.flatten(x)

        x = self.quant_in_fc(x)      # int8-sim
        x = self.m.fc(x)             # quantized linear
        x = self.dequant_after_fc(x)

        x = self.m.kan(x)            # float
        return x


def _get_available_backend(preferred: str = "fbgemm") -> str:
    supported = torch.backends.quantized.supported_engines
    if preferred in supported:
        return preferred
    for fallback in ("qnnpack", "fbgemm", "onednn"):
        if fallback in supported:
            return fallback
    raise RuntimeError(
        f"No supported quantization backend found. Available: {supported}"
    )


def _set_backend(backend: str):
    backend = _get_available_backend(backend)
    torch.backends.quantized.engine = backend
    return backend


def _fuse_conv_bn_relu_for_qat(base: nn.Module):
    if not (hasattr(base, "conv3") and isinstance(base.conv3, nn.Sequential)):
        return
    if len(base.conv3) < 3:
        return

    was_training = base.training
    base.eval()
    tq.fuse_modules(base.conv3, [["0", "1", "2"]], inplace=True)
    if was_training:
        base.train()

def get_qat_qconfig_compatible(backend: str = "fbgemm"):
    """
    QAT config compatible with eager-mode convert() for Conv/Linear:
      - Activations: quint8 per-tensor affine
      - Weights: qint8 per-channel symmetric (supported by quantized conv/linear)
    """
    backend = _get_available_backend(backend)

    act_fq = tq.FusedMovingAvgObsFakeQuantize.with_args(
        observer=tq.MovingAverageMinMaxObserver,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        quant_min=0,
        quant_max=255,
        reduce_range=False,
    )

    weight_fq = tq.FusedMovingAvgObsFakeQuantize.with_args(
        observer=tq.MovingAveragePerChannelMinMaxObserver,
        dtype=torch.qint8,
        qscheme=torch.per_channel_symmetric,   # key change
        quant_min=-128,
        quant_max=127,
        reduce_range=False,
        ch_axis=0,  # Conv2d out_channels axis, Linear out_features axis
    )

    return tq.QConfig(activation=act_fq, weight=weight_fq)


def prepare_qat_model(
    model_fp32: nn.Module,
    backend: str = "fbgemm",
    inplace: bool = False,
) -> nn.Module:
    backend = _set_backend(backend)

    base = model_fp32 if inplace else copy.deepcopy(model_fp32)

    # 1) Fuse conv3 safely
    _fuse_conv_bn_relu_for_qat(base)

    # 2) Wrap
    qat_wrapped = CNN_PSK_QATWrapper(base)

    # 3) Attach QAT qconfig
    qat_wrapped.qconfig = tq.get_default_qat_qconfig(backend)  # backend already resolved

    # Keep custom / unsupported parts in float
    qat_wrapped.m.conv1.qconfig = None
    qat_wrapped.m.conv2.qconfig = None
    qat_wrapped.m.ssrp_ms.qconfig = None
    qat_wrapped.m.flatten.qconfig = None
    qat_wrapped.m.kan.qconfig = None

    # Keep PCAw_Pool float (inside conv1)
    for mod in qat_wrapped.m.conv1.modules():
        if mod.__class__.__name__ == "PCAw_Pool":
            mod.qconfig = None

    # 4) Prepare for QAT (inserts fake quant modules)
    qat_wrapped.train()
    tq.prepare_qat(qat_wrapped, inplace=True)

    return qat_wrapped


def convert_qat_model(
    qat_model: nn.Module,
    backend: str = "fbgemm",
    inplace: bool = False,
) -> nn.Module:
    _set_backend(backend)  # will auto-resolve to supported backend
    m = qat_model if inplace else copy.deepcopy(qat_model)
    m.eval()
    m = m.cpu()
    return tq.convert(m, inplace=True)
