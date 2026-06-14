import torch


def mexican_hat(X: torch.Tensor, wavelet_weights: torch.Tensor) -> torch.Tensor:
    """Edge and peak detector.

    2nd derivative of Gaussian.
    """
    term1 = (X**2) - 1
    term2 = torch.exp(-0.5 * X**2)
    wavelet = (2 / 3**0.5 * torch.pi**0.25) * term1 * term2
    wavelet_weighted = wavelet * wavelet_weights.unsqueeze(0).expand_as(wavelet)
    wavelet_output = wavelet_weighted.sum(dim=2)
    return wavelet_output


def DoGW(X: torch.Tensor, wavelet_weights: torch.Tensor) -> torch.Tensor:
    """
    First derivative of a Gaussian distribution (localized gradient → temporal differentiator).

        $$\\psi(t) = -t \\cdot e^{-\frac{t^2}{2}}$$

    Ignores static/continuous sounds, fires on fast quiet-loud transitionc
    Asymmetric → shifts freqs by 90°
    """
    # Implementing Derivative of Gaussian Wavelet
    dog = -X * torch.exp(-0.5 * X**2)
    wavelet = dog
    wavelet_weighted = wavelet * wavelet_weights.unsqueeze(0).expand_as(wavelet)
    wavelet_output = wavelet_weighted.sum(dim=2)
    return wavelet_output


def morlet(X: torch.Tensor, wavelet_weights: torch.Tensor) -> torch.Tensor:
    """
    Pitch tracking & std spectrograms.

    Creates Localized Wave Packet. General time-freq analysis, pitch detection, sustained musical notes analysis.
    """
    omega0 = 5.0  # Central frequency
    real = torch.cos(omega0 * X)
    envelope = torch.exp(-0.5 * X**2)
    wavelet = envelope * real
    wavelet_weighted = wavelet * wavelet_weights.unsqueeze(0).expand_as(wavelet)
    wavelet_output = wavelet_weighted.sum(dim=2)
    return wavelet_output


def meyer(X: torch.Tensor, wavelet_weights: torch.Tensor) -> torch.Tensor:
    """
    HQ decomposition & reconstruction.

    Smooth frquency block, orthogonal wavelet → easy (de)composition.
    """
    # Implement Meyer Wavelet here
    # Constants for the Meyer wavelet transition boundaries
    v = torch.abs(X)
    pi = torch.pi

    def meyer_aux(v):
        return torch.where(
            v <= 1 / 2,
            torch.ones_like(v),
            torch.where(v >= 1, torch.zeros_like(v), torch.cos(pi / 2 * nu(2 * v - 1))),
        )

    def nu(t):
        return t**4 * (35 - 84 * t + 70 * t**2 - 20 * t**3)

    # Meyer wavelet calculation using the auxiliary function
    wavelet = torch.sin(pi * v) * meyer_aux(v)
    wavelet_weighted = wavelet * wavelet_weights.unsqueeze(0).expand_as(wavelet)
    wavelet_output = wavelet_weighted.sum(dim=2)
    return wavelet_output


def shannon(X: torch.Tensor, wavelet_weights: torch.Tensor) -> torch.Tensor:
    """
    Stric & aggressive frequency filtering.

    Sharp-cutoff bandpass / square filter.
    """
    # Windowing the sinc function to limit its support
    pi = torch.pi
    sinc = torch.sinc(X / pi)  # sinc(x) = sin(pi*x) / (pi*x)

    # Applying a Hamming window to limit the infinite support of the sinc function
    window = torch.hamming_window(
        X.size(-1), periodic=False, dtype=X.dtype, device=X.device
    )
    # Shannon wavelet is the product of the sinc function and the window
    wavelet = sinc * window
    wavelet_weighted = wavelet * wavelet_weights.unsqueeze(0).expand_as(wavelet)
    wavelet_output = wavelet_weighted.sum(dim=2)
    return wavelet_output
    # You can try many more wavelet types ...


def unknown(**kwargs):
    raise ValueError("Unsupported wavelet type")
