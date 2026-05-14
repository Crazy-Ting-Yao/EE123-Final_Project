from __future__ import annotations

import numpy as np


def psnr(ref: np.ndarray, x: np.ndarray, *, data_range: float = 255.0) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    mse = float(np.mean((ref - x) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10((float(data_range) ** 2) / mse)


def _gaussian_kernel_1d(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    if size % 2 != 1:
        raise ValueError("gaussian kernel size must be odd")
    r = size // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2.0 * sigma**2))
    k /= np.sum(k)
    return k


def _convolve_separable_reflect(img: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Separable 2D convolution with reflect padding (pure numpy)."""
    img = np.asarray(img, dtype=np.float64)
    r = k.size // 2

    # Convolve rows
    tmp = np.empty_like(img, dtype=np.float64)
    pad = np.pad(img, ((0, 0), (r, r)), mode="reflect")
    for i in range(img.shape[0]):
        tmp[i, :] = np.convolve(pad[i, :], k, mode="valid")

    # Convolve cols
    out = np.empty_like(img, dtype=np.float64)
    pad2 = np.pad(tmp, ((r, r), (0, 0)), mode="reflect")
    for j in range(img.shape[1]):
        out[:, j] = np.convolve(pad2[:, j], k, mode="valid")

    return out


def ssim(
    ref: np.ndarray,
    x: np.ndarray,
    *,
    data_range: float = 255.0,
    gaussian_size: int = 11,
    gaussian_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """SSIM (Wang et al.) for 2D grayscale images, pure numpy implementation."""
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if ref.shape != x.shape:
        raise ValueError(f"ssim shape mismatch: {ref.shape} vs {x.shape}")
    if ref.ndim != 2:
        raise ValueError("ssim expects 2D grayscale arrays")

    L = float(data_range)
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2

    g = _gaussian_kernel_1d(gaussian_size, gaussian_sigma)
    mu1 = _convolve_separable_reflect(ref, g)
    mu2 = _convolve_separable_reflect(x, g)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = _convolve_separable_reflect(ref * ref, g) - mu1_sq
    sigma2_sq = _convolve_separable_reflect(x * x, g) - mu2_sq
    sigma12 = _convolve_separable_reflect(ref * x, g) - mu1_mu2

    # Numerical guard: small negative due to round-off.
    sigma1_sq = np.maximum(sigma1_sq, 0.0)
    sigma2_sq = np.maximum(sigma2_sq, 0.0)

    num = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return float(np.mean(ssim_map))


def ssim_rgb_mean(
    ref: np.ndarray,
    x: np.ndarray,
    *,
    data_range: float = 255.0,
    gaussian_size: int = 11,
    gaussian_sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Mean SSIM over RGB channels (common simple extension to color)."""
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if ref.shape != x.shape:
        raise ValueError(f"ssim_rgb_mean shape mismatch: {ref.shape} vs {x.shape}")
    if ref.ndim != 3 or ref.shape[2] != 3:
        raise ValueError("ssim_rgb_mean expects (H, W, 3) RGB arrays")

    vals = []
    for c in range(3):
        vals.append(
            ssim(
                ref[..., c],
                x[..., c],
                data_range=data_range,
                gaussian_size=gaussian_size,
                gaussian_sigma=gaussian_sigma,
                k1=k1,
                k2=k2,
            )
        )
    return float(np.mean(vals))

