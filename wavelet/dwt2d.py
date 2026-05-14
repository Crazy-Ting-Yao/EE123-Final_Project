"""2D Discrete Wavelet Transform via the Lifting Scheme.

The 2-D transform is *separable*: a 1-D lifting transform is applied
along the rows of the image, and then along the columns of the two
intermediate row-bands.  This produces four sub-bands per level::

    LL : row-low,  col-low    -- coarse approximation
    LH : row-low,  col-high   -- horizontal detail (vertical edges)
    HL : row-high, col-low    -- vertical detail   (horizontal edges)
    HH : row-high, col-high   -- diagonal detail

The multi-level (J-level) decomposition is obtained by recursively
applying the single-level transform to the LL sub-band.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .lifting import lifting_forward, lifting_inverse


SubBand = np.ndarray
DetailLevel = Tuple[SubBand, SubBand, SubBand]   # (LH, HL, HH)


# ---------------------------------------------------------------------------
# Single-level 2D DWT
# ---------------------------------------------------------------------------

def dwt2d_forward(image, wavelet: str = "haar"):
    """One level of 2D DWT.

    Parameters
    ----------
    image : array_like, shape (H, W)
        Both ``H`` and ``W`` must be even.
    wavelet : {"legall", "haar"}

    Returns
    -------
    (LL, LH, HL, HH) : tuple of ndarrays of shape (H/2, W/2)
    """
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {image.shape}")
    H, W = image.shape
    if H % 2 or W % 2:
        raise ValueError(
            f"image dimensions must be even for one DWT level, got {(H, W)}"
        )

    # Step 1: transform along rows  (axis=1)
    L_row, H_row = lifting_forward(image, wavelet=wavelet, axis=1)

    # Step 2: transform along columns of each row-band  (axis=0)
    LL, LH = lifting_forward(L_row, wavelet=wavelet, axis=0)
    HL, HH = lifting_forward(H_row, wavelet=wavelet, axis=0)

    return LL, LH, HL, HH


def dwt2d_inverse(LL, LH, HL, HH, wavelet: str = "haar"):
    """Inverse of :func:`dwt2d_forward` (one level)."""
    L_row = lifting_inverse(LL, LH, wavelet=wavelet, axis=0)
    H_row = lifting_inverse(HL, HH, wavelet=wavelet, axis=0)
    image = lifting_inverse(L_row, H_row, wavelet=wavelet, axis=1)
    return image


# ---------------------------------------------------------------------------
# Multi-level (J-level) decomposition
# ---------------------------------------------------------------------------

def dwt2d_multilevel(
    image,
    levels: int = 3,
    wavelet: str = "haar",
) -> Tuple[SubBand, List[DetailLevel]]:
    """J-level dyadic DWT applied recursively to the LL sub-band.

    The image is required to have height and width divisible by
    ``2 ** levels``.

    Returns
    -------
    LL : ndarray
        The coarsest approximation sub-band, shape
        ``(H / 2**levels, W / 2**levels)``.
    details : list of (LH, HL, HH)
        Length ``levels``.  ``details[0]`` holds the *finest* detail
        sub-bands (largest spatial size); ``details[-1]`` holds the
        *coarsest* details (same size as ``LL``).
    """
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}")

    image = np.asarray(image, dtype=np.float64)
    H, W = image.shape
    factor = 1 << levels
    if H % factor or W % factor:
        raise ValueError(
            f"image shape {(H, W)} must be divisible by 2**levels = {factor}"
        )

    details: List[DetailLevel] = []
    LL = image
    for _ in range(levels):
        LL, LH, HL, HH = dwt2d_forward(LL, wavelet=wavelet)
        details.append((LH, HL, HH))
    return LL, details


def idwt2d_multilevel(
    LL: SubBand,
    details: List[DetailLevel],
    wavelet: str = "haar",
):
    """Inverse of :func:`dwt2d_multilevel`."""
    image = LL
    # Re-combine from the coarsest level outwards.
    for LH, HL, HH in reversed(details):
        image = dwt2d_inverse(image, LH, HL, HH, wavelet=wavelet)
    return image
