from __future__ import annotations

from typing import Literal, Tuple

import numpy as np


ScanMethod = Literal["row_major", "zigzag"]


def scan2d(band: np.ndarray, method: ScanMethod = "zigzag") -> np.ndarray:
    """Flatten 2D coefficients into a 1D scanning order."""
    band = np.asarray(band)
    if band.ndim != 2:
        raise ValueError(f"scan2d expects 2D input, got shape {band.shape}")

    H, W = band.shape
    if method == "row_major":
        return band.reshape(-1)
    if method == "zigzag":
        idx = _zigzag_indices(H, W)
        return band[idx[:, 0], idx[:, 1]]

    raise ValueError(f"unknown scan method {method!r}")


def unscan2d(vec: np.ndarray, shape: Tuple[int, int], method: ScanMethod) -> np.ndarray:
    """Inverse of :func:`scan2d`."""
    H, W = shape
    vec = np.asarray(vec)
    if vec.size != H * W:
        raise ValueError(
            f"unscan2d: vec has {vec.size} elements but target shape is {shape}"
        )

    out = np.empty(shape, dtype=vec.dtype)
    if method == "row_major":
        out[:] = vec.reshape(shape)
        return out
    if method == "zigzag":
        idx = _zigzag_indices(H, W)
        out[idx[:, 0], idx[:, 1]] = vec
        return out

    raise ValueError(f"unknown scan method {method!r}")


def _zigzag_indices(H: int, W: int) -> np.ndarray:
    """Standard zig-zag traversal (like JPEG) over an HxW grid."""
    coords = []
    for s in range(H + W - 1):
        # s is the anti-diagonal index: i + j = s.
        if s % 2 == 0:
            i_start = min(s, H - 1)
            i_end = max(0, s - (W - 1))
            for i in range(i_start, i_end - 1, -1):
                j = s - i
                if 0 <= j < W:
                    coords.append((i, j))
        else:
            i_start = max(0, s - (W - 1))
            i_end = min(s, H - 1)
            for i in range(i_start, i_end + 1):
                j = s - i
                if 0 <= j < W:
                    coords.append((i, j))
    idx = np.asarray(coords, dtype=np.int64)
    if idx.shape[0] != H * W:
        raise RuntimeError(
            f"zigzag index generation error: got {idx.shape[0]} indices, expected {H*W}"
        )
    return idx

