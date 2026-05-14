"""1D Lifting Scheme transforms.

Implements the canonical Split -> Predict -> Update pipeline of
Sweldens (1998) for two wavelets that are well-suited to image coding:

    * LeGall 5/3 (biorthogonal, used by JPEG2000 lossless mode)
    * Haar (1/1)

Both are written so that, in floating point, ``inverse(forward(x)) == x``
to round-off precision.  Symmetric (whole-sample) boundary extension is
used at both ends so that signals/images of any even length along the
transform axis are handled without artefacts.

The internal helpers operate along ``axis 0`` of an N-D array; the public
``lifting_forward`` / ``lifting_inverse`` wrappers move the requested
axis to the front, do the work, and move it back.  This keeps the maths
in one place and lets the 2-D code in :mod:`wavelet.dwt2d` simply call
the wrapper twice (once per spatial axis).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# LeGall 5/3 lifting
# ---------------------------------------------------------------------------
#
# Forward:
#     d[k] = x_o[k] - 1/2 * (x_e[k] + x_e[k+1])           (predict, high-pass)
#     s[k] = x_e[k] + 1/4 * (d[k-1] + d[k])               (update,  low-pass)
#
# Symmetric extension at the right boundary of "predict" uses
# x_e[K] = x_e[K-1], and at the left boundary of "update" uses d[-1] = d[0].

def _legall_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward LeGall 5/3 along axis 0.  ``x.shape[0]`` must be even."""
    N = x.shape[0]
    if N % 2 != 0:
        raise ValueError(f"length along transform axis must be even, got {N}")

    x_e = x[0::2].copy()
    x_o = x[1::2].copy()

    # Predict
    d = x_o.copy()
    d[:-1] -= 0.5 * (x_e[:-1] + x_e[1:])
    d[-1]  -= x_e[-1]                     # symmetric: x_e[K] = x_e[K-1]

    # Update
    s = x_e.copy()
    s[1:]  += 0.25 * (d[:-1] + d[1:])
    s[0]   += 0.5  * d[0]                 # symmetric: d[-1] = d[0]

    return s, d


def _legall_inverse(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Inverse LeGall 5/3 along axis 0."""
    if s.shape != d.shape:
        raise ValueError("approximation and detail must share the same shape")

    # Reverse update
    x_e = s.copy()
    x_e[1:] -= 0.25 * (d[:-1] + d[1:])
    x_e[0]  -= 0.5  * d[0]

    # Reverse predict
    x_o = d.copy()
    x_o[:-1] += 0.5 * (x_e[:-1] + x_e[1:])
    x_o[-1]  += x_e[-1]

    out = np.empty((2 * s.shape[0],) + s.shape[1:], dtype=np.float64)
    out[0::2] = x_e
    out[1::2] = x_o
    return out


# ---------------------------------------------------------------------------
# Haar lifting
# ---------------------------------------------------------------------------
#
# Forward:
#     d[k] = x_o[k] - x_e[k]
#     s[k] = x_e[k] + 1/2 * d[k]   ==  (x_e[k] + x_o[k]) / 2

def _haar_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    N = x.shape[0]
    if N % 2 != 0:
        raise ValueError(f"length along transform axis must be even, got {N}")
    x_e = x[0::2]
    x_o = x[1::2]
    d = x_o - x_e
    s = x_e + 0.5 * d
    return s.copy(), d.copy()


def _haar_inverse(s: np.ndarray, d: np.ndarray) -> np.ndarray:
    x_e = s - 0.5 * d
    x_o = d + x_e
    out = np.empty((2 * s.shape[0],) + s.shape[1:], dtype=np.float64)
    out[0::2] = x_e
    out[1::2] = x_o
    return out


# ---------------------------------------------------------------------------
# Axis-aware public wrappers
# ---------------------------------------------------------------------------

_WAVELETS = {
    "legall": (_legall_forward, _legall_inverse),
    "haar":   (_haar_forward,   _haar_inverse),
}


def _resolve(wavelet: str):
    try:
        return _WAVELETS[wavelet]
    except KeyError as exc:
        raise ValueError(
            f"unknown wavelet {wavelet!r}; "
            f"available: {sorted(_WAVELETS)}"
        ) from exc


def lifting_forward(x, wavelet: str = "haar", axis: int = -1):
    """Forward 1D lifting transform along ``axis``.

    Parameters
    ----------
    x : array_like
        Input signal.  ``x.shape[axis]`` must be even.
    wavelet : {"legall", "haar"}
        Lifting kernel.
    axis : int
        Axis along which to transform (default last).

    Returns
    -------
    (s, d) : ndarray, ndarray
        Approximation (low-pass) and detail (high-pass) coefficients,
        each of length ``x.shape[axis] // 2`` along ``axis``.
    """
    fwd, _ = _resolve(wavelet)
    x = np.asarray(x, dtype=np.float64)
    moved = np.moveaxis(x, axis, 0)
    s, d = fwd(moved)
    return np.moveaxis(s, 0, axis), np.moveaxis(d, 0, axis)


def lifting_inverse(s, d, wavelet: str = "haar", axis: int = -1):
    """Inverse of :func:`lifting_forward` along ``axis``."""
    _, inv = _resolve(wavelet)
    s = np.moveaxis(np.asarray(s, dtype=np.float64), axis, 0)
    d = np.moveaxis(np.asarray(d, dtype=np.float64), axis, 0)
    out = inv(s, d)
    return np.moveaxis(out, 0, axis)
