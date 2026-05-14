from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Tuple

import numpy as np


QuantMode = Literal["uniform", "deadzone"]


@dataclass(frozen=True)
class QuantConfig:
    """Quantization configuration for Phase 2.

    Phase 2 requirement:
      - smaller step for LL (higher precision)
      - larger step for detail bands (lower precision)
    """

    step_ll: float
    step_detail: float
    mode: QuantMode = "deadzone"
    deadzone_width: float = 0.5

    def __post_init__(self) -> None:
        if self.step_ll <= 0:
            raise ValueError(f"step_ll must be > 0, got {self.step_ll}")
        if self.step_detail <= 0:
            raise ValueError(
                f"step_detail must be > 0, got {self.step_detail}"
            )
        if not (0.0 <= self.deadzone_width <= 1.0):
            raise ValueError(
                f"deadzone_width must be in [0, 1], got {self.deadzone_width}"
            )


def quantize_scalar(
    x: np.ndarray,
    step: float,
    *,
    mode: QuantMode = "deadzone",
    deadzone_width: float = 0.5,
) -> np.ndarray:
    """Scalar quantize to integers.

    Returns int array q where dequantized value is q * step.
    """
    x = np.asarray(x, dtype=np.float64)
    if step <= 0:
        raise ValueError(f"step must be > 0, got {step}")

    if mode == "uniform":
        q = np.round(x / step).astype(np.int32)
        return q

    if mode == "deadzone":
        # Dead-zone means values smaller than threshold snap to 0.
        # Threshold in the units of the original signal.
        threshold = deadzone_width * step
        absx = np.abs(x)
        q = np.zeros_like(x, dtype=np.int32)
        mask = absx >= threshold
        q[mask] = np.round(x[mask] / step).astype(np.int32)
        return q

    raise ValueError(f"unknown mode {mode!r} (expected 'uniform' or 'deadzone')")


def dequantize_scalar(q: np.ndarray, step: float) -> np.ndarray:
    """Inverse of :func:`quantize_scalar`."""
    q = np.asarray(q)
    return q.astype(np.float64) * float(step)


def quantize_bands(
    LL: np.ndarray,
    details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: QuantConfig,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Quantize LL and all detail bands to integer coefficients."""
    q_LL = quantize_scalar(
        LL, config.step_ll, mode=config.mode, deadzone_width=config.deadzone_width
    )
    q_details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for LH, HL, HH in details:
        q_LH = quantize_scalar(
            LH,
            config.step_detail,
            mode=config.mode,
            deadzone_width=config.deadzone_width,
        )
        q_HL = quantize_scalar(
            HL,
            config.step_detail,
            mode=config.mode,
            deadzone_width=config.deadzone_width,
        )
        q_HH = quantize_scalar(
            HH,
            config.step_detail,
            mode=config.mode,
            deadzone_width=config.deadzone_width,
        )
        q_details.append((q_LH, q_HL, q_HH))
    return q_LL, q_details


def dequantize_bands(
    q_LL: np.ndarray,
    q_details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    config: QuantConfig,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Dequantize integer coefficients back to floating point."""
    LL = dequantize_scalar(q_LL, config.step_ll)
    details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for q_LH, q_HL, q_HH in q_details:
        LH = dequantize_scalar(q_LH, config.step_detail)
        HL = dequantize_scalar(q_HL, config.step_detail)
        HH = dequantize_scalar(q_HH, config.step_detail)
        details.append((LH, HL, HH))
    return LL, details

