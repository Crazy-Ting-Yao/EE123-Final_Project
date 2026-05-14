"""Lifting-scheme wavelet transforms for the Final Project.

Phase 1 — DSP Core.

Public API
----------
lifting_forward / lifting_inverse
    1D LeGall 5/3 or Haar transform along an arbitrary axis (default ``haar``).

dwt2d_forward / dwt2d_inverse
    Single-level 2D DWT (rows then columns) returning (LL, LH, HL, HH).

dwt2d_multilevel / idwt2d_multilevel
    J-level dyadic decomposition applied recursively to the LL sub-band.
"""

from .lifting import lifting_forward, lifting_inverse
from .dwt2d import (
    dwt2d_forward,
    dwt2d_inverse,
    dwt2d_multilevel,
    idwt2d_multilevel,
)
from .quantization import QuantConfig, quantize_bands, dequantize_bands
from .scanning import scan2d, unscan2d
from .priority_packetizer import (
    Packet,
    Packetization,
    packetize_wavelet,
    coeffs_from_received_packets,
    rebuild_bands_from_coeffs,
    decode_packets_to_image,
)
from .channel import BurstConfig, simulate_packet_loss, tier_stats

__all__ = [
    "lifting_forward",
    "lifting_inverse",
    "dwt2d_forward",
    "dwt2d_inverse",
    "dwt2d_multilevel",
    "idwt2d_multilevel",
    "QuantConfig",
    "quantize_bands",
    "dequantize_bands",
    "scan2d",
    "unscan2d",
    "Packet",
    "Packetization",
    "packetize_wavelet",
    "coeffs_from_received_packets",
    "rebuild_bands_from_coeffs",
    "decode_packets_to_image",
    "BurstConfig",
    "simulate_packet_loss",
    "tier_stats",
]
