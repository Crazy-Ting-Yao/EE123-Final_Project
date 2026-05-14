from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Sequence, Tuple

import numpy as np

from .dwt2d import idwt2d_multilevel
from .quantization import QuantConfig, dequantize_bands, quantize_bands
from .scanning import ScanMethod, scan2d, unscan2d


PacketTier = Literal[1, 2]


@dataclass(frozen=True)
class SegmentMeta:
    """A contiguous slice of the flattened coefficient stream."""

    kind: Literal["LL", "LH", "HL", "HH"]
    level_index: int
    shape: Tuple[int, int]
    start: int
    end: int
    tier: PacketTier


@dataclass(frozen=True)
class Packet:
    index: int
    start: int
    end: int
    tier: PacketTier
    payload: np.ndarray  # 1D int coefficients


@dataclass(frozen=True)
class Packetization:
    """Phase 2 priority packetization result.

    The payload is integer quantized coefficients (not yet bit-packed).
    Missing packets later will be treated as zeros for Error Concealment.
    """

    packet_size_coeffs: int
    total_len: int
    packets: Tuple[Packet, ...]
    segments: Tuple[SegmentMeta, ...]
    quant_config: QuantConfig
    scan_method: ScanMethod


def _build_segments_from_multilevel(
    LL: np.ndarray,
    details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[List[SegmentMeta], int]:
    """Define the global coefficient order used for packetization."""
    levels = len(details)
    segments: List[SegmentMeta] = []

    cursor = 0
    # 1) Coarsest approximation (LL) first => highest priority.
    ll_shape = tuple(int(x) for x in LL.shape)
    ll_len = int(ll_shape[0] * ll_shape[1])
    segments.append(
        SegmentMeta(
            kind="LL",
            level_index=levels - 1,
            shape=(ll_shape[0], ll_shape[1]),
            start=cursor,
            end=cursor + ll_len,
            tier=1,
        )
    )
    cursor += ll_len

    # 2) Then details from coarse->fine:
    #    details[levels-1], details[levels-2], ... details[0]
    #    for each level, order (LH, HL, HH).
    for lev in range(levels - 1, -1, -1):
        LH, HL, HH = details[lev]
        for kind, band in (("LH", LH), ("HL", HL), ("HH", HH)):
            shape = tuple(int(x) for x in band.shape)
            n = int(shape[0] * shape[1])
            segments.append(
                SegmentMeta(
                    kind=kind,
                    level_index=lev,
                    shape=(shape[0], shape[1]),
                    start=cursor,
                    end=cursor + n,
                    tier=2,
                )
            )
            cursor += n

    return segments, cursor


def packetize_wavelet(
    LL: np.ndarray,
    details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    quant_config: QuantConfig,
    scan_method: ScanMethod = "zigzag",
    packet_size_coeffs: int = 1024,
) -> Packetization:
    """Quantize + scan + priority packetize coefficients."""
    if packet_size_coeffs <= 0:
        raise ValueError(f"packet_size_coeffs must be > 0, got {packet_size_coeffs}")

    segments, total_len = _build_segments_from_multilevel(LL, details)
    q_LL, q_details = quantize_bands(LL, details, quant_config)

    # Build the full quantized coefficient stream in the same global order
    # defined by `segments`.
    coeffs = []
    for seg in segments:
        if seg.kind == "LL":
            band_q = q_LL
        else:
            q_LH, q_HL, q_HH = q_details[seg.level_index]
            if seg.kind == "LH":
                band_q = q_LH
            elif seg.kind == "HL":
                band_q = q_HL
            elif seg.kind == "HH":
                band_q = q_HH
            else:  # pragma: no cover
                raise RuntimeError(f"unexpected segment kind {seg.kind}")

        vec = scan2d(band_q, method=scan_method)
        if vec.size != (seg.end - seg.start):
            raise RuntimeError(
                f"segment size mismatch for {seg.kind}@{seg.level_index}: "
                f"vec={vec.size}, slice={seg.end - seg.start}"
            )
        coeffs.append(vec.astype(np.int32, copy=False))

    coeffs_vec = np.concatenate(coeffs, axis=0).astype(np.int32, copy=False)
    if coeffs_vec.size != total_len:
        raise RuntimeError(
            f"full coefficient stream size mismatch: got {coeffs_vec.size}, want {total_len}"
        )

    ll_ranges = [(s.start, s.end) for s in segments if s.kind == "LL"]

    def _intersects_ll(a: int, b: int) -> bool:
        # intersection of [a,b) with LL ranges.
        for s, e in ll_ranges:
            if not (e <= a or s >= b):
                return True
        return False

    # Split the stream into fixed-size packets.
    packets: List[Packet] = []
    packet_count = int(np.ceil(total_len / packet_size_coeffs))
    for p in range(packet_count):
        start = p * packet_size_coeffs
        end = min(start + packet_size_coeffs, total_len)
        tier: PacketTier = 1 if _intersects_ll(start, end) else 2
        payload = coeffs_vec[start:end].copy()
        packets.append(Packet(index=p, start=start, end=end, tier=tier, payload=payload))

    return Packetization(
        packet_size_coeffs=int(packet_size_coeffs),
        total_len=int(total_len),
        packets=tuple(packets),
        segments=tuple(segments),
        quant_config=quant_config,
        scan_method=scan_method,
    )


def coeffs_from_received_packets(
    packetization: Packetization,
    received_packet_indices: Sequence[int],
) -> np.ndarray:
    """Rebuild quantized coefficient stream with missing packets set to 0."""
    received = set(int(i) for i in received_packet_indices)
    coeffs_vec = np.zeros(packetization.total_len, dtype=np.int32)
    for pkt in packetization.packets:
        if pkt.index in received:
            coeffs_vec[pkt.start : pkt.end] = pkt.payload
    return coeffs_vec


def rebuild_bands_from_coeffs(
    packetization: Packetization,
    coeffs_vec: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Convert a quantized coefficient stream back to (q_LL, q_details)."""
    if coeffs_vec.size != packetization.total_len:
        raise ValueError(
            f"coeffs_vec has {coeffs_vec.size} elements but packetization expects {packetization.total_len}"
        )

    levels = max(seg.level_index for seg in packetization.segments) + 1

    q_LL: np.ndarray | None = None
    q_details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = [
        (None, None, None)  # type: ignore[list-item]
        for _ in range(levels)
    ]

    # Fill each segment back into its 2D band.
    for seg in packetization.segments:
        vec = coeffs_vec[seg.start : seg.end]
        band_q = unscan2d(vec, seg.shape, packetization.scan_method).astype(
            np.int32, copy=False
        )

        if seg.kind == "LL":
            q_LL = band_q
        else:
            q_LH, q_HL, q_HH = q_details[seg.level_index]
            if seg.kind == "LH":
                q_LH = band_q
            elif seg.kind == "HL":
                q_HL = band_q
            elif seg.kind == "HH":
                q_HH = band_q
            else:  # pragma: no cover
                raise RuntimeError(f"unexpected segment kind {seg.kind}")
            q_details[seg.level_index] = (q_LH, q_HL, q_HH)  # type: ignore[arg-type]

    if q_LL is None:
        raise RuntimeError("internal error: LL segment not found")
    # Type checker: we've ensured q_LL exists and q_details filled by segments.
    return q_LL, q_details  # type: ignore[return-value]


def decode_packets_to_image(
    packetization: Packetization,
    received_packet_indices: Sequence[int],
    *,
    wavelet: str = "haar",
) -> np.ndarray:
    """Decode received packets into an image using inverse DWT.

    Missing packets are treated as zero coefficients, which is the
    simplest Error Concealment baseline for Phase 3.
    """
    coeffs_vec = coeffs_from_received_packets(packetization, received_packet_indices)
    q_LL, q_details = rebuild_bands_from_coeffs(packetization, coeffs_vec)
    LL, details = dequantize_bands(q_LL, q_details, packetization.quant_config)
    return idwt2d_multilevel(LL, details, wavelet=wavelet)

