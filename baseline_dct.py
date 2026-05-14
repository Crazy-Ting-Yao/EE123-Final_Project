from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from wavelet.scanning import scan2d, unscan2d


def _dct_matrix(N: int = 8) -> np.ndarray:
    """Orthonormal DCT-II transform matrix."""
    k = np.arange(N).reshape(-1, 1)
    n = np.arange(N).reshape(1, -1)
    C = np.cos(np.pi / N * (n + 0.5) * k)
    C[0, :] *= 1.0 / np.sqrt(N)
    C[1:, :] *= np.sqrt(2.0 / N)
    return C


_C8 = _dct_matrix(8)


def dct2(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block, dtype=np.float64)
    return _C8 @ block @ _C8.T


def idct2(coeff: np.ndarray) -> np.ndarray:
    coeff = np.asarray(coeff, dtype=np.float64)
    return _C8.T @ coeff @ _C8


JPEG_LUMA_Q = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ],
    dtype=np.float64,
)


def _scale_quant_table(q: float) -> np.ndarray:
    """Scale a base quant table by q (bigger q => more aggressive)."""
    q = float(q)
    if q <= 0:
        raise ValueError("q must be > 0")
    return JPEG_LUMA_Q * q


@dataclass(frozen=True)
class DCTPacket:
    index: int
    start: int
    end: int
    payload: np.ndarray  # 1D int coeffs


@dataclass(frozen=True)
class DCTPacketization:
    image_shape: Tuple[int, int]
    block_size: int
    packet_size_coeffs: int
    scan_method: str
    quant_scale: float
    packets: Tuple[DCTPacket, ...]
    total_len: int


def dct_packetize_image(
    image: np.ndarray,
    *,
    block_size: int = 8,
    quant_scale: float = 1.0,
    scan_method: str = "zigzag",
    packet_size_coeffs: int = 1024,
) -> DCTPacketization:
    """JPEG-style block DCT baseline transmitter (no priority tiers)."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("baseline DCT expects a 2D grayscale image")
    H, W = img.shape
    if H % block_size or W % block_size:
        raise ValueError("image shape must be divisible by block_size")
    if block_size != 8:
        raise ValueError("only 8x8 supported for baseline")
    if packet_size_coeffs <= 0:
        raise ValueError("packet_size_coeffs must be > 0")

    Q = _scale_quant_table(quant_scale)

    vecs: List[np.ndarray] = []
    for y in range(0, H, block_size):
        for x in range(0, W, block_size):
            blk = img[y : y + block_size, x : x + block_size] - 128.0
            c = dct2(blk)
            q = np.round(c / Q).astype(np.int32)
            vecs.append(scan2d(q, method=scan_method))

    coeffs = np.concatenate(vecs, axis=0).astype(np.int32, copy=False)
    total_len = int(coeffs.size)
    packet_count = int(np.ceil(total_len / packet_size_coeffs))
    packets: List[DCTPacket] = []
    for i in range(packet_count):
        start = i * packet_size_coeffs
        end = min(start + packet_size_coeffs, total_len)
        packets.append(
            DCTPacket(index=i, start=start, end=end, payload=coeffs[start:end].copy())
        )

    return DCTPacketization(
        image_shape=(int(H), int(W)),
        block_size=int(block_size),
        packet_size_coeffs=int(packet_size_coeffs),
        scan_method=str(scan_method),
        quant_scale=float(quant_scale),
        packets=tuple(packets),
        total_len=total_len,
    )


def dct_decode_packets(
    pktz: DCTPacketization, received_packet_indices: Sequence[int]
) -> np.ndarray:
    """Decode with missing packets treated as zeros (baseline concealment)."""
    rec = set(int(i) for i in received_packet_indices)
    coeffs = np.zeros(pktz.total_len, dtype=np.int32)
    for p in pktz.packets:
        if p.index in rec:
            coeffs[p.start : p.end] = p.payload

    H, W = pktz.image_shape
    bs = pktz.block_size
    Q = _scale_quant_table(pktz.quant_scale)
    out = np.empty((H, W), dtype=np.float64)

    block_vec_len = bs * bs
    cursor = 0
    for y in range(0, H, bs):
        for x in range(0, W, bs):
            vec = coeffs[cursor : cursor + block_vec_len]
            cursor += block_vec_len
            q = unscan2d(vec, (bs, bs), pktz.scan_method)
            c = q.astype(np.float64) * Q
            blk = idct2(c) + 128.0
            out[y : y + bs, x : x + bs] = blk

    return out

