from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from baseline_dct import dct_decode_packets, dct_packetize_image
from metrics import psnr, ssim, ssim_rgb_mean
from wavelet import QuantConfig, dwt2d_multilevel, decode_packets_to_image, packetize_wavelet
from wavelet.ecc_repetition import expand_tier1_repetition, logical_received_from_physical


def _stable_name_hash(name: str) -> int:
    """Deterministic small int from a string (avoid PYTHONHASHSEED issues)."""
    h = 0
    for ch in name:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return int(h % 100000)


def _make_benchmark_image(shape=(256, 256)) -> np.ndarray:
    """Deterministic benchmark image (no external dataset needed)."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    ramp = xx / (W - 1) * 255.0
    texture = 25.0 * np.sin(2 * np.pi * xx / 16.0) + 18.0 * np.cos(2 * np.pi * yy / 21.0)
    img = 0.72 * ramp + texture + 78.0

    # Add a high-frequency checkered patch to highlight packet-loss artefacts
    patch = (((xx // 6 + yy // 6) % 2) * 255.0)
    mask = (xx > W * 0.62) & (yy > H * 0.62)
    img = np.where(mask, 0.35 * img + 0.65 * patch, img)

    return np.clip(img, 0.0, 255.0)


def _load_grayscale_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64)


def _load_rgb_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64)


def _crop_to_factor(img: np.ndarray, factor: int) -> np.ndarray:
    H, W = img.shape[0], img.shape[1]
    H2 = (H // factor) * factor
    W2 = (W // factor) * factor
    if H2 <= 0 or W2 <= 0:
        raise ValueError(f"image too small for factor {factor}: {img.shape}")
    if img.ndim == 2:
        return img[:H2, :W2]
    if img.ndim == 3:
        return img[:H2, :W2, :]
    raise ValueError(f"expected 2D or 3D image, got shape {img.shape}")


def _ssim_for_image(ref: np.ndarray, x: np.ndarray) -> float:
    if ref.ndim == 2:
        return ssim(ref, x)
    return ssim_rgb_mean(ref, x)


def _encode_rgb(
    ref: np.ndarray,
    *,
    levels: int,
    wavelet: str,
    packet_size: int,
) -> tuple:
    """Return per-channel (wave_pktz, dct_pktz) tuples."""
    qcfg = QuantConfig(step_ll=1.0, step_detail=3.0, mode="deadzone", deadzone_width=0.5)
    wave_list = []
    dct_list = []
    for c in range(3):
        ch = ref[..., c]
        LL, details = dwt2d_multilevel(ch, levels=levels, wavelet=wavelet)
        wave_list.append(
            packetize_wavelet(
                LL,
                details,
                quant_config=qcfg,
                scan_method="zigzag",
                packet_size_coeffs=packet_size,
            )
        )
        dct_list.append(
            dct_packetize_image(
                ch,
                block_size=8,
                quant_scale=1.5,
                scan_method="zigzag",
                packet_size_coeffs=packet_size,
            )
        )
    return tuple(wave_list), tuple(dct_list)


def _drop_packets_indices(n_packets: int, p_loss: float, rng: np.random.Generator) -> List[int]:
    """Random i.i.d packet drops; return received indices."""
    keep = rng.random(n_packets) >= p_loss
    return [i for i, ok in enumerate(keep) if ok]


def _wavelet_logical_received_after_channel(
    wave_pktz,
    *,
    p_loss: float,
    rng: np.random.Generator,
    tier1_copies: int,
) -> List[int]:
    """Same :math:`p` on every *physical* packet; Tier 1 uses repetition ECC.

    Returns sorted logical packet indices that survive erasure (any replica
    of a logical Tier-1 packet is enough).
    """
    physical = expand_tier1_repetition(wave_pktz.packets, tier1_copies)
    rec_phys = set(_drop_packets_indices(len(physical), float(p_loss), rng))
    return logical_received_from_physical(physical, rec_phys)


def _to_display_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def save_side_by_side(
    ref: np.ndarray,
    w: np.ndarray,
    d: np.ndarray,
    *,
    out_path: Path,
    title: str,
) -> None:
    """Save a single PNG with 3 panels (ref / wavelet / dct)."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5))
    for ax in axes:
        ax.axis("off")

    if ref.ndim == 3:
        axes[0].imshow(_to_display_u8(ref))
        axes[1].imshow(_to_display_u8(w))
        axes[2].imshow(_to_display_u8(d))
    else:
        axes[0].imshow(ref, cmap="gray", vmin=0, vmax=255)
        axes[1].imshow(w, cmap="gray", vmin=0, vmax=255)
        axes[2].imshow(d, cmap="gray", vmin=0, vmax=255)

    axes[0].set_title("Reference")
    axes[1].set_title("Wavelet-Priority")
    axes[2].set_title("Block-DCT baseline")

    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Dict[str, float]]) -> np.ndarray:
    header = "loss_rate,wavelet_psnr,wavelet_ssim,dct_psnr,dct_ssim"
    data = np.array(
        [[r[h] for h in header.split(",")] for r in rows],
        dtype=np.float64,
    )
    np.savetxt(path, data, delimiter=",", header=header, comments="")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 4 benchmarking + visualization")
    ap.add_argument("--min_loss", type=float, default=0.0)
    ap.add_argument("--max_loss", type=float, default=0.30)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260506)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument(
        "--images",
        nargs="*",
        default=[],
        help="Optional image paths (grayscale unless --rgb). If omitted, uses a synthetic test image.",
    )
    ap.add_argument(
        "--side_by_side_image",
        default=None,
        help="Which image to export side-by-side visuals for (path). Defaults to the first of --images.",
    )
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--packet_size", type=int, default=512)
    ap.add_argument("--wavelet", default="haar", choices=["legall", "haar"])
    ap.add_argument(
        "--tier1_copies",
        type=int,
        default=3,
        help="Repetition-code replicas for each Tier-1 *logical* packet on the wire. "
        "Every physical packet (Tier-1 replicas and Tier-2 singles) is erased i.i.d. "
        "with the same p_loss. Use 1 to disable Tier-1 redundancy.",
    )
    ap.add_argument(
        "--rgb",
        action="store_true",
        help="Process RGB (3 channels). Logical received indices from channel 0 "
        "are reused for R/G/B (same colour artefacts).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine dataset
    image_paths: List[Path] = [Path(p) for p in args.images] if args.images else []
    use_synthetic = len(image_paths) == 0

    # The common divisibility constraint:
    # - multi-level DWT needs divisibility by 2**levels
    # - block DCT baseline needs divisibility by 8
    factor = int(np.lcm(1 << int(args.levels), 8))

    if use_synthetic:
        gray = _make_benchmark_image((256, 256))
        if args.rgb:
            rgb = np.stack([gray, gray, gray], axis=-1)
            dataset = [("synthetic_rgb", rgb)]
        else:
            dataset = [("synthetic", gray)]
    else:
        dataset = []
        for p in image_paths:
            if args.rgb:
                img = _crop_to_factor(_load_rgb_image(p), factor)
            else:
                img = _crop_to_factor(_load_grayscale_image(p), factor)
            dataset.append((p.stem, img))

    loss_rates = np.arange(args.min_loss, args.max_loss + 1e-12, args.step, dtype=np.float64)

    # Run per-image benchmarks and also accumulate mean curves.
    all_data: List[np.ndarray] = []
    for name, ref in dataset:
        rows: List[Dict[str, float]] = []

        if ref.ndim == 3:
            wave_pktzs, dct_pktzs = _encode_rgb(
                ref,
                levels=args.levels,
                wavelet=args.wavelet,
                packet_size=args.packet_size,
            )
            for p in loss_rates:
                rng = np.random.default_rng(
                    int(args.seed + _stable_name_hash(name) + round(p * 10000))
                )
                rec_w_idx = _wavelet_logical_received_after_channel(
                    wave_pktzs[0],
                    p_loss=float(p),
                    rng=rng,
                    tier1_copies=int(args.tier1_copies),
                )
                rec_d_idx = _drop_packets_indices(
                    len(dct_pktzs[0].packets), float(p), rng
                )

                w_ch = [
                    decode_packets_to_image(
                        wave_pktzs[c], rec_w_idx, wavelet=args.wavelet
                    )
                    for c in range(3)
                ]
                d_ch = [dct_decode_packets(dct_pktzs[c], rec_d_idx) for c in range(3)]
                w_img = np.stack(w_ch, axis=-1)
                d_img = np.stack(d_ch, axis=-1)

                rows.append(
                    {
                        "loss_rate": float(p),
                        "wavelet_psnr": psnr(ref, w_img),
                        "wavelet_ssim": _ssim_for_image(ref, w_img),
                        "dct_psnr": psnr(ref, d_img),
                        "dct_ssim": _ssim_for_image(ref, d_img),
                    }
                )
        else:
            LL, details = dwt2d_multilevel(ref, levels=args.levels, wavelet=args.wavelet)
            qcfg = QuantConfig(
                step_ll=1.0, step_detail=3.0, mode="deadzone", deadzone_width=0.5
            )
            wave_pktz = packetize_wavelet(
                LL,
                details,
                quant_config=qcfg,
                scan_method="zigzag",
                packet_size_coeffs=args.packet_size,
            )
            dct_pktz = dct_packetize_image(
                ref,
                block_size=8,
                quant_scale=1.5,
                scan_method="zigzag",
                packet_size_coeffs=args.packet_size,
            )

            for p in loss_rates:
                rng = np.random.default_rng(
                    int(args.seed + _stable_name_hash(name) + round(p * 10000))
                )
                rec_w_idx = _wavelet_logical_received_after_channel(
                    wave_pktz,
                    p_loss=float(p),
                    rng=rng,
                    tier1_copies=int(args.tier1_copies),
                )
                rec_d_idx = _drop_packets_indices(len(dct_pktz.packets), float(p), rng)

                w_img = decode_packets_to_image(
                    wave_pktz, rec_w_idx, wavelet=args.wavelet
                )
                d_img = dct_decode_packets(dct_pktz, rec_d_idx)

                rows.append(
                    {
                        "loss_rate": float(p),
                        "wavelet_psnr": psnr(ref, w_img),
                        "wavelet_ssim": ssim(ref, w_img),
                        "dct_psnr": psnr(ref, d_img),
                        "dct_ssim": ssim(ref, d_img),
                    }
                )

        suffix = "_rgb" if ref.ndim == 3 else ""
        per_csv = out_dir / f"phase4_metrics_{name}{suffix}.csv"
        data = _write_csv(per_csv, rows)
        all_data.append(data)

    # Mean curve across images
    if len(all_data) == 1:
        mean_data = all_data[0]
    else:
        stack = np.stack(all_data, axis=0)  # (K, M, 5)
        mean_data = np.mean(stack, axis=0)

    mean_suffix = "_rgb" if args.rgb else ""
    csv_path = out_dir / f"phase4_metrics_mean{mean_suffix}.csv"
    np.savetxt(
        csv_path,
        mean_data,
        delimiter=",",
        header="loss_rate,wavelet_psnr,wavelet_ssim,dct_psnr,dct_ssim",
        comments="",
    )

    # Plot SSIM vs loss
    plot_path = out_dir / f"phase4_ssim_vs_loss_mean{mean_suffix}.png"
    plt.figure(figsize=(7.2, 4.8))
    plt.plot(mean_data[:, 0] * 100, mean_data[:, 2], label="Wavelet-Priority (SSIM)")
    plt.plot(mean_data[:, 0] * 100, mean_data[:, 4], label="Block-DCT baseline (SSIM)")
    plt.xlabel("Packet loss rate (%)")
    plt.ylabel("SSIM")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Side-by-side visuals (pick one image)
    if use_synthetic:
        sbs_name, sbs_ref = dataset[0]
    else:
        if args.side_by_side_image is None:
            sbs_name, sbs_ref = dataset[0]
        else:
            p = Path(args.side_by_side_image)
            sbs_name = p.stem
            if args.rgb:
                sbs_ref = _crop_to_factor(_load_rgb_image(p), factor)
            else:
                sbs_ref = _crop_to_factor(_load_grayscale_image(p), factor)

    if sbs_ref.ndim == 3:
        wave_pktzs, dct_pktzs = _encode_rgb(
            sbs_ref,
            levels=args.levels,
            wavelet=args.wavelet,
            packet_size=args.packet_size,
        )
        for p in (0.0, 0.10, 0.20, 0.30):
            if p < args.min_loss - 1e-12 or p > args.max_loss + 1e-12:
                continue
            rng = np.random.default_rng(
                int(args.seed + _stable_name_hash(sbs_name) + round(p * 10000))
            )
            rec_w_idx = _wavelet_logical_received_after_channel(
                wave_pktzs[0],
                p_loss=float(p),
                rng=rng,
                tier1_copies=int(args.tier1_copies),
            )
            rec_d_idx = _drop_packets_indices(len(dct_pktzs[0].packets), float(p), rng)
            w_img = np.stack(
                [
                    decode_packets_to_image(
                        wave_pktzs[c], rec_w_idx, wavelet=args.wavelet
                    )
                    for c in range(3)
                ],
                axis=-1,
            )
            d_img = np.stack(
                [dct_decode_packets(dct_pktzs[c], rec_d_idx) for c in range(3)], axis=-1
            )

            title = (
                f"{sbs_name} | loss={p*100:.0f}% | "
                f"W: PSNR={psnr(sbs_ref, w_img):.2f} SSIM={_ssim_for_image(sbs_ref, w_img):.3f} | "
                f"DCT: PSNR={psnr(sbs_ref, d_img):.2f} SSIM={_ssim_for_image(sbs_ref, d_img):.3f}"
            )
            out_path = (
                out_dir
                / f"phase4_side_by_side_{sbs_name}_rgb_{int(p*100):02d}pct.png"
            )
            save_side_by_side(sbs_ref, w_img, d_img, out_path=out_path, title=title)
    else:
        LL, details = dwt2d_multilevel(sbs_ref, levels=args.levels, wavelet=args.wavelet)
        qcfg = QuantConfig(
            step_ll=1.0, step_detail=3.0, mode="deadzone", deadzone_width=0.5
        )
        wave_pktz = packetize_wavelet(
            LL,
            details,
            quant_config=qcfg,
            scan_method="zigzag",
            packet_size_coeffs=args.packet_size,
        )
        dct_pktz = dct_packetize_image(
            sbs_ref,
            block_size=8,
            quant_scale=1.5,
            scan_method="zigzag",
            packet_size_coeffs=args.packet_size,
        )

        for p in (0.0, 0.10, 0.20, 0.30):
            if p < args.min_loss - 1e-12 or p > args.max_loss + 1e-12:
                continue
            rng = np.random.default_rng(
                int(args.seed + _stable_name_hash(sbs_name) + round(p * 10000))
            )
            rec_w_idx = _wavelet_logical_received_after_channel(
                wave_pktz,
                p_loss=float(p),
                rng=rng,
                tier1_copies=int(args.tier1_copies),
            )
            rec_d_idx = _drop_packets_indices(len(dct_pktz.packets), float(p), rng)
            w_img = decode_packets_to_image(
                wave_pktz, rec_w_idx, wavelet=args.wavelet
            )
            d_img = dct_decode_packets(dct_pktz, rec_d_idx)

            title = (
                f"{sbs_name} | loss={p*100:.0f}% | "
                f"W: PSNR={psnr(sbs_ref, w_img):.2f} SSIM={ssim(sbs_ref, w_img):.3f} | "
                f"DCT: PSNR={psnr(sbs_ref, d_img):.2f} SSIM={ssim(sbs_ref, d_img):.3f}"
            )
            out_path = out_dir / f"phase4_side_by_side_{sbs_name}_{int(p*100):02d}pct.png"
            save_side_by_side(sbs_ref, w_img, d_img, out_path=out_path, title=title)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved per-image CSVs to: {out_dir}")
    print(f"Saved side-by-side images to: {out_dir}")


if __name__ == "__main__":
    main()

