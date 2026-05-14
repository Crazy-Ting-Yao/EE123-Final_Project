3042012028 Huang, Tingyao

# Title: Priority-Encoded Wavelet Compression for Packet-Lossy Channels

# Abstract:

This project implements a traditional Discrete Wavelet Transform image compression system using the Lifting Scheme to address the instability of bitstreams in packet-based radio transmission. Unlike block-based DCT (JPEG) which suffers from catastrophic synchronization loss, DWT allows for multi-resolution sub-band coding.

The core novelty lies in a Priority-Aware Bit-Allocation strategy: critical low-frequency approximation coefficients (LL sub-band) are protected with higher redundancy/priority, while high-frequency detail coefficients (HH, LH, HL) are packetized for "best-effort" delivery. The system is evaluated by simulating non-stationary radio interference and measuring graceful degradation using PSNR and SSIM metrics.

# Intended End Results
- A Modular Software Codebase 
    -   Encoder: Implements the Lifting Scheme (Split, Predict, Update) for 2D Discrete Wavelet Transform.
    -   Quantizer: A dead-zone or uniform scalar quantizer to reduce bit-depth.
    -   Priority Packetizer: A custom script that organizes wavelet coefficients into discrete packets, tagging "Approximation" (LL) data as high-priority and "Detail" (LH, HL, HH) data as low-priority.
    -   Radio Channel Simulator: A module that simulates packet loss (Random or Burst) based on a configurable probability Ploss​.
    -   Decoder: A robust inverse DWT that can handle missing packets by performing Error Concealment (e.g., zero-padding or interpolation of missing coefficients).
- Experimental Dataset & Metrics:
    -   Visual Samples: Side-by-side comparisons of my method vs. standard JPEG under identical packet loss conditions (e.g., 5%, 10%, 20%).
    -   Performance Curves: Plots showing PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity) vs. Packet Loss Rate.
    -   Rate-Distortion Analysis: A table comparing the compression ratio achieved against the resulting image quality.

# Relevant Research Papers

These papers provide the mathematical foundation for wavelets and the logic behind priority-based transmission in lossy environments.

"The Lifting Scheme: A Construction of Second Generation Wavelets" by Wim Sweldens (1998):
This is the seminal paper on the "Lifting Scheme." It explains how to implement DWT without the complex convolution of traditional filters, making it perfect in efficient hardware/circuit implementation.

"Robust Image Transmission Performed by SPIHT and Turbo-Codes" by Rogers and Cosman (1998/Updated 2026 contexts):
While "SPIHT" is a specific wavelet algorithm, this paper is the gold standard for understanding how to protect wavelet bitstreams over noisy channels. It discusses how the loss of synchronization in wavelet trees affects the final image.

"Error-Resilient Image Coding and Transmission over Wireless Channels" (ISCAS 2002 / IEEE):
This paper directly addresses the focus on packet-based radio. It discusses "Channel-Optimized" wavelet coding, where image information is spread across frequency bands to minimize the impact of packet loss.

# Timeline
## Phase 1: The DSP Core (Week 1)
### Goal: Get the image into the wavelet domain and back without losing data.
[x] Implement 1D Lifting: Write a function for the Split, Predict, and Update steps using the LeGall 5/3 or Haar filters.
[x] Expand to 2D DWT: Apply the 1D transform to rows, then to columns.
[x] Multi-level Decomposition: Implement a recursive loop to perform the transform on the LL sub-band multiple times (e.g., 3 levels).
[x] Inverse Transform: Implement the reverse lifting steps and verify that Input Image - Reconstructed Image ≈ 0.

Phase 1 deliverables (this folder):
- `wavelet/lifting.py` — 1D LeGall 5/3 and Haar lifting (Split / Predict / Update) with symmetric boundary extension. **Default wavelet for the pipeline is Haar**; use `wavelet="legall"` for LeGall 5/3.
- `wavelet/dwt2d.py` — separable 2D DWT (rows then columns) plus multi-level decomposition that recurses on the LL sub-band.
- `test_phase1.py` — round-trip checks (1D, single-level 2D, 3-level 2D, structured image); maximum reconstruction error is ~1e-13 (floating-point round-off).
- `visualize_dwt.py` — renders the canonical sub-band pyramid alongside the input and the reconstruction (`phase1_pyramid.png`).
Run `python test_phase1.py` to reproduce the round-trip verification.

## Phase 2: Compression & Priority Logic (Week 2)
### Goal: Turn coefficients into a prioritized bitstream.
[x] Sub-band Quantization: Create a quantizer that applies a smaller step size to the LL band (high precision) and a larger step size to the HH/LH/HL bands (lower precision).
[x] Coefficient Zig-Zag/Scanning: Order the coefficients so that the most important (LL) come first (then detail bands), with per-band scan support (`row_major` or `zigzag`).
[x] Priority Packetization: Divide the quantized coefficient stream into fixed-size packets and label packets containing LL data as Tier 1 (Critical), others as Tier 2 (Best Effort).

Phase 2 deliverables (this folder):
- `wavelet/quantization.py` — LL/Detail quantizer (uniform or dead-zone).
- `wavelet/scanning.py` — zig-zag / row-major 2D coefficient scanning + inverse.
- `wavelet/priority_packetizer.py` — packetization with LL-first ordering + (future) error-concealment baseline (missing packets -> zeros).
- `test_phase2.py` — verifies packetization/unpacketization correctness.
## Phase 3: The Radio Channel & Error Handling (Week 3)
### Goal: Simulate the "Packet-based Radio" environment.
[x] Channel Simulator: Write a script that takes your packets and randomly "drops" them based on a probability $P$.
[x] Simulate Burst Loss: (Optional Novelty) Create a model where packets are dropped in groups, simulating a temporary fade in radio signal.
[x] Error Concealment: Update your decoder to handle missing packets. If a detail packet is missing, fill that area with zeros (zero-padding).
[x] Baseline Setup: Implement a basic Block-DCT (JPEG-style) transmitter to serve as your comparison baseline.

Phase 3 deliverables (this folder):
- `wavelet/channel.py` — packet loss models: `random` and bursty `gilbert_elliott`.
- `wavelet/priority_packetizer.py` — decoding already supports concealment via missing packets -> zero coefficients.
- `baseline_dct.py` — 8×8 Block-DCT baseline with quantization + packet loss reconstruction.
- `test_phase3.py` — runs wavelet vs DCT through random/burst loss and confirms both reconstruct.
## Phase 4: Benchmarking & Visualization (Week 4)
### Goal: Prove that your method works better than the traditional way.
[x] Metric Implementation: Write scripts to calculate PSNR and SSIM for the reconstructed images.
[x] Data Generation: Run your simulation across a range of loss rates (0% to 30%).
[x] Plotting: Generate a plot of SSIM vs. Packet Loss Rate comparing Wavelet-Priority vs. JPEG-Baseline.
[x] Final Visuals: Export side-by-side images showing how your method stays "recognizable" while the baseline "breaks" under high noise.

Phase 4 deliverables (this folder):
- `metrics.py` — PSNR + SSIM (pure numpy). For RGB, `ssim_rgb_mean()` reports the mean SSIM over R/G/B channels.
- `benchmark_phase4.py` — generates per-image CSVs, a mean CSV, SSIM-vs-loss plot, and side-by-side PNGs.
- `prepare_grayscale.py` — optional luma conversion to `grayscale/*_gray.png`.
- `wavelet/ecc_repetition.py` — repetition-code expansion for Tier-1 logical packets (erasure channel, uniform `p_loss` per physical packet).

Run (grayscale, default synthetic if no `--images`):
- `python benchmark_phase4.py`
- `python benchmark_phase4.py --images grayscale/kodim03_gray.png ... --tier1_copies 3`
- Default lifting wavelet is **Haar**; use `--wavelet legall` for LeGall 5/3.
- **Tier-1 protection** uses a **repetition code** (`--tier1_copies`, default 3): each Tier-1 *logical* packet is sent that many times; the channel applies the **same** `p_loss` to every *physical* packet. Use `--tier1_copies 1` for no redundancy (fair airtime vs. Tier 2 only).

Run (full-color RGB: logical received indices from the R channel are reused for G/B):
- `python benchmark_phase4.py --rgb --images kodim/kodim03.png kodim/kodim05.png ... --tier1_copies 3 --out_dir outputs_rgb`

Outputs (grayscale): `outputs/phase4_metrics_mean.csv`, `outputs/phase4_ssim_vs_loss_mean.png`, `outputs/phase4_side_by_side_<name>_00pct.png`, …

Outputs (RGB): `outputs_rgb/phase4_metrics_mean_rgb.csv`, `outputs_rgb/phase4_ssim_vs_loss_mean_rgb.png`, `outputs_rgb/phase4_side_by_side_<name>_rgb_10pct.png`, …



