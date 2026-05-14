from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def to_grayscale(in_path: Path) -> Image.Image:
    img = Image.open(in_path).convert("RGB")
    # Standard luminance conversion (ITU-R BT.601)
    arr = np.asarray(img, dtype=np.float64)
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    gray_u8 = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    return Image.fromarray(gray_u8, mode="L")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert images to grayscale (Luma).")
    ap.add_argument("--out_dir", default="grayscale", help="output directory")
    ap.add_argument("images", nargs="+", help="input image paths")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in args.images:
        in_path = Path(p)
        out_path = out_dir / f"{in_path.stem}_gray.png"
        g = to_grayscale(in_path)
        g.save(out_path)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

