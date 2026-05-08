"""
Pull the four Open Model Zoo IRs we depend on into ./models/.

Uses the omz_downloader CLI from openvino-dev. Downloads FP16 by default
(small, fast, fine for CPU). Pass --precisions FP16,FP32 if you want both
so the benchmark can compare them.

Run:
    python download_models.py
    python download_models.py --precisions FP16,FP32
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODELS = [
    "face-detection-adas-0001",
    "facial-landmarks-35-adas-0002",
    "head-pose-estimation-adas-0001",
    "open-closed-eye-0001",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="models")
    ap.add_argument("--precisions", default="FP16",
                    help="Comma-separated list, e.g. 'FP16' or 'FP16,FP32'")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if shutil.which("omz_downloader") is None:
        print("error: omz_downloader is not on PATH.", file=sys.stderr)
        print("  install it with:  pip install openvino-dev", file=sys.stderr)
        return 1

    failed: list[str] = []
    for name in MODELS:
        print(f"\n>> downloading {name}")
        cmd = [
            "omz_downloader",
            "--name", name,
            "--output_dir", str(out),
            "--precisions", args.precisions,
        ]
        rc = subprocess.call(cmd)
        if rc != 0:
            failed.append(name)

    if failed:
        print("\n!! failed: " + ", ".join(failed), file=sys.stderr)
        return 2
    print("\nall models downloaded ok ->", out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
