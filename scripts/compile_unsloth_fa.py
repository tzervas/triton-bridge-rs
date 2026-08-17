#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline-compile Unsloth Flash Attention Triton → PTX/CUBIN.

Run on the 5080 (needs torch + triton + unsloth). Do NOT run in CPU CI.

    CUDA_COMPUTE_CAP=90 python scripts/compile_unsloth_fa.py \
        --out precompiled/sm90_flash_fwd.ptx

Then add NOTICE (Unsloth is Apache-2.0) and a numerical gate in Rust.
This script does not vendor Unsloth .py files into git.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sm", type=int, default=int(os.environ.get("CUDA_COMPUTE_CAP", "90")))
    args = p.parse_args()

    try:
        import torch
        import triton
    except ImportError as e:
        print(f"FAIL_ENV: need torch+triton on a GPU host ({e})", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("FAIL_ENV: torch.cuda.is_available() is False", file=sys.stderr)
        return 2

    print(f"triton {triton.__version__} torch {torch.__version__} sm={args.sm}")
    print("Look up unsloth.kernels.flash_attention_2 (or current path).")
    print("Call triton.compile on the fwd kernel for this SM, write PTX/CUBIN.")
    print("Do not copy AGPL MoE kernels.")
    print(f"Would write: {args.out}")
    print("This scaffold stops before import unsloth so a missing install is not a crash.")
    print("GPU job: fill in the compile call once unsloth is on the workstation venv.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
