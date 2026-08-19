#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Offline-compile Unsloth Flash Attention Triton → PTX/CUBIN.

Run on the 5080 (needs torch + triton + unsloth). Do NOT run in CPU CI.

    CUDA_COMPUTE_CAP=90 python scripts/compile_unsloth_fa.py \
        --out precompiled/sm90_flash_fwd.ptx --sm 90

Then add NOTICE (Unsloth is Apache-2.0) and a numerical gate in Rust.
This script does not vendor Unsloth .py files into git.

Apache-2.0 kernels only. MoE / grouped-GEMM paths are skipped (AGPL).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


# Historical name first, then current Unsloth / zoo locations.
_FA_MODULES = (
    "unsloth.kernels.flash_attention_2",
    "unsloth.kernels.flash_attention",
    "unsloth.kernels.flash_attn",
    "unsloth.kernels.flash_attn_2",
    "unsloth.kernels.flex_attention",
    "unsloth_zoo.kernels.flash_attention_2",
    "unsloth_zoo.kernels.flash_attention",
    "unsloth_zoo.flex_attention",
)

_FA_NAME_HINTS = (
    "flash_fwd",
    "_fwd_kernel",
    "fwd_kernel",
    "flash_attention_forward",
    "_flash_attn_forward",
    "flash_attn_fwd",
    "_attn_fwd",
    "attention_forward",
)

_SKIP_NAME_FRAGMENTS = (
    "moe",
    "grouped_gemm",
    "gpt_oss",
    "grpo",
    "backward",
    "_bwd",
)


def _fail_env(msg: str) -> int:
    print(f"FAIL_ENV: {msg}", file=sys.stderr)
    return 2


def _is_jit(obj: Any) -> bool:
    try:
        from triton.runtime.jit import JITFunction
    except ImportError:
        return False
    return isinstance(obj, JITFunction)


def _looks_like_fa(name: str) -> bool:
    n = name.lower()
    if any(s in n for s in _SKIP_NAME_FRAGMENTS):
        return False
    return any(h in n for h in _FA_NAME_HINTS) or (
        "flash" in n and "attn" in n
    )


def _discover_kernel() -> tuple[Any, str] | None:
    """Return (JITFunction, dotted_path) for the first Apache-2.0 FA fwd kernel."""
    tried: list[str] = []
    for mod_name in _FA_MODULES:
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            tried.append(f"{mod_name} ({e})")
            continue
        for attr, obj in inspect.getmembers(mod):
            if not _is_jit(obj):
                continue
            if not _looks_like_fa(attr) and not _looks_like_fa(mod_name):
                continue
            path = f"{mod_name}.{attr}"
            print(f"found JIT kernel {path} args={list(obj.arg_names)}")
            return obj, path
        # Module imported but no JIT FA — still record it.
        jits = [n for n, o in inspect.getmembers(mod) if _is_jit(o)]
        tried.append(f"{mod_name} (imported; jit={jits or 'none'})")

    # Last chance: walk unsloth.kernels for any FA-named JIT.
    try:
        import unsloth.kernels as uk  # type: ignore[import-not-found]
    except ImportError:
        print("lookup tried:\n  " + "\n  ".join(tried), file=sys.stderr)
        return None
    pkg_dir = Path(getattr(uk, "__file__", "")).parent
    for py in sorted(pkg_dir.glob("*.py")):
        if any(s in py.name.lower() for s in _SKIP_NAME_FRAGMENTS):
            continue
        if "flash" not in py.name.lower() and "attn" not in py.name.lower():
            continue
        mod_name = f"unsloth.kernels.{py.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            tried.append(f"{mod_name} ({e})")
            continue
        for attr, obj in inspect.getmembers(mod):
            if _is_jit(obj) and _looks_like_fa(attr):
                path = f"{mod_name}.{attr}"
                print(f"found JIT kernel {path} args={list(obj.arg_names)}")
                return obj, path
        tried.append(f"{mod_name} (imported; no FA JIT)")

    print("lookup tried:\n  " + "\n  ".join(tried), file=sys.stderr)
    return None


def _default_constexpr(name: str, default: Any) -> Any:
    if default is not inspect.Parameter.empty:
        return default
    n = name.lower()
    if "causal" in n:
        return False
    if "block" in n:
        return 64
    if "head" in n or "dmodel" in n or "headdim" in n or n.endswith("_d"):
        return 64
    if n.startswith("is_") or n.startswith("has_"):
        return False
    return 64


def _runtime_ty(name: str) -> str:
    n = name.lower()
    if n in {"sm_scale", "scale", "softmax_scale"}:
        return "fp32"
    if any(
        key in n
        for key in (
            "stride",
            "seq",
            "n_ctx",
            "nctx",
            "batch",
            "num_",
            "n_heads",
            "nheads",
        )
    ):
        return "i32"
    if n in {"z", "h", "m", "n"}:
        return "i32"
    return "*fp32"


def _compile_jit(fn: Any, sm: int) -> Any:
    """triton.compile the discovered FA fwd kernel for `sm`."""
    import triton
    from triton.backends.compiler import GPUTarget

    fn.create_binder()
    signature: dict[str, str] = {}
    constexprs: dict[str, Any] = {}
    for param in fn.params:
        name = param.name
        if param.is_constexpr:
            constexprs[name] = _default_constexpr(name, param.default)
            signature[name] = "constexpr"
        else:
            signature[name] = _runtime_ty(name)

    print(f"triton.compile signature={signature}")
    print(f"triton.compile constexprs={constexprs}")
    src = fn.ASTSource(fn=fn, signature=signature, constexprs=constexprs, attrs={})
    target = GPUTarget("cuda", int(sm), 32)
    compiled = triton.compile(src, target=target)
    return compiled


def _write_payload(compiled: Any, out: Path, sm: int, kernel_path: str, host_sm: str) -> None:
    asm = getattr(compiled, "asm", {}) or {}
    ptx = asm.get("ptx")
    cubin = asm.get("cubin")
    if not ptx and not cubin:
        raise RuntimeError("triton.compile returned no ptx/cubin in compiled.asm")

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".cubin" and cubin:
        out.write_bytes(cubin)
        written = f"cubin {out} ({len(cubin)} bytes)"
    else:
        if not ptx:
            raise RuntimeError("asked for PTX but compiled.asm has no 'ptx'")
        text = ptx if isinstance(ptx, str) else ptx.decode("utf-8", "replace")
        if ".version" not in text:
            raise RuntimeError("compiled PTX missing .version (refusing to write)")
        out.write_text(text)
        written = f"ptx {out} ({len(text)} bytes)"

    meta = {
        "kernel_path": kernel_path,
        "entry": getattr(getattr(compiled, "metadata", None), "name", None),
        "sm_requested": sm,
        "host_sm": host_sm,
        "num_warps": getattr(getattr(compiled, "metadata", None), "num_warps", None),
        "shared": getattr(getattr(compiled, "metadata", None), "shared", None),
        "note": (
            "Host GPU is SM 12.0 (5080). sm90 PTX is a compile target, not a "
            "guarantee it is the right binary to launch on this device."
        ),
    }
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {written}")
    print(f"wrote metadata {meta_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sm", type=int, default=int(os.environ.get("CUDA_COMPUTE_CAP", "90")))
    args = p.parse_args()

    try:
        import torch
        import triton
    except ImportError as e:
        return _fail_env(f"need torch+triton on a GPU host ({e})")

    if not torch.cuda.is_available():
        return _fail_env("torch.cuda.is_available() is False")

    host_major, host_minor = torch.cuda.get_device_capability(0)
    host_sm = f"{host_major}.{host_minor}"
    print(f"triton {triton.__version__} torch {torch.__version__} sm={args.sm}")
    print(f"host gpu={torch.cuda.get_device_name(0)!r} compute_capability={host_sm}")
    if (host_major, host_minor) != (args.sm // 10, args.sm % 10) and args.sm != host_major * 10 + host_minor:
        print(
            f"HONEST: host SM is {host_sm}; compiling for sm{args.sm}. "
            "sm90 PTX may not be the right binary to launch on SM 12.0 (5080).",
            file=sys.stderr,
        )

    try:
        import unsloth  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as e:
        return _fail_env(
            "need unsloth in the workstation venv "
            f"(torch+triton present; unsloth missing: {e})"
        )

    found = _discover_kernel()
    if found is None:
        return _fail_env(
            "no Apache-2.0 Unsloth FA fwd JIT kernel found. "
            "Current unsloth.kernels has no flash_attention_2.py "
            "(flex_attention is torch.compile, not a Triton JIT). "
            "Refusing to invent PTX."
        )

    fn, kernel_path = found
    try:
        compiled = _compile_jit(fn, args.sm)
        _write_payload(compiled, args.out, args.sm, kernel_path, host_sm)
    except Exception as e:
        return _fail_env(f"triton.compile failed for {kernel_path}: {type(e).__name__}: {e}")

    print("Do not copy AGPL MoE kernels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
