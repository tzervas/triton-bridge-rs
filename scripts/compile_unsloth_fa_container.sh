#!/usr/bin/env bash
# Compile Unsloth FA Triton → PTX inside the compare image + persist volume.
# Host Python often has no unsloth (Job C FAIL). Do not invent PTX.
#
#   ./scripts/compile_unsloth_fa_container.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNSLOTH_RS="${UNSLOTH_RS:-$(cd "$ROOT/../unsloth-rs" && pwd)}"
IMG="${COMPARE_PY_IMAGE:-localhost/unsloth-rs-compare-py:local}"
SITE_VOL="${COMPARE_SITE_VOL:-unsloth-rs-compare-site}"
OUT="${1:-$ROOT/precompiled/sm90_flash_fwd.ptx}"
SM="${SM:-90}"

if [[ ! -e /dev/nvidia0 ]]; then
  echo "FAIL_ENV: /dev/nvidia0 missing" >&2
  exit 2
fi
if [[ ! -f "$UNSLOTH_RS/compare/Containerfile.py" ]]; then
  echo "FAIL_ENV: unsloth-rs compare image sources missing" >&2
  exit 2
fi
if ! podman image exists "$IMG" >/dev/null 2>&1; then
  echo "==> build $IMG"
  podman build -t "$IMG" -f "$UNSLOTH_RS/compare/Containerfile.py" "$UNSLOTH_RS"
fi
if ! podman volume exists "$SITE_VOL" >/dev/null 2>&1; then
  podman volume create "$SITE_VOL" >/dev/null
fi

echo "HONEST: host GPU is SM 12.0; --sm ${SM} is a compile target, not a launch guarantee."
mkdir -p "$(dirname "$OUT")"
# Bind the script + dest. PYTHONPATH is the persist volume from compare/run.sh.
podman run --rm --device nvidia.com/gpu=all \
  --entrypoint /bin/bash \
  -e CUDA_COMPUTE_CAP="${CUDA_COMPUTE_CAP:-90}" \
  -e PYTHONPATH=/opt/site-extra \
  -e UNSLOTH_SKIP_TORCHVISION_CHECK=1 \
  -v "$SITE_VOL:/opt/site-extra:Z" \
  -v "$ROOT/scripts:/opt/tb-scripts:ro,Z" \
  -v "$(dirname "$OUT"):/out:Z" \
  "$IMG" \
  -lc "python -c 'import unsloth' || pip install --target /opt/site-extra unsloth
       python /opt/tb-scripts/compile_unsloth_fa.py --out /out/$(basename "$OUT") --sm ${SM}"
