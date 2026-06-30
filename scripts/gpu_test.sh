#!/usr/bin/env bash
#
# gpu_test.sh — run the CUDA-only test groups on a GPU box.
#
# These groups can't run in CPU CI (GitHub runners have no GPU):
#   1. math-cuda kernel parity        (make test-math-cuda)
#   2. end-to-end GPU dispatch + proof (make test-cuda-integration)
#   3. GPU error-path / CPU fallback   (make test-cuda-fallback)
#
# Runs on the rented Vast box from the gpu-tests.yml merge-queue workflow. All three groups
# run even if one fails (so the log shows every failure); the script exits non-zero if ANY
# group failed, which fails the workflow job and blocks the merge.
#
# Env:
#   CUDARC_PIN   cudarc CUDA-version feature to pin (default cuda-12080). See the sed below.
#   SYSROOT_DIR  rv64 sysroot (default /opt/lambda-vm-sysroot, provisioned by the template).

set -euo pipefail

CUDARC_PIN="${CUDARC_PIN:-cuda-12080}"
export SYSROOT_DIR="${SYSROOT_DIR:-/opt/lambda-vm-sysroot}"

log() { printf '\n=== %s ===\n' "$*"; }

# --- GPU toolchain sanity (fail loudly rather than silently falling back to CPU) ---
log "GPU toolchain"
if ! command -v nvcc >/dev/null 2>&1; then
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        [ -x "$d/nvcc" ] && export PATH="$d:$PATH" && break
    done
fi
command -v nvcc >/dev/null 2>&1 || { echo "ERROR: nvcc not found — CUDA toolkit missing" >&2; exit 1; }
nvcc --version | tail -n 2
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

# --- Pin cudarc so it binds a fixed driver-symbol set --------------------------
# crypto/math-cuda/Cargo.toml uses `cuda-version-from-build-system` + `fallback-latest`;
# when detection falls back to "latest", cudarc requests symbols some boxes' driver doesn't
# export (e.g. cuDevSmResourceSplit / cuCtxGetDevice_v2) -> runtime panic. Pinning to a fixed
# CUDA version (12.8, matching the cuda_max_good>=12.8 offer floor) avoids that.
log "pinning cudarc to $CUDARC_PIN"
sed -i "s/\"cuda-version-from-build-system\"/\"${CUDARC_PIN}\"/; /\"fallback-latest\"/d" \
    crypto/math-cuda/Cargo.toml

# --- Build the asm guest ELFs used by Groups 2 & 3 (clang on .s; fast) ----------
# (math-cuda parity tests need no ELF; cuda_path_integration / cuda_fallback prove an asm ELF.)
log "compiling asm guest programs"
make compile-programs-asm

# --- Run the three CUDA test groups via the Makefile targets --------------------
fail=0
run() {  # $1 = make target
    log "make $1"
    if ! make "$1"; then
        echo "::error::GPU test group failed: $1"
        fail=1
    fi
}
run test-math-cuda         # Group 1: kernel parity
run test-cuda-integration  # Group 2: end-to-end GPU dispatch + proof verifies
run test-cuda-fallback     # Group 3: GPU error -> CPU fallback still verifies

if [ "$fail" -ne 0 ]; then
    log "FAILED — one or more GPU test groups failed"
    exit 1
fi
log "all GPU test groups passed"
