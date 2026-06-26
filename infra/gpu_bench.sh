#!/bin/bash
# Run the headline ethrex prover benchmark on a GPU box, with the CUDA path enabled.
#
# Usage: infra/gpu_bench.sh [runs]
#   runs  number of prove iterations (default 3)
#
# Assumes the box was provisioned by the Vast template onstart
# (yetanotherco/scripts/bootstrap-onstart.sh): Rust 1.94.0 + nightly-2026-02-01,
# LLVM/clang, and the rv64 sysroot at /opt/lambda-vm-sysroot are already in place;
# CUDA/nvcc come from the base image. This script does NOT provision — it only
# builds with `prover/cuda`, generates the bench fixture, and runs the prove loop.
#
# It proves the SAME workload as the CPU benchmark (.github/workflows/benchmark-pr.yml):
# the ethrex guest ELF against a generated 20-transfer (distinct sender->recipient)
# block. Each run prints the CLI's "Proving time:" / "Peak heap:" lines, which the
# orchestrating workflow parses.

set -euo pipefail

RUNS="${1:-3}"

# Headline program (keep in sync with benchmark-pr.yml ELF/INPUT).
ELF="executor/program_artifacts/rust/ethrex.elf"
INPUT="executor/tests/ethrex_bench_20.bin"
TRANSFERS=20

log() { printf '\n=== %s ===\n' "$*"; }

# --- 0. Locate cargo + sysroot (provisioned by the template onstart) ---------
if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
export PATH="$HOME/.cargo/bin:$PATH"
export SYSROOT_DIR="${SYSROOT_DIR:-/opt/lambda-vm-sysroot}"

# --- 1. Sanity-check the GPU toolchain ---------------------------------------
log "GPU + toolchain check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "::error::nvidia-smi not found — no GPU driver on this box" >&2
    exit 1
fi
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader || true

# nvcc may live under /usr/local/cuda/bin without being on PATH.
if ! command -v nvcc >/dev/null 2>&1; then
    for d in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
        if [ -x "$d/nvcc" ]; then
            export PATH="$d:$PATH"
            export CUDA_HOME="${CUDA_HOME:-$(dirname "$d")}"
            break
        fi
    done
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "::error::nvcc not found — CUDA toolkit missing (math-cuda needs it to compile kernels)" >&2
    exit 1
fi
nvcc --version | tail -n 2

if ! command -v cargo >/dev/null 2>&1; then
    echo "::error::cargo not found — template onstart provisioning incomplete" >&2
    exit 1
fi
if [ ! -f "$SYSROOT_DIR/include/stdlib.h" ]; then
    echo "::error::rv64 sysroot missing at $SYSROOT_DIR — onstart provisioning incomplete" >&2
    exit 1
fi

# --- 2. Build the ethrex guest ELF (same target as the CPU bench) ------------
log "building ethrex guest ELF"
make "$ELF"

# --- 3. Generate the 20-transfer fixture -------------------------------------
log "generating $INPUT ($TRANSFERS distinct transfers)"
( cd tooling/ethrex-fixtures && cargo build --release )
GEN=tooling/ethrex-fixtures/target/release/ethrex-fixtures
"$GEN" "$TRANSFERS" "$INPUT" distinct

# --- 4. Build the CLI with the GPU (cuda) path -------------------------------
# jemalloc-stats gives the deterministic "Peak heap:" line; prover/cuda routes
# the LDE (and friends) through crypto/math-cuda. math-cuda/build.rs auto-detects
# the RTX 5090 arch (compute_120) via nvidia-smi, so no arch pin is needed.
log "building CLI with --features jemalloc-stats,prover/cuda"
cargo build --release -p cli --features jemalloc-stats,prover/cuda

# --- 5. Prove loop -----------------------------------------------------------
log "proving $ELF x$RUNS (GPU)"
for i in $(seq 1 "$RUNS"); do
    echo "--- Run $i/$RUNS ---"
    ./target/release/cli prove "$ELF" --private-input "$INPUT" -o /tmp/proof.bin --time
    rm -f /tmp/proof.bin
done

log "done"
