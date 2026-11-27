#!/usr/bin/env bash
#
# gpu_test.sh — run the CUDA-only test groups on a GPU box.
#
# Exercises the CUDA path, which CPU CI can't (GitHub runners have no GPU):
#   1. math-cuda kernel parity          (make test-math-cuda)
#   2. end-to-end GPU dispatch + proof   (make test-cuda-integration)
#   3. num_parts==1 (DECODE) device path (make test-cuda-d1)
#   4. GPU error-path / CPU fallback     (make test-cuda-fallback)
#   5. prover/stark/crypto/ecsm suite    (make test-prover-cuda) — CPU CI's prover tests on GPU
#   6. production-posture coverage       (make test-prover-comprehensive-cuda)
#   7. cross-posture proof rejection     (make test-cross-posture-cuda)
#   8. fixed-seed proof identities      (make test-prover-identities-cuda)
#
# Runs on the rented Vast box from the gpu-tests.yml merge-queue workflow. All groups
# run even if one fails (so the log shows every failure); the script exits non-zero if ANY
# group failed, which fails the workflow job and blocks the merge.
#
# Env:
#   SYSROOT_DIR  rv64 sysroot (default /opt/lambda-vm-sysroot, provisioned by the template).

set -euo pipefail

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
# Full nvidia-smi up front: GPU model, driver + CUDA runtime version, memory — for the log.
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

# cudarc's CUDA-version pin now lives permanently in crypto/math-cuda/Cargo.toml
# (feature `cuda-12080`), so this script no longer patches the manifest. Kernels
# are AOT-compiled to cubin by build.rs, so no PTX/driver-version juggling either.

# --- Build the guest ELFs the tests prove ---------------------------------------
# math-cuda parity needs none; cuda_path_integration / cuda_fallback prove an asm ELF; the
# prover suite (Groups 5 & 6) proves asm AND rust guests. Build both up front.
log "compiling guest programs (asm + rust)"
make compile-programs-asm
make compile-programs-rust

# STARK builds a cdylib alongside its Rust library. Switching feature graphs
# in one Cargo target directory invalidates its fingerprint and its dependents.
build_root=${CARGO_TARGET_DIR:-"$PWD/target"}
export GPU_CUDA_FAULT_TARGET_DIR="$build_root/gpu-faults"

# --- Run the CUDA test groups via the Makefile targets --------------------------
fail=0
run() {  # $1 = make target
    log "make $1"
    local output start=$SECONDS
    output=$(mktemp)
    if ! make "$1" 2>&1 | tee "$output" || grep -qi 'empty cubin stubs' "$output"; then
        echo "::error::GPU test group failed: $1"
        fail=1
    fi
    printf 'GPU test group %s took %ss\n' "$1" "$((SECONDS - start))"
    rm -f "$output"
}
CARGO_TARGET_DIR="$build_root/gpu-parity" run test-math-cuda  # Group 1: kernel parity
run test-cuda-integration           # Group 2: end-to-end GPU dispatch + proof verifies
run test-cuda-d1                    # Group 3: num_parts==1 (DECODE) device DEEP/FRI + verify
run test-cuda-fallback              # Group 4: GPU error -> CPU fallback still verifies
CARGO_TARGET_DIR="$build_root/gpu-suite" run test-prover-cuda  # Group 5: multi-package CUDA suite
run test-prover-comprehensive-cuda  # Group 6: production proofs and parameter pins
run test-cross-posture-cuda         # Group 7: production refusal of TEST proofs
run test-prover-identities-cuda     # Group 8: deterministic proofs under the host and card levers

if [ "$fail" -ne 0 ]; then
    log "FAILED — one or more GPU test groups failed"
    exit 1
fi
log "all GPU test groups passed"
