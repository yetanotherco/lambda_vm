#!/usr/bin/env bash
# Build the fibonacci-bench and recursion-bench ELFs for the recursion smoke test.
#
# Uses the same toolchain + flags as bench_vs/run.sh, plus pins serde to the last
# pre-`serde_core`-split version (1.0.219) inside each guest's own workspace lock
# so build-std works on the riscv64im-lambda-vm-elf target.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
TARGET_SPEC="$ROOT_DIR/executor/programs/riscv64im-lambda-vm-elf.json"

TOOLCHAIN="nightly-2026-02-01"

build_one() {
    local name="$1"
    local dir="$ROOT_DIR/bench_vs/lambda/$name"
    echo "[recursion-elfs] building $name ..."
    (
        cd "$dir"
        # Recursion guest pulls in lambda-vm-prover and its serde stack; pin serde
        # to 1.0.219 (pre-`serde_core` split) so `-Z build-std=core,alloc` works.
        if [ "$name" = "recursion" ]; then
            cargo "+$TOOLCHAIN" update -p serde --precise 1.0.219 2>/dev/null || true
        fi
        cargo "+$TOOLCHAIN" build --release \
            --target "$TARGET_SPEC" \
            -Z build-std=core,alloc \
            -Z build-std-features=compiler-builtins-mem \
            -Z json-target-spec
    )
}

build_one empty
build_one fibonacci
build_one recursion
build_one keccak-roundtrip

echo "[recursion-elfs] done"
