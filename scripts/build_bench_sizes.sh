#!/bin/bash
# Build a bench program at multiple sizes via env var.
# Places ELFs in executor/program_artifacts/bench/ with size suffix.
#
# Usage: build_bench_sizes.sh <program> <env_var> <values...>
# Example: build_bench_sizes.sh modular_exp NUM_ITERATIONS 5000 10000 20000

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH_DIR="$ROOT_DIR/executor/programs/bench"
ARTIFACTS_DIR="$ROOT_DIR/executor/program_artifacts/bench"
TARGET_SPEC="$ROOT_DIR/executor/programs/riscv64im-lambda-vm-elf.json"
SHARED_TARGET_DIR="$ROOT_DIR/executor/programs/target"

program=$1; shift
env_var=$1; shift

if [ ! -d "$BENCH_DIR/$program" ]; then
    echo "Error: $BENCH_DIR/$program not found" >&2
    exit 1
fi

mkdir -p "$ARTIFACTS_DIR"

for val in "$@"; do
    echo "Building ${program} with ${env_var}=${val}..."

    cd "$BENCH_DIR/$program"
    env CARGO_TARGET_DIR="$SHARED_TARGET_DIR" "${env_var}=${val}" \
        rustup run nightly-2026-02-01 cargo build --release \
            --target "$TARGET_SPEC" \
            -Z build-std=core,alloc,std,compiler_builtins,panic_abort \
            -Z build-std-features=compiler-builtins-mem \
            -Z json-target-spec 2>&1 | tail -1

    cp "$SHARED_TARGET_DIR/riscv64im-lambda-vm-elf/release/$program" \
       "$ARTIFACTS_DIR/${program}_${val}.elf"

    echo "  -> ${program}_${val}.elf"
done

echo "Done."
