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
        # Pin each guest's target dir to its OWN local `target/` (read_guest_elf
        # in the smoke test reads from `bench_vs/lambda/<name>/target/...`).
        #
        # We must set this EXPLICITLY rather than rely on the inherited value or
        # on unsetting it:
        #   * When spawned from `cargo test`, the inherited CARGO_TARGET_DIR
        #     points at the host workspace's build cache. That cache is shared
        #     across git worktrees that all build crates named
        #     `math`/`stark`/`crypto`/`lambda-vm-prover`, so build-std artifacts
        #     from a sibling worktree leak in, giving bogus "multiple different
        #     versions of crate `math`" errors that reference another worktree.
        #   * Merely unsetting it makes cargo walk up to discover a workspace
        #     root, which can resolve to the wrong worktree's path-dep cache.
        # An explicit, worktree-local path avoids both: the path is anchored
        # under THIS guest dir (and therefore THIS worktree), fully isolating it.
        export CARGO_TARGET_DIR="$dir/target"
        # Recursion/deserialize-only guests pull in lambda-vm-prover and its
        # serde stack; pin serde to 1.0.219 (pre-`serde_core` split) so
        # `-Z build-std=core,alloc` works.
        if [ "$name" = "recursion" ] || [ "$name" = "deserialize-only" ]; then
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
build_one deserialize-only
build_one keccak-roundtrip

echo "[recursion-elfs] done"
