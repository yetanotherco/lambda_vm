#!/usr/bin/env bash
# Measure the recursion-guest verifier WITH vs WITHOUT the Keccak precompile.
#
# The recursion guest (the STARK verifier compiled to RISC-V) lives in YOUR
# guest crate, outside this repo. This script A/Bs the `keccak` precompile shim
# by toggling a `[patch.crates-io] keccak = { path = <keccak-precompile> }`
# entry in the guest's Cargo.toml, running your verify benchmark each time, and
# printing both results so you can see the cycle drop.
#
# Usage:
#   GUEST_DIR=/path/to/recursion-guest \
#   VERIFY_BENCH_CMD='cargo run --release -- bench-verify' \
#     scripts/measure_verifier_precompile.sh
#
# Required env:
#   GUEST_DIR         Root of your recursion-guest crate (the one with the
#                     Cargo.toml that builds the verifier-as-RISC-V program).
#   VERIFY_BENCH_CMD  The command (run FROM $GUEST_DIR) that builds + runs the
#                     guest verify and prints its RISC-V cycle count. Whatever
#                     you already use to get the "40.5B / 67M" numbers.
#
# Optional env:
#   CYCLE_GREP        A grep -oE pattern to extract the cycle number from the
#                     bench output (default tries common forms). Purely for the
#                     summary line; full output is always shown.
#
# Correctness: the shim routes only the Keccak-f[1600] permutation to the VM
# precompile (a0=state ptr, a7=usize::MAX-1), reusing sha3's sponge/padding, so
# every hash is byte-identical — the verify result is unchanged, only faster.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SHIM_DIR="$ROOT_DIR/keccak-precompile"

: "${GUEST_DIR:?set GUEST_DIR to your recursion-guest crate root}"
: "${VERIFY_BENCH_CMD:?set VERIFY_BENCH_CMD to the command that runs your guest verify bench}"
CYCLE_GREP="${CYCLE_GREP:-[0-9][0-9.,]*[ ]*(cycles|B|M|instructions)}"

GUEST_DIR="$(cd "$GUEST_DIR" && pwd)"
MANIFEST="$GUEST_DIR/Cargo.toml"
[ -f "$MANIFEST" ] || { echo "no Cargo.toml in $GUEST_DIR" >&2; exit 1; }
[ -f "$SHIM_DIR/Cargo.toml" ] || { echo "shim not found at $SHIM_DIR" >&2; exit 1; }

BACKUP="$(mktemp)"
cp "$MANIFEST" "$BACKUP"
restore() { cp "$BACKUP" "$MANIFEST"; rm -f "$BACKUP"; }
trap restore EXIT

GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'

run_bench() {
    local label="$1"
    echo -e "\n${BOLD}=== verify bench: $label ===${NC}"
    ( cd "$GUEST_DIR" && eval "$VERIFY_BENCH_CMD" ) | tee "/tmp/verify_bench_${label}.out"
}

add_patch() {
    # Add `keccak = { path = SHIM }` under [patch.crates-io], creating the
    # section if absent. Duplicate [patch.crates-io] tables are a cargo error,
    # so we append into the existing one when present.
    local line="keccak = { path = \"$SHIM_DIR\" }"
    if grep -qE '^\[patch\.crates-io\]' "$MANIFEST"; then
        # insert right after the section header
        awk -v l="$line" '
            { print }
            /^\[patch\.crates-io\]/ && !done { print l; done=1 }
        ' "$MANIFEST" > "$MANIFEST.tmp" && mv "$MANIFEST.tmp" "$MANIFEST"
    else
        printf '\n[patch.crates-io]\n%s\n' "$line" >> "$MANIFEST"
    fi
}

extract() { grep -oiE "$CYCLE_GREP" "$1" | head -1 || true; }

# 1) Baseline (software Keccak)
restore  # ensure clean
cp "$MANIFEST" "$BACKUP"
run_bench "baseline"

# 2) With precompile shim
add_patch
echo -e "${GREEN}[patched] added: keccak = { path = $SHIM_DIR }${NC}"
run_bench "precompile"

# restore happens via trap

echo ""
echo -e "${BOLD}=== Summary ===${NC}"
echo "  baseline (software Keccak) : $(extract /tmp/verify_bench_baseline.out)"
echo "  with precompile shim       : $(extract /tmp/verify_bench_precompile.out)"
echo ""
echo "Full outputs: /tmp/verify_bench_baseline.out  /tmp/verify_bench_precompile.out"
echo "Guest Cargo.toml restored."
