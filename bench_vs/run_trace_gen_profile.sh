#!/usr/bin/env bash
#
# Trace-generation profiling on a big-RAM box (no swap → clean CPU + true peak mem).
#
# Produces, per ethrex block size:
#   - [gen] per-phase trace-generation timers  (phase2 op-routing, bitwise, per-table, ...)
#   - the PROVER TIMING report (execute / trace build / round1 / rounds2-4 / FFT / Merkle)
#   - peak resident memory  (GNU `/usr/bin/time -v`, Linux)
#
# Usage:
#   bash bench_vs/run_trace_gen_profile.sh [TX_COUNTS...]
#   bash bench_vs/run_trace_gen_profile.sh            # defaults: 2 5 10 20 40
#   bash bench_vs/run_trace_gen_profile.sh 5 10 20    # custom set
#
# Prereqs handled automatically: builds the ethrex guest ELF and generates a
# fixture per TX count. Assumes Rust nightly-2026-02-01 is installed (the guest
# build needs it). On a fresh Linux box, run `make deps-linux` once first.

set -euo pipefail
cd "$(dirname "$0")/.."          # repo root
ROOT=$(pwd)

TX_COUNTS=("${@:-}")
if [ -z "${TX_COUNTS[*]}" ]; then TX_COUNTS=(2 5 10 20 40); fi

OUTDIR="$ROOT/bench_vs/results/trace_gen_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"
echo ">>> results dir: $OUTDIR"

# Pick a peak-memory wrapper: GNU time (-v) on Linux, BSD time (-l) on macOS, else none.
MEMWRAP=()
if /usr/bin/time -v true 2>/dev/null; then MEMWRAP=(/usr/bin/time -v)
elif /usr/bin/time -l true 2>/dev/null; then MEMWRAP=(/usr/bin/time -l)
fi

echo ">>> [1/3] building ethrex guest ELF (build-std; first time is slow)"
make executor/program_artifacts/rust/ethrex.elf

echo ">>> [2/3] building tooling + instrumented prover bench"
( cd tooling/ethrex-fixtures && cargo build --release )
CARGO_PROFILE_RELEASE_DEBUG=1 cargo bench -p lambda-vm-prover \
    --bench profile_ethrex --features "parallel,instruments" --no-run

echo ">>> [3/3] running profiles for TX counts: ${TX_COUNTS[*]}"
echo ">>> rayon threads = ${RAYON_NUM_THREADS:-<all cores>}"

for n in "${TX_COUNTS[@]}"; do
    fixture="executor/tests/ethrex_${n}_transfers.bin"
    log="$OUTDIR/ethrex_${n}tx.log"
    echo "----- ${n} transfers -> $log -----"

    if [ ! -f "$fixture" ]; then
        echo "  generating fixture ($n tx)..."
        ( cd tooling/ethrex-fixtures && cargo run --release -- "$n" "$ROOT/$fixture" )
    fi

    # Run the prebuilt bench binary directly so the mem wrapper measures only the prove.
    BIN=$(ls -t target/release/deps/profile_ethrex-* 2>/dev/null | grep -v '\.d$' | head -1)
    {
        echo "=== ethrex ${n} transfers ==="
        date
        "${MEMWRAP[@]}" "$BIN" "ethrex_${n}_transfers"
    } 2>&1 | tee "$log"
done

echo
echo ">>> DONE. Per-run logs in: $OUTDIR"
echo ">>> Summary of trace-generation phase timers:"
for n in "${TX_COUNTS[@]}"; do
    echo "----- ${n} tx -----"
    grep -E "\[gen\]|Trace build|TOTAL|Maximum resident|maximum resident|peak memory" \
        "$OUTDIR/ethrex_${n}tx.log" 2>/dev/null || true
done
