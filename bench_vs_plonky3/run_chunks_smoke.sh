#!/usr/bin/env bash
# Chunks-protocol smoke test: one run of lambda-chunks at log=17 on fib_pair.
#
# Just verifies the chunks pipeline (Phase 5.1) runs end-to-end on production
# inputs without crashing or rejecting the verifier. Not statistically rigorous.
#
# Usage:
#   ./bench_vs_plonky3/run_chunks_smoke.sh [LOG_ROWS]
#
# Defaults: LOG_ROWS=17 (~131k rows of trace, fits in seconds on any machine).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_ROWS="${1:-17}"
NUM_SEQUENCES=16
BLOWUP=2
QUERIES=219

echo "=== chunks smoke test (Phase 5.1) ==="
echo "  log_rows=$LOG_ROWS  num_sequences=$NUM_SEQUENCES  blowup=$BLOWUP  queries=$QUERIES"
echo ""

echo "[build] cargo build --release -p bench-vs-plonky3 --bin prove_bench"
cargo build --release -p bench-vs-plonky3 --bin prove_bench --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -3
echo ""

TARGET_DIR=$(cargo metadata --manifest-path "$ROOT_DIR/Cargo.toml" --format-version 1 --no-deps 2>/dev/null \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])' \
    2>/dev/null || echo "$ROOT_DIR/target")
BIN="$TARGET_DIR/release/prove_bench"

for PROVER in lambda lambda-chunks; do
    echo "--- prover=$PROVER ---"
    "$BIN" --prover "$PROVER" \
        --log-rows "$LOG_ROWS" --num-sequences "$NUM_SEQUENCES" \
        --blowup "$BLOWUP" --queries "$QUERIES" --grinding 0
    echo ""
done

echo "Smoke test OK. If both provers printed METRICS lines and no error,"
echo "the chunks pipeline is functional. Run run_chunks_bench.sh for an A/B/C."
