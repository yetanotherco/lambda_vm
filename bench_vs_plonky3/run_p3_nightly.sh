#!/usr/bin/env bash
# Orchestrates the Lambda-vs-Plonky3 nightly benchmark.
#
# Runs 5 configurations of run.sh into separate report-dirs under
# `$REPORT_BASE`. The same 5 dirs are consumed by
# `.github/scripts/publish_bench_vs.sh` to render the 3-section Slack post
# (Headline + Size scaling + Column scaling).
#
# Usage:
#   ./bench_vs_plonky3/run_p3_nightly.sh [REPORT_BASE]
#
# Defaults: REPORT_BASE=bench_vs_artifacts/p3
#
# Each run is 10 iterations × 2 provers; the 5 runs together take ~3 min on
# the bench server.

set -euo pipefail

REPORT_BASE="${1:-bench_vs_artifacts/p3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SH="$SCRIPT_DIR/run.sh"

if [ ! -x "$RUN_SH" ]; then
    echo "run.sh not found or not executable at $RUN_SH" >&2
    exit 1
fi

run_one() {
    local label=$1
    local log_rows=$2
    local num_sequences=$3
    local out_dir="$REPORT_BASE/$label"
    echo
    echo "=== ${label} (log_rows=${log_rows}, num_sequences=${num_sequences}) ==="
    bash "$RUN_SH" \
        --log-rows "$log_rows" \
        --num-sequences "$num_sequences" \
        --runs 10 \
        --scalar \
        --report-dir "$out_dir" \
        --no-color
}

# Size sweep + headline (32 cols).
run_one size_log19 19 16
run_one size_log20 20 16
run_one headline   21 16

# Column sweep @ log_rows=21.
run_one cols_n4    21  4
run_one cols_n64   21 64
