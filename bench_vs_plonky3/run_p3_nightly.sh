#!/usr/bin/env bash
# Orchestrates the Lambda-vs-Plonky3 nightly benchmark.
#
# Runs the headline configuration (log_rows=21, num_sequences=16 → 32 cols)
# into `$REPORT_BASE/headline/`. Consumed by
# `.github/scripts/publish_bench_vs.sh` to render the Headline section of
# the Slack post.
#
# Usage:
#   ./bench_vs_plonky3/run_p3_nightly.sh [REPORT_BASE]
#
# Defaults: REPORT_BASE=bench_vs_artifacts/p3

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

run_one headline 21 16
