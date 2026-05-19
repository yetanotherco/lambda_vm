#!/usr/bin/env bash
# Phase 0 of the rebaseline plan: establish the corrected baseline after the
# base/ext field path fix is in place.
#
# Runs the Lambda-vs-P3 bench at log_rows=21 across three widths
# (num_sequences ∈ {16, 32, 64}) with 10 runs per point in scalar mode.
# Each width writes its report to
# bench_vs_plonky3/reports/bench_vs_p3_baseline_fixed_<SUFFIX>_n<N>/.
#
# At the end it prints a consolidated summary and the scp command to pull
# the reports back to the mac.
#
# Usage:
#   ./bench_vs_plonky3/run_baseline_fixed.sh [SUFFIX]
#
# SUFFIX defaults to the current timestamp. Examples:
#   ./bench_vs_plonky3/run_baseline_fixed.sh
#   ./bench_vs_plonky3/run_baseline_fixed.sh baseline_v2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SUFFIX="${1:-$(date +%Y%m%d_%H%M)}"
LOG_ROWS=21
RUNS=10
NSEQS=(16 32 64)
BASE="$ROOT_DIR/bench_vs_plonky3/reports/bench_vs_p3_baseline_fixed_${SUFFIX}"

echo "=== Phase 0: corrected baseline (base/ext fix) ==="
echo "log_rows:        $LOG_ROWS"
echo "num_sequences:   ${NSEQS[*]}"
echo "runs per point:  $RUNS"
echo "mode:            scalar (RUSTFLAGS -avx2,-avx512f)"
echo "output base:     $BASE"
echo

cd "$ROOT_DIR"

for N in "${NSEQS[@]}"; do
    OUT="${BASE}_n${N}"
    echo "--- running n_seq=$N -> $OUT ---"
    ./bench_vs_plonky3/run.sh \
        --log-rows "$LOG_ROWS" \
        --num-sequences "$N" \
        --runs "$RUNS" \
        --scalar \
        --report-dir "$OUT"
    echo
done

echo "=== Summary ==="
printf "%-6s %-12s %-10s %-12s %-10s %-10s\n" "n_seq" "L median s" "L CV %" "P3 median s" "P3 CV %" "ratio"
printf "%-6s %-12s %-10s %-12s %-10s %-10s\n" "-----" "----------" "------" "-----------" "------" "------"
for N in "${NSEQS[@]}"; do
    TSV="${BASE}_n${N}/results.tsv"
    if [[ ! -f "$TSV" ]]; then
        printf "%-6s MISSING %s\n" "$N" "$TSV"
        continue
    fi
    # results.tsv columns: log_rows rows lambda_median_s lambda_cv_pct p3_median_s p3_cv_pct ratio_lambda_over_p3 runs
    # tail -n1 skips the header
    read -r _lr _rows L_MED L_CV P_MED P_CV RATIO _runs < <(tail -n1 "$TSV")
    printf "%-6s %-12s %-10s %-12s %-10s %-10s\n" "$N" "$L_MED" "$L_CV" "$P_MED" "$P_CV" "$RATIO"
done

echo
echo "Expected gates:"
echo "  - lambda CV < 2%"
echo "  - ratio @ n_seq=16 around 1.4-1.5x (vm4 measured ~1.377x at log_rows=19)"
echo "  - first-run AUDIT line must show base_transition_constraints=2*n_seq"
echo "    (check bench_vs_p3_baseline_fixed_${SUFFIX}_n*/raw/lambda_log${LOG_ROWS}_run1.stdout)"
echo
echo "Pull artifacts to mac (when done):"
echo "  scp -r vm-benchmarks-1:${BASE}_n* \\"
echo "      ~/Documents/lambda_vm3/bench_vs_plonky3/reports/"
echo
echo "DONE"
