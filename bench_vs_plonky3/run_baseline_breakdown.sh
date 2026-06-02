#!/usr/bin/env bash
# Phase 0+ of the rebaseline plan: same as run_baseline_fixed.sh but with
# Lambda phase breakdown enabled (--breakdown flag, --features instruments).
#
# Each run captures BREAKDOWN lines from the Lambda prover via
# stark::instruments::take() and aggregates them into breakdown.tsv per
# width report-dir, so we can see exactly which phase moved (or didn't)
# after the base/ext fix.
#
# Use this when you want to compare per-phase against vm5's May 13 breakdown
# (e.g. r2_constraints, r1_main_lde, r4_fri_commit).
#
# Usage:
#   ./bench_vs_plonky3/run_baseline_breakdown.sh [SUFFIX]
#
# SUFFIX defaults to the current timestamp. Examples:
#   ./bench_vs_plonky3/run_baseline_breakdown.sh
#   ./bench_vs_plonky3/run_baseline_breakdown.sh fix_baseline_breakdown_v1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

SUFFIX="${1:-$(date +%Y%m%d_%H%M)}"
LOG_ROWS=21
RUNS=10
NSEQS=(16 32 64)
BASE="$ROOT_DIR/bench_vs_plonky3/reports/bench_vs_p3_baseline_breakdown_${SUFFIX}"

echo "=== Phase 0+: corrected baseline WITH Lambda phase breakdown ==="
echo "log_rows:        $LOG_ROWS"
echo "num_sequences:   ${NSEQS[*]}"
echo "runs per point:  $RUNS"
echo "mode:            scalar (RUSTFLAGS -avx2,-avx512f)"
echo "instruments:     enabled (Lambda phase timings)"
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
        --breakdown \
        --report-dir "$OUT"
    echo
done

echo "=== Summary (totals) ==="
printf "%-6s %-12s %-10s %-12s %-10s %-10s\n" "n_seq" "L median s" "L CV %" "P3 median s" "P3 CV %" "ratio"
printf "%-6s %-12s %-10s %-12s %-10s %-10s\n" "-----" "----------" "------" "-----------" "------" "------"
for N in "${NSEQS[@]}"; do
    TSV="${BASE}_n${N}/results.tsv"
    if [[ ! -f "$TSV" ]]; then
        printf "%-6s MISSING %s\n" "$N" "$TSV"
        continue
    fi
    read -r _lr _rows L_MED L_CV P_MED P_CV RATIO _runs < <(tail -n1 "$TSV")
    printf "%-6s %-12s %-10s %-12s %-10s %-10s\n" "$N" "$L_MED" "$L_CV" "$P_MED" "$P_CV" "$RATIO"
done

echo
echo "=== Lambda phase medians (ms) across $RUNS runs per width ==="
printf "%-18s" "phase"
for N in "${NSEQS[@]}"; do printf " %-10s" "n=$N"; done
echo
printf "%-18s" "------------------"
for N in "${NSEQS[@]}"; do printf " %-10s" "----------"; done
echo

PHASES=(prove_total prepass main_commits r1_main_lde r1_main_merkle rounds_2_4 r2_constraints r2_comp_decompose r2_comp_commit r3_ood r4_deep_comp r4_deep_extend r4_fri_commit r4_queries)
for phase in "${PHASES[@]}"; do
    printf "%-18s" "$phase"
    for N in "${NSEQS[@]}"; do
        TSV="${BASE}_n${N}/breakdown.tsv"
        if [[ ! -f "$TSV" ]]; then
            printf " %-10s" "-"
            continue
        fi
        # column 6 = phase, column 7 = ms
        MEDIAN=$(awk -F'\t' -v p="$phase" '$6==p {print $7}' "$TSV" | sort -g | awk '
            { a[NR]=$1+0 }
            END {
                if (NR==0) { print "-" }
                else if (NR%2==1) printf "%.1f", a[(NR+1)/2]
                else printf "%.1f", (a[NR/2]+a[NR/2+1])/2
            }')
        printf " %-10s" "$MEDIAN"
    done
    echo
done

echo
echo "=== Normalized phase comparison: Lambda vs P3 (median ms over $RUNS runs) ==="
echo "    NOTE: only fair when P3 ran scalar (AUDIT val_packing_width=1)."
NORM_PHASES=(norm_prove_total norm_trace_commit norm_trace_lde norm_trace_merkle \
    norm_constraint_eval norm_quotient_commit norm_quotient_merkle norm_open norm_fri norm_deep_ood)

# median of breakdown.tsv `ms` (col 7) for a given prover (col 3) + phase (col 6)
norm_median() { # $1=tsv $2=prover $3=phase
    awk -F'\t' -v pr="$2" -v ph="$3" '$3==pr && $6==ph {print $7}' "$1" | sort -g | awk '
        { a[NR]=$1+0 }
        END {
            if (NR==0) { print "-"; exit }
            if (NR%2==1) printf "%.1f", a[(NR+1)/2]
            else printf "%.1f", (a[NR/2]+a[NR/2+1])/2
        }'
}

for N in "${NSEQS[@]}"; do
    TSV="${BASE}_n${N}/breakdown.tsv"
    [[ -f "$TSV" ]] || continue
    echo "--- n_seq=$N (main_cols=$((2 * N))) ---"
    printf "%-22s %12s %12s %8s\n" "phase" "lambda(ms)" "p3(ms)" "L/P3"
    for ph in "${NORM_PHASES[@]}"; do
        L=$(norm_median "$TSV" lambda "$ph")
        P=$(norm_median "$TSV" p3 "$ph")
        R=$(awk -v a="$L" -v b="$P" 'BEGIN { if (b+0 > 0) printf "%.2f", a/b; else print "-" }')
        printf "%-22s %12s %12s %8s\n" "$ph" "$L" "$P" "$R"
    done
    echo
done

echo
echo "Compare against vm5 May 13 baseline (no fix, P3 with vector-lane Keccak):"
echo "  ~/Documents/lambda_vm5/bench_vs_plonky3/reports/bench_vs_p3_20260513_2033_upstream/breakdown_log21/breakdown.tsv"
echo "  (only n=16 baseline available there; for n=32/64 see cols_log21_n* dirs)"
echo
echo "Pull artifacts to mac (when done):"
echo "  scp -r vm-benchmarks-1:${BASE}_n* \\"
echo "      ~/Documents/lambda_vm3/bench_vs_plonky3/reports/"
echo
echo "DONE"
