#!/bin/bash
#
# Orchestrator for Lambda-vs-Plonky3 sweeps against upstream Plonky3
# (post-migration from yetanotherco/Plonky3 fork).
#
# Runs three sweeps into separate report-dirs under a single dated base:
#   - size   : log_rows in {17, 19, 20, 21, 23}, num-sequences=16 (cols=32)
#   - cols   : log_rows=21, num-sequences in {4, 8, 16, 32, 64}
#   - breakdown: log_rows in {19, 21}, --breakdown (per-phase + P3 spans)
#
# All runs: --scalar, --runs 10. ~30 min total on the bench server.
#
# Usage:
#   ./run_all.sh                       # DATE=$(date +%Y%m%d_%H%M), e.g. 20260513_1530
#   DATE=20260513_run2 ./run_all.sh    # explicit suffix
#
set -u

cd /home/app/juan/lambda_vm

DATE="${DATE:-$(date +%Y%m%d_%H%M)}"
BASE="/home/app/juan/lambda_vm/bench_vs_p3_${DATE}_upstream"

mkdir -p "$BASE"
echo "=== run_all.sh starting at $(date -u +%FT%TZ) ==="
echo "Report base: $BASE"
echo

# --- 1. Size sweep ----------------------------------------------------------
for L in 17 19 20 21 23; do
    OUT="$BASE/size_log${L}"
    echo "--- size_log${L} (num_sequences=16) ---"
    ./bench_vs_plonky3/run.sh \
        --scalar \
        --log-rows "$L" \
        --num-sequences 16 \
        --runs 10 \
        --report-dir "$OUT" \
        --no-color \
        2>&1 | tee "$BASE/size_log${L}.stdout"
done

# --- 2. Column sweep @ log=21 ----------------------------------------------
for N in 4 8 16 32 64; do
    OUT="$BASE/cols_log21_n${N}"
    echo "--- cols_log21_n${N} (log_rows=21, cols=$((2*N))) ---"
    ./bench_vs_plonky3/run.sh \
        --scalar \
        --log-rows 21 \
        --num-sequences "$N" \
        --runs 10 \
        --report-dir "$OUT" \
        --no-color \
        2>&1 | tee "$BASE/cols_log21_n${N}.stdout"
done

# --- 3. Breakdown log=19 and log=21 ----------------------------------------
for L in 19 21; do
    OUT="$BASE/breakdown_log${L}"
    echo "--- breakdown_log${L} ---"
    ./bench_vs_plonky3/run.sh \
        --scalar \
        --breakdown \
        --log-rows "$L" \
        --num-sequences 16 \
        --runs 10 \
        --report-dir "$OUT" \
        --no-color \
        2>&1 | tee "$BASE/breakdown_log${L}.stdout"
done

echo
echo "=== run_all.sh finished at $(date -u +%FT%TZ) ==="
echo "Results: $BASE"
ls -la "$BASE"
