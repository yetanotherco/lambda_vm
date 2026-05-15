#!/usr/bin/env bash
# Chunks-protocol A/B/C bench: lambda single-H vs lambda-chunks vs p3.
#
# Runs each prover RUNS times per log-rows size, captures METRICS lines into
# a dated report dir (one .tsv per prover plus a metrics.txt header), and
# prints a median table at the end.
#
# Usage:
#   ./bench_vs_plonky3/run_chunks_bench.sh \
#       [--log-rows 17 19 21] [--runs 10] [--num-sequences 16] \
#       [--report-dir DIR] [--scalar] [--no-p3]
#
# Defaults: log-rows=17 19 21, runs=10, num-sequences=16, p3 enabled.
# --scalar disables AVX2/AVX-512 on x86_64 (apples-to-apples vs Lambda scalar).
# --no-p3 skips p3 (e.g., when you only care about chunks-vs-single-H).
#
# IMPORTANT: on the bench server (vm-benchmarks-1) report dirs get cleaned
# between sessions, so scp the report dir back to the local checkout when
# done — see `feedback_pull_bench_dirs_locally` in claude memory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_ROWS=()
RUNS=10
NUM_SEQUENCES=16
BLOWUP=2
QUERIES=219
GRINDING=0
REPORT_DIR=""
SCALAR=false
RUN_P3=true
WORKLOAD="fib_pair"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-rows)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LOG_ROWS+=("$1"); shift
            done
            ;;
        --runs)         RUNS="$2"; shift 2 ;;
        --num-sequences) NUM_SEQUENCES="$2"; shift 2 ;;
        --report-dir)   REPORT_DIR="$2"; shift 2 ;;
        --scalar)       SCALAR=true; shift ;;
        --no-p3)        RUN_P3=false; shift ;;
        --workload)     WORKLOAD="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,18p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ ${#LOG_ROWS[@]} -eq 0 ]; then
    LOG_ROWS=(17 19 21)
fi

if [ -z "$REPORT_DIR" ]; then
    REPORT_DIR="$ROOT_DIR/bench_vs_plonky3/reports/bench_vs_p3_chunks_$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$REPORT_DIR"

# --- scalar RUSTFLAGS (x86_64 only) ---
SCALAR_RUSTFLAGS=""
if $SCALAR; then
    case "$(uname -m)" in
        x86_64|amd64)
            SCALAR_RUSTFLAGS="-C target-feature=-avx2,-avx512f"
            export RUSTFLAGS="${RUSTFLAGS:-} $SCALAR_RUSTFLAGS"
            ;;
    esac
fi

# --- Provers to run ---
PROVERS=(lambda lambda-chunks)
if $RUN_P3 && [ "$WORKLOAD" = "fib_pair" ]; then
    # P3 only has fib_pair; --workload quadratic_pair forces --no-p3 semantics.
    PROVERS+=(p3)
fi

# --- Echo config ---
echo "=== chunks A/B/C bench ==="
echo "  workload:        $WORKLOAD"
echo "  log_rows:        ${LOG_ROWS[*]}"
echo "  runs/size:       $RUNS"
echo "  num_sequences:   $NUM_SEQUENCES  (cols=$((2 * NUM_SEQUENCES)))"
echo "  blowup/queries:  $BLOWUP / $QUERIES"
echo "  provers:         ${PROVERS[*]}"
echo "  scalar:          $($SCALAR && echo on || echo off)"
echo "  report dir:      $REPORT_DIR"
echo "  git sha:         $(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo ""

# --- Build once ---
echo "[build] cargo build --release -p bench-vs-plonky3 --bin prove_bench"
cargo build --release -p bench-vs-plonky3 --bin prove_bench \
    --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -3
echo ""

TARGET_DIR=$(cargo metadata --manifest-path "$ROOT_DIR/Cargo.toml" --format-version 1 --no-deps 2>/dev/null \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])' \
    2>/dev/null || echo "$ROOT_DIR/target")
BIN="$TARGET_DIR/release/prove_bench"
if [ ! -x "$BIN" ]; then
    echo "error: prove_bench not built at $BIN" >&2
    exit 1
fi

# --- Run all (prover × log_rows × run) ---
TIMESTAMP=$(date -u +%Y-%m-%dT%H:%M:%SZ)
GIT_SHA=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)
GIT_DIRTY=$(git -C "$ROOT_DIR" diff --quiet HEAD -- 2>/dev/null && echo clean || echo dirty)

{
    echo "timestamp_utc=$TIMESTAMP"
    echo "git_sha=$GIT_SHA"
    echo "git_tree=$GIT_DIRTY"
    echo "arch=$(uname -m)"
    echo "workload=$WORKLOAD"
    echo "log_rows=${LOG_ROWS[*]}"
    echo "runs_per_size=$RUNS"
    echo "num_sequences=$NUM_SEQUENCES"
    echo "blowup=$BLOWUP"
    echo "queries=$QUERIES"
    echo "grinding=$GRINDING"
    echo "scalar=$($SCALAR && echo on || echo off)"
    [ -n "$SCALAR_RUSTFLAGS" ] && echo "rustflags=$SCALAR_RUSTFLAGS"
    echo "provers=${PROVERS[*]}"
} > "$REPORT_DIR/metrics.txt"

for prover in "${PROVERS[@]}"; do
    OUT_FILE="$REPORT_DIR/${prover}.tsv"
    : > "$OUT_FILE"
    for lr in "${LOG_ROWS[@]}"; do
        for run in $(seq 1 "$RUNS"); do
            STDOUT_FILE="$REPORT_DIR/raw_${prover}_log${lr}_run${run}.stdout"
            if ! "$BIN" \
                --prover "$prover" \
                --workload "$WORKLOAD" \
                --log-rows "$lr" \
                --num-sequences "$NUM_SEQUENCES" \
                --blowup "$BLOWUP" \
                --queries "$QUERIES" \
                --grinding "$GRINDING" \
                > "$STDOUT_FILE" 2>&1
            then
                echo "  [$prover lr=$lr run=$run] FAILED:" >&2
                cat "$STDOUT_FILE" >&2
                exit 1
            fi
            grep '^METRICS' "$STDOUT_FILE" >> "$OUT_FILE" || {
                echo "  [$prover lr=$lr run=$run] no METRICS line in stdout" >&2
                exit 1
            }
            echo "  [$prover lr=$lr run=$run] ok"
        done
    done
    echo "  -> $OUT_FILE"
    echo ""
done

# --- Median table per size ---
echo "=== median prove_s per size ==="
printf "%-10s" "log_rows"
for prover in "${PROVERS[@]}"; do
    printf "  %-16s" "$prover (s)"
done
if $RUN_P3; then
    printf "  %-16s  %-16s" "L-chunks/L" "L-chunks/P3"
fi
printf "\n"

for lr in "${LOG_ROWS[@]}"; do
    printf "%-10s" "$lr"
    LAMBDA_MED=""
    CHUNKS_MED=""
    P3_MED=""
    for prover in "${PROVERS[@]}"; do
        MED=$(awk -F'\t' -v lr="$lr" '
            { for (i=1; i<=NF; i++) {
                split($i, kv, "=")
                if (kv[1]=="log_rows") row=kv[2]
                if (kv[1]=="prove_s")  t=kv[2]
              }
              if (row==lr) print t }' "$REPORT_DIR/${prover}.tsv" \
            | sort -g \
            | awk '{ a[NR]=$1 } END {
                if (NR==0) { print "n/a"; exit }
                if (NR%2==1) printf "%.6f\n", a[(NR+1)/2]
                else printf "%.6f\n", (a[NR/2]+a[NR/2+1])/2
              }')
        printf "  %-16s" "$MED"
        if [ "$prover" = "lambda" ];        then LAMBDA_MED="$MED"; fi
        if [ "$prover" = "lambda-chunks" ]; then CHUNKS_MED="$MED"; fi
        if [ "$prover" = "p3" ];            then P3_MED="$MED"; fi
    done
    if $RUN_P3 && [ -n "$LAMBDA_MED" ] && [ -n "$CHUNKS_MED" ] && [ -n "$P3_MED" ]; then
        R1=$(awk -v a="$CHUNKS_MED" -v b="$LAMBDA_MED" 'BEGIN { if (b+0==0) print "n/a"; else printf "%.3fx\n", a/b }')
        R2=$(awk -v a="$CHUNKS_MED" -v b="$P3_MED"     'BEGIN { if (b+0==0) print "n/a"; else printf "%.3fx\n", a/b }')
        printf "  %-16s  %-16s" "$R1" "$R2"
    fi
    printf "\n"
done

echo ""
echo "Done. Report dir: $REPORT_DIR"
echo "On the bench server, scp this dir back to your local checkout when done."
