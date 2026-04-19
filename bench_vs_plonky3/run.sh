#!/bin/bash
# Benchmark: Lambda STARK vs Plonky3 — single-shot prove time on the shared
# Fibonacci AIR (columns = 2 * num_sequences, blowup = 2, fri_queries = 219).
#
# Usage:
#   ./bench_vs_plonky3/run.sh [--log-rows K ...] [--num-sequences N] [--runs N]
#                             [--lambda-only | --p3-only] [--report-dir DIR]
#                             [--scalar] [--no-color]
#
# Defaults: --log-rows 19, --num-sequences 16, --runs 3.
# With multiple --log-rows values, prints one median row per size.
#
# --scalar: on x86_64 drops AVX2 / AVX-512 so Goldilocks (and most of Keccak)
# run scalar; residual SSE2 in p3-keccak remains. Triggers a rebuild when
# toggling; subsequent runs with the same RUSTFLAGS are cached.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_p3"
REPORT_DIR=""
NO_COLOR=false
SCALAR=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

LOG_ROWS=()
NUM_SEQUENCES=16
RUNS=3
RUN_LAMBDA=true
RUN_P3=true

# --- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-rows)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LOG_ROWS+=("$1")
                shift
            done
            ;;
        --num-sequences)
            if [[ $# -lt 2 ]]; then echo "--num-sequences requires an argument"; exit 1; fi
            NUM_SEQUENCES=$2
            shift 2
            ;;
        --runs)
            if [[ $# -lt 2 ]]; then echo "--runs requires an argument"; exit 1; fi
            RUNS=$2
            shift 2
            ;;
        --lambda-only)
            RUN_P3=false
            shift
            ;;
        --p3-only)
            RUN_LAMBDA=false
            shift
            ;;
        --report-dir)
            if [[ $# -lt 2 ]]; then echo "--report-dir requires an argument"; exit 1; fi
            REPORT_DIR=$2
            shift 2
            ;;
        --scalar)
            SCALAR=true
            shift
            ;;
        --no-color)
            NO_COLOR=true
            shift
            ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ${#LOG_ROWS[@]} -eq 0 ]; then
    LOG_ROWS=(19)
fi

if ! $RUN_LAMBDA && ! $RUN_P3; then
    echo "At least one prover must be enabled"
    exit 1
fi

if [ "$RUNS" -lt 1 ]; then
    echo "--runs must be >= 1"
    exit 1
fi

if $NO_COLOR; then
    RED=''
    GREEN=''
    YELLOW=''
    BOLD=''
    NC=''
fi

mkdir -p "$TMP_DIR"
rm -rf "$TMP_DIR"/*

if [ -n "$REPORT_DIR" ]; then
    mkdir -p "$REPORT_DIR/raw"
fi

# --- Scalar (no SIMD) toggle ------------------------------------------------
# When --scalar is on, disable AVX2/AVX-512 so Goldilocks (and most of Keccak)
# run scalar for an apples-to-apples comparison against Lambda STARK. The
# residual SSE2 path on p3-keccak is intentionally left enabled — its
# contribution to total prove time is ~7%.
# Cargo caches per-RUSTFLAGS, so toggling scalar vs vector triggers a rebuild
# on first use but is cached afterwards.
SCALAR_RUSTFLAGS=""
SCALAR_ACTIVE=false
if $SCALAR; then
    case "$(uname -m)" in
        x86_64|amd64)
            SCALAR_RUSTFLAGS="-C target-feature=-avx2,-avx512f"
            SCALAR_ACTIVE=true
            ;;
        *)
            echo "warning: --scalar: only supported on x86_64; host is $(uname -m), not pinning RUSTFLAGS" >&2
            ;;
    esac
    if [ -n "$SCALAR_RUSTFLAGS" ]; then
        if [ -n "${RUSTFLAGS:-}" ]; then
            export RUSTFLAGS="${RUSTFLAGS} ${SCALAR_RUSTFLAGS}"
        else
            export RUSTFLAGS="$SCALAR_RUSTFLAGS"
        fi
    fi
fi

# --- Build ------------------------------------------------------------------
echo -e "${BOLD}=== STARK prove benchmark: Lambda vs Plonky3 ===${NC}"
echo -e "  log-rows:       ${YELLOW}${LOG_ROWS[*]}${NC}"
echo -e "  num-sequences:  ${YELLOW}${NUM_SEQUENCES}${NC}  (columns = $((2 * NUM_SEQUENCES)))"
echo -e "  runs/size:      ${YELLOW}${RUNS}${NC}  (median reported)"
echo -e "  p3 extension:   ${YELLOW}degree 3 (forked p3-goldilocks, matches Lambda)${NC}"
if $SCALAR_ACTIVE; then
    echo -e "  scalar mode:    ${YELLOW}on${NC}  (arch=$(uname -m), RUSTFLAGS=\"${RUSTFLAGS:-}\")"
elif $SCALAR; then
    echo -e "  scalar mode:    ${YELLOW}requested (unsupported on $(uname -m))${NC}  (SIMD enabled, compiler default)"
else
    echo -e "  scalar mode:    ${YELLOW}off${NC}  (SIMD enabled, compiler default)"
fi
echo ""

echo -e "${GREEN}[build]${NC} prove_bench"
cargo build --release -p bench-vs-plonky3 --bin prove_bench \
    --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -5

# Resolve the actual target directory via cargo metadata so we find the binary
# whether cargo used ./target/ (default) or a custom CARGO_TARGET_DIR.
TARGET_DIR=$(cargo metadata --manifest-path "$ROOT_DIR/Cargo.toml" \
    --format-version 1 --no-deps 2>/dev/null \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])' \
    2>/dev/null || echo "$ROOT_DIR/target")
BIN="$TARGET_DIR/release/prove_bench"
if [ ! -x "$BIN" ]; then
    echo -e "${RED}[build] prove_bench not produced at $BIN${NC}"
    exit 1
fi

# --- Helpers ----------------------------------------------------------------
extract_proving_time() {
    sed -nE '/Proving time: [0-9.]+s/ {
        s/.*Proving time: ([0-9.]+)s.*/\1/
        p
        q
    }'
}

median_of() {
    # prints median of the given numeric arguments (rounded to 3 decimals).
    # Uses shell `sort -g` for portability (macOS awk lacks gawk's asort).
    printf '%s\n' "$@" | LC_ALL=C sort -g | LC_NUMERIC=C awk '
        { a[NR] = $0 + 0 }
        END {
            if (NR == 0) { print "n/a"; exit }
            if (NR % 2 == 1) {
                printf "%.3f\n", a[(NR + 1) / 2]
            } else {
                printf "%.3f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2
            }
        }'
}

ratio_fmt() {
    LC_NUMERIC=C awk -v num="$1" -v den="$2" 'BEGIN {
        if (den + 0 == 0) { print "n/a"; exit }
        printf "%.3f\n", num / den
    }'
}

# --- Run benchmark ----------------------------------------------------------

RESULT_LOG_ROWS=()
RESULT_ROWS=()
RESULT_LAMBDA=()
RESULT_P3=()
RESULT_RATIO=()

run_prover() {
    local prover=$1   # lambda | p3
    local log_rows=$2
    local times=()
    for run_i in $(seq 1 "$RUNS"); do
        local out_file="$TMP_DIR/${prover}_${log_rows}_${run_i}.stdout"
        if ! "$BIN" --prover "$prover" \
                --log-rows "$log_rows" \
                --num-sequences "$NUM_SEQUENCES" > "$out_file" 2>&1; then
            echo -e "  ${RED}[${prover}] FAILED on log-rows=${log_rows} run ${run_i}${NC}"
            cat "$out_file"
            exit 1
        fi
        local t
        t=$(extract_proving_time < "$out_file")
        if [ -z "$t" ]; then
            echo -e "  ${RED}[${prover}] could not parse proving time (log-rows=${log_rows}, run ${run_i})${NC}"
            cat "$out_file"
            exit 1
        fi
        times+=("$t")
        if [ -n "$REPORT_DIR" ]; then
            cp "$out_file" "$REPORT_DIR/raw/${prover}_log${log_rows}_run${run_i}.stdout"
        fi
    done
    median_of "${times[@]}"
    printf '%s\n' "${times[@]}" > "$TMP_DIR/${prover}_${log_rows}.times"
}

for lr in "${LOG_ROWS[@]}"; do
    rows=$((1 << lr))
    echo -e "${BOLD}--- log-rows=${lr}  (rows = ${rows}) ---${NC}"

    lambda_median="n/a"
    p3_median="n/a"

    if $RUN_LAMBDA; then
        echo -ne "  ${GREEN}[lambda]${NC} "
        lambda_median=$(run_prover lambda "$lr")
        echo -e "median ${BOLD}${lambda_median}s${NC} from $RUNS runs: $(paste -sd, "$TMP_DIR/lambda_${lr}.times")"
    fi

    if $RUN_P3; then
        echo -ne "  ${GREEN}[p3]${NC}     "
        p3_median=$(run_prover p3 "$lr")
        echo -e "median ${BOLD}${p3_median}s${NC} from $RUNS runs: $(paste -sd, "$TMP_DIR/p3_${lr}.times")"
    fi

    local_ratio="n/a"
    if $RUN_LAMBDA && $RUN_P3; then
        local_ratio=$(ratio_fmt "$lambda_median" "$p3_median")
    fi

    RESULT_LOG_ROWS+=("$lr")
    RESULT_ROWS+=("$rows")
    RESULT_LAMBDA+=("$lambda_median")
    RESULT_P3+=("$p3_median")
    RESULT_RATIO+=("$local_ratio")
done

# --- Summary table ----------------------------------------------------------

echo ""
echo -e "${BOLD}=== Summary ===${NC}"
if $RUN_LAMBDA && $RUN_P3; then
    printf "  %-9s  %-12s  %14s  %14s  %10s\n" "log-rows" "rows" "Lambda (s)" "P3 (s)" "L/P3"
    printf "  %-9s  %-12s  %14s  %14s  %10s\n" "--------" "----" "----------" "------" "----"
else
    printf "  %-9s  %-12s  %14s\n" "log-rows" "rows" "Time (s)"
    printf "  %-9s  %-12s  %14s\n" "--------" "----" "--------"
fi

for i in "${!RESULT_LOG_ROWS[@]}"; do
    lr="${RESULT_LOG_ROWS[$i]}"
    rows="${RESULT_ROWS[$i]}"
    lt="${RESULT_LAMBDA[$i]}"
    pt="${RESULT_P3[$i]}"
    rt="${RESULT_RATIO[$i]}"
    if $RUN_LAMBDA && $RUN_P3; then
        color=$GREEN
        verdict="Lambda faster"
        if awk -v l="$lt" -v p="$pt" 'BEGIN{ exit !(l+0 > p+0) }'; then
            color=$RED
            verdict="P3 faster"
        fi
        printf "  %-9s  %-12s  %13ss  %13ss  ${color}%9sx${NC}  (${color}%s${NC})\n" \
            "$lr" "$rows" "$lt" "$pt" "$rt" "$verdict"
    elif $RUN_LAMBDA; then
        printf "  %-9s  %-12s  %13ss\n" "$lr" "$rows" "$lt"
    else
        printf "  %-9s  %-12s  %13ss\n" "$lr" "$rows" "$pt"
    fi
done

echo ""
if $RUN_LAMBDA && $RUN_P3; then
    echo -e "Timing window: single-shot end-to-end prove."
fi

# --- Machine-readable report ------------------------------------------------

if [ -n "$REPORT_DIR" ]; then
    # Slash-joined helpers for metrics.txt (mirrors the format used by
    # bench_vs/run.sh).
    join_slash() {
        local joined=""
        for value in "$@"; do
            joined="${joined:+$joined/}$value"
        done
        printf "%s\n" "$joined"
    }

    {
        printf "log_rows\trows\tlambda_median_s\tp3_median_s\tratio_lambda_over_p3\truns\n"
        for i in "${!RESULT_LOG_ROWS[@]}"; do
            printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
                "${RESULT_LOG_ROWS[$i]}" \
                "${RESULT_ROWS[$i]}" \
                "${RESULT_LAMBDA[$i]}" \
                "${RESULT_P3[$i]}" \
                "${RESULT_RATIO[$i]}" \
                "$RUNS"
        done
    } > "$REPORT_DIR/results.tsv"

    # Capture commit + timestamp so the artifact is self-describing.
    git_sha="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    git_dirty="clean"
    if ! git -C "$ROOT_DIR" diff --quiet HEAD -- 2>/dev/null; then
        git_dirty="dirty"
    fi
    timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    {
        echo "timestamp_utc=$timestamp_utc"
        echo "git_sha=$git_sha"
        echo "git_tree=$git_dirty"
        echo "arch=$(uname -m)"
        echo "num_sequences=$NUM_SEQUENCES"
        echo "columns=$((2 * NUM_SEQUENCES))"
        echo "blowup=2"
        echo "fri_queries=219"
        echo "grinding=0"
        echo "runs_per_size=$RUNS"
        echo "p3_extension=degree3_fork"
        if $SCALAR_ACTIVE; then
            echo "scalar=on"
            echo "rustflags=$SCALAR_RUSTFLAGS"
        elif $SCALAR; then
            echo "scalar=requested_unsupported"
        else
            echo "scalar=off"
        fi
        echo "timing_window=single_shot_end_to_end_prove_no_verify"
        echo "log_rows_series=$(join_slash "${RESULT_LOG_ROWS[@]}")"
        echo "rows_series=$(join_slash "${RESULT_ROWS[@]}")"
        echo "lambda_medians=$(join_slash "${RESULT_LAMBDA[@]}")"
        echo "p3_medians=$(join_slash "${RESULT_P3[@]}")"
        echo "ratios_lambda_over_p3=$(join_slash "${RESULT_RATIO[@]}")"
    } > "$REPORT_DIR/metrics.txt"
fi
