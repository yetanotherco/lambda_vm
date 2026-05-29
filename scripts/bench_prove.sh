#!/bin/bash
# Benchmark the prover: wall-clock time + peak heap (jemalloc).
#
# Usage: bench_prove.sh [elf_path] [runs=1] [base_branch=main | --no-compare] [--instruments]
#
# If on a feature branch, automatically benchmarks base_branch too and prints a comparison.
# Pass --no-compare to skip comparison and only benchmark the current branch.
# Pass --instruments to also build with instruments and capture detailed timing breakdown.
#
# Peak heap is deterministic (jemalloc stats.allocated high-water mark, 10ms polling).
# Peak RSS is collected in raw data for reference but not shown in summary (it's noisy).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_prove"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# Parse --instruments from any position in args
INSTRUMENTS=false
POSITIONAL=()
for arg in "$@"; do
    if [ "$arg" = "--instruments" ]; then
        INSTRUMENTS=true
    else
        POSITIONAL+=("$arg")
    fi
done

DEFAULT_ELF="$ROOT_DIR/executor/program_artifacts/asm/fib_iterative_372k.elf"
ELF=${POSITIONAL[0]:-$DEFAULT_ELF}
RUNS=${POSITIONAL[1]:-1}
BASE_BRANCH=${POSITIONAL[2]:-main}
OUTPUT="$TMP_DIR/proof.bin"

CURRENT_BRANCH=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD)

# Restore branch on exit
trap 'git -C "$ROOT_DIR" checkout "$CURRENT_BRANCH" 2>/dev/null' EXIT

# Detect OS for time command (getrusage-based peak RSS, collected for raw data)
if [[ "$(uname)" == "Darwin" ]]; then
    TIME_CMD="command time -l"
    RSS_UNIT="bytes"
else
    TIME_CMD="/usr/bin/time -v"
    RSS_UNIT="kB"
fi

# --- Build phase -----------------------------------------------------------

rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"

build_branch() {
    local branch=$1 label=$2
    echo -e "${GREEN}[$label] Checking out $branch...${NC}"
    git -C "$ROOT_DIR" checkout "$branch"
    # Extra cargo features applied ONLY to the PR/feature side (not the
    # base branch, which may lack them — e.g. `phased-fft`). Set via
    #   PR_EXTRA_FEATURES=phased-fft ./scripts/bench_prove.sh ...
    local features="jemalloc-stats"
    if [ "$label" != "base" ] && [ -n "${PR_EXTRA_FEATURES:-}" ]; then
        features="$features $PR_EXTRA_FEATURES"
    fi
    echo -e "${GREEN}[$label] Building release CLI (--features \"$features\")...${NC}"
    cargo build --release -p cli --features "$features" --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1
    cp "$ROOT_DIR/target/release/cli" "$TMP_DIR/cli-$label"
    echo -e "${GREEN}[$label] Binary saved.${NC}"
}

COMPARE=false
if [ "$BASE_BRANCH" = "--no-compare" ] || [ "$CURRENT_BRANCH" = "$BASE_BRANCH" ]; then
    build_branch "$CURRENT_BRANCH" "current"
else
    COMPARE=true
    build_branch "$CURRENT_BRANCH" "feature"
    build_branch "$BASE_BRANCH" "base"
    git -C "$ROOT_DIR" checkout "$CURRENT_BRANCH"
fi

# --- Benchmark function ----------------------------------------------------

# Extract peak RSS from `command time` output (raw data only, not shown in summary).
parse_peak_rss_kb() {
    local time_output=$1
    if [[ "$RSS_UNIT" == "bytes" ]]; then
        local bytes
        bytes=$(grep -i "maximum resident set size" "$time_output" | awk '{print $1}')
        echo $((bytes / 1024))
    else
        grep -i "maximum resident set size" "$time_output" | awk '{print $NF}'
    fi
}

bench_one() {
    local cli=$1 label=$2
    echo -e "\n${BOLD}=== Benchmarking: $label ===${NC}"

    for i in $(seq 1 "$RUNS"); do
        echo -e "${YELLOW}--- $label run $i/$RUNS ---${NC}"

        local time_output="$TMP_DIR/time_tmp.txt"
        local stdout_output="$TMP_DIR/stdout_tmp.txt"

        START=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')

        # stdout has "Peak heap:" and "Proving time:", stderr has getrusage from `time`
        $TIME_CMD "$cli" prove "$ELF" -o "$OUTPUT" --time > "$stdout_output" 2> "$time_output"
        EXIT_CODE=$?

        END=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')

        if [ "$EXIT_CODE" -ne 0 ]; then
            echo -e "${RED}ERROR: prove exited with code $EXIT_CODE${NC}"
            cat "$time_output"
            cat "$stdout_output"
            continue
        fi

        ELAPSED_MS=$(( (END - START) / 1000000 ))
        echo "$ELAPSED_MS" >> "$TMP_DIR/${label}_times.txt"

        # Peak RSS (raw data only)
        RSS_KB=$(parse_peak_rss_kb "$time_output")
        echo "$RSS_KB" >> "$TMP_DIR/${label}_rss.txt"

        # Peak heap from jemalloc (deterministic)
        PEAK_HEAP_MB=$(grep -o 'Peak heap: [0-9]*' "$stdout_output" | awk '{print $3}')
        if [ -n "$PEAK_HEAP_MB" ]; then
            echo "$PEAK_HEAP_MB" >> "$TMP_DIR/${label}_heap.txt"
        fi

        ELAPSED_S=$(awk "BEGIN {printf \"%.1f\", $ELAPSED_MS / 1000}")
        HEAP_STR=""
        if [ -n "$PEAK_HEAP_MB" ]; then
            HEAP_STR="  heap: ${PEAK_HEAP_MB} MB"
        fi
        echo "  Time: ${ELAPSED_S}s${HEAP_STR}"

        rm -f "$OUTPUT" "$time_output" "$stdout_output"
    done
}

# --- Run benchmarks --------------------------------------------------------

if $COMPARE; then
    bench_one "$TMP_DIR/cli-base"    "base"
    bench_one "$TMP_DIR/cli-feature" "feature"
else
    bench_one "$TMP_DIR/cli-current" "current"
fi

# --- Statistics helpers ----------------------------------------------------

median() {
    sort -n "$1" | awk '{a[NR]=$1} END {
        if (NR%2==1) print a[(NR+1)/2];
        else print (a[NR/2]+a[NR/2+1])/2
    }'
}

mean() {
    awk '{s+=$1} END {printf "%.0f", s/NR}' "$1"
}

stddev() {
    awk '{s+=$1; ss+=$1*$1} END {
        m=s/NR; printf "%.0f", sqrt(ss/NR - m*m)
    }' "$1"
}

cv_pct() {
    awk '{s+=$1; ss+=$1*$1} END {
        m=s/NR; sd=sqrt(ss/NR - m*m);
        if (m > 0) printf "%.1f", sd*100/m;
        else printf "0.0";
    }' "$1"
}

# Format a signed percentage: "+0.3%" or "-12.6%"
signed_pct() {
    local diff=$1 base=$2
    awk "BEGIN {
        pct = $diff * 100 / $base;
        if (pct > 0) printf \"+%.1f%%\", pct;
        else if (pct < 0) printf \"%.1f%%\", pct;
        else printf \"+0.0%%\";
    }"
}

# --- Results ---------------------------------------------------------------

echo ""
echo -e "${BOLD}=== Results ===${NC}"
echo -e "Program: $(basename "$ELF")"
echo -e "Runs:    $RUNS"
echo ""

print_stats() {
    local label=$1
    local time_file="$TMP_DIR/${label}_times.txt"
    local heap_file="$TMP_DIR/${label}_heap.txt"

    local mean_ms=$(mean "$time_file")
    local median_ms=$(median "$time_file")
    local mean_s=$(awk "BEGIN {printf \"%.1f\", $mean_ms / 1000}")
    local median_s=$(awk "BEGIN {printf \"%.1f\", $median_ms / 1000}")

    local heap_str="N/A"
    if [ -f "$heap_file" ]; then
        local median_heap=$(median "$heap_file")
        heap_str="${median_heap} MB"
    fi

    local cv_str=""
    local num_runs=$(wc -l < "$time_file" | tr -d ' ')
    if [ "$num_runs" -gt 1 ]; then
        local cv=$(cv_pct "$time_file")
        cv_str="  CV: ${cv}%"
    fi

    printf "  %-10s  time(mean): %7ss  time(median): %7ss  heap(median): %s%s\n" \
        "$label" "$mean_s" "$median_s" "$heap_str" "$cv_str"
}

if $COMPARE; then
    print_stats "base"
    print_stats "feature"

    echo ""
    echo -e "${BOLD}  Comparison (feature vs $BASE_BRANCH):${NC}"

    BASE_MEAN=$(mean "$TMP_DIR/base_times.txt")
    FEAT_MEAN=$(mean "$TMP_DIR/feature_times.txt")
    TIME_DIFF=$((FEAT_MEAN - BASE_MEAN))
    echo -e "    Time: $(signed_pct "$TIME_DIFF" "$BASE_MEAN")"

    if [ -f "$TMP_DIR/base_heap.txt" ] && [ -f "$TMP_DIR/feature_heap.txt" ]; then
        BASE_HEAP=$(median "$TMP_DIR/base_heap.txt")
        FEAT_HEAP=$(median "$TMP_DIR/feature_heap.txt")
        HEAP_DIFF=$((FEAT_HEAP - BASE_HEAP))
        echo -e "    Heap: ${HEAP_DIFF} MB ($(signed_pct "$HEAP_DIFF" "$BASE_HEAP")) ← DETERMINISTIC"
    fi
else
    print_stats "current"
fi

# --- Instruments run (optional) -----------------------------------------------

if $INSTRUMENTS; then
    echo ""
    echo -e "${BOLD}=== Instruments Breakdown ===${NC}"
    echo -e "${GREEN}Building with --features instruments...${NC}"
    cargo build --release -p cli --features instruments --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1

    INSTRUMENTS_OUTPUT="$TMP_DIR/instruments.txt"
    echo -e "${GREEN}Running instrumented prove...${NC}"
    "$ROOT_DIR/target/release/cli" prove "$ELF" -o "$TMP_DIR/proof_instruments.bin" 2>&1 | tee "$INSTRUMENTS_OUTPUT"
    rm -f "$TMP_DIR/proof_instruments.bin"

    echo ""
    echo -e "Instruments report saved to: ${BOLD}$INSTRUMENTS_OUTPUT${NC}"
fi

echo ""
echo "Raw data in $TMP_DIR/"
