#!/bin/bash
# Benchmark TABLE_PARALLELISM on high-core-count machines.
#
# Usage: bench_table_parallelism.sh [elf_path] [runs_per_config]
#
# Tests K = 1, num_cores/4, num_cores/3, num_cores/2 (and a few extras).
# Outputs a summary table at the end.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_ELF="$ROOT_DIR/executor/program_artifacts/asm/fib_iterative_2000k.elf"
ELF=${1:-$DEFAULT_ELF}
RUNS=${2:-2}

NCORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)

# K values to test: 1, cores/4, cores/3, cores/2, and a couple extras
K_VALUES=(1)
for divisor in 4 3 2; do
    val=$((NCORES / divisor))
    [ "$val" -gt 1 ] && K_VALUES+=("$val")
done
# Remove duplicates and sort
K_VALUES=($(printf '%s\n' "${K_VALUES[@]}" | sort -un))

echo "=== Table Parallelism Benchmark ==="
echo "Cores: $NCORES"
echo "ELF: $(basename "$ELF")"
echo "Runs per config: $RUNS"
echo "K values: ${K_VALUES[*]}"
echo ""

# Build once
echo "Building release CLI..."
cargo build --release -p cli --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1
CLI="$ROOT_DIR/target/release/cli"
TMP_DIR="/tmp/bench_tp"
rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"

RESULTS_FILE="$TMP_DIR/results.txt"
echo "K,run,time_s,heap_mb" > "$RESULTS_FILE"

for k in "${K_VALUES[@]}"; do
    echo ""
    echo "--- TABLE_PARALLELISM=$k ---"
    for i in $(seq 1 "$RUNS"); do
        START=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')
        OUTPUT=$(TABLE_PARALLELISM=$k "$CLI" prove "$ELF" -o "$TMP_DIR/proof.bin" --time 2>&1)
        END=$(date +%s%N 2>/dev/null || python3 -c 'import time; print(int(time.time()*1e9))')

        ELAPSED_MS=$(( (END - START) / 1000000 ))
        ELAPSED_S=$(echo "scale=1; $ELAPSED_MS / 1000" | bc)

        HEAP_MB=$(echo "$OUTPUT" | grep -i "peak heap" | grep -oE '[0-9]+' | head -1 || echo "?")

        echo "  run $i/$RUNS: ${ELAPSED_S}s  heap: ${HEAP_MB} MB"
        echo "$k,$i,$ELAPSED_S,$HEAP_MB" >> "$RESULTS_FILE"
    done
done

echo ""
echo "========================================="
echo "  SUMMARY (median of $RUNS runs)"
echo "========================================="
printf "%-6s  %-10s  %-10s\n" "K" "Time(s)" "Heap(MB)"
printf "%-6s  %-10s  %-10s\n" "---" "-------" "-------"

for k in "${K_VALUES[@]}"; do
    times=$(grep "^$k," "$RESULTS_FILE" | cut -d, -f3 | sort -n)
    heaps=$(grep "^$k," "$RESULTS_FILE" | cut -d, -f4 | sort -n)
    # Median: middle value (or first of two middle for even count)
    median_time=$(echo "$times" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    median_heap=$(echo "$heaps" | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}')
    printf "%-6s  %-10s  %-10s\n" "$k" "$median_time" "$median_heap"
done

echo ""
echo "Raw data: $RESULTS_FILE"
