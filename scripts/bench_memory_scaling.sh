#!/bin/bash
# Benchmark memory scaling: how peak heap grows with program size.
#
# Generates fib programs of various sizes, proves each, and reports
# peak heap (jemalloc) and proving time in a table.
#
# Usage:
#   scripts/bench_memory_scaling.sh [options]
#
# Options:
#   --sizes "160 372 700 1400"    Space-separated list of program sizes in thousands of cycles
#   --max-rows-log2 15            Power of 2 for max rows per table (default: use production defaults)
#   --runs 1                      Number of runs per size (median is reported if >1)
#   --compare <commit>            Also benchmark a comparison commit (e.g. ea254b8, main~3)
#   --output <dir>                Directory for results and artifacts (default: /tmp/bench_scaling)
#
# Examples:
#   # Quick local test
#   scripts/bench_memory_scaling.sh --sizes "160 372 700" --max-rows-log2 15
#
#   # Full scaling sweep on a server
#   scripts/bench_memory_scaling.sh --sizes "160 372 700 1400 2800 5600" --max-rows-log2 14 --runs 3
#
#   # Compare current branch against pre-PR commit
#   scripts/bench_memory_scaling.sh --sizes "160 700 1400" --max-rows-log2 15 --compare ea254b8

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# --- Defaults -----------------------------------------------------------------

SIZES="160 372 700 1400"
MAX_ROWS_LOG2=""
RUNS=1
COMPARE_REF=""
OUTPUT_DIR="/tmp/bench_scaling"

# --- Parse args ---------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --sizes)        SIZES="$2"; shift 2 ;;
        --max-rows-log2) MAX_ROWS_LOG2="$2"; shift 2 ;;
        --runs)         RUNS="$2"; shift 2 ;;
        --compare)      COMPARE_REF="$2"; shift 2 ;;
        --output)       OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help)
            head -25 "$0" | tail -23
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

CURRENT_BRANCH=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD)
CURRENT_SHA=$(git -C "$ROOT_DIR" rev-parse --short HEAD)

# Restore branch on exit if we checked out something else
if [ -n "$COMPARE_REF" ]; then
    trap 'git -C "$ROOT_DIR" checkout "$CURRENT_BRANCH" 2>/dev/null' EXIT
fi

# --- Helpers ------------------------------------------------------------------

median() {
    sort -n "$1" | awk '{a[NR]=$1} END {
        if (NR%2==1) print a[(NR+1)/2];
        else print (a[NR/2]+a[NR/2+1])/2
    }'
}

# Generate a fib iterative .s file for a given number of thousands of cycles.
# 5 instructions per loop iteration, so iterations = cycles_k * 1000 / 5.
generate_fib_elf() {
    local cycles_k=$1
    local elf_path=$2
    local iterations=$(( cycles_k * 1000 / 5 ))
    local asm_path="${elf_path%.elf}.s"

    cat > "$asm_path" << EOF
	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	li	t0, 0
	li	t1, 1
	li	a0, ${iterations}
.loop:
	add	t2, t0, t1
	mv	t0, t1
	mv	t1, t2
	addi	a0, a0, -1
	bnez	a0, .loop
	mv	a0, t1
	li	a7, 5
	ecall
EOF
    clang --target=riscv64 -fuse-ld=lld -nostdlib -Wl,-e,main "$asm_path" -o "$elf_path" 2>/dev/null
}

# Patch max_rows constants in the source tree to use a uniform 2^N for all tables.
# This is needed for compare builds where the CLI doesn't have --max-rows-log2.
patch_max_rows() {
    local log2=$1
    local mod_rs="$ROOT_DIR/prover/src/tables/mod.rs"
    if [ ! -f "$mod_rs" ]; then
        echo -e "${RED}Warning: $mod_rs not found, skipping max_rows patch${NC}"
        return
    fi
    for table in CPU MEMW DVRM MUL LT LOAD BRANCH; do
        # Match "pub const TABLE: usize = 1 << N" and replace N with our value
        sed -i '' -E "s/(pub const ${table}: usize = 1 << )[0-9]+/\1${log2}/" "$mod_rs"
    done
    echo -e "${GREEN}  Patched max_rows to 2^${log2} in mod.rs${NC}"
}

# Restore max_rows constants to their original values (stash/restore via git).
restore_max_rows() {
    git -C "$ROOT_DIR" checkout -- "$ROOT_DIR/prover/src/tables/mod.rs" 2>/dev/null || true
}

# Build the CLI binary for a given label. If compare_ref is set, checks out that ref first.
# If MAX_ROWS_LOG2 is set, patches the source before building (for older commits without the flag).
build_cli() {
    local label=$1
    local ref=${2:-}
    if [ -n "$ref" ]; then
        echo -e "${GREEN}[$label] Checking out $ref...${NC}"
        git -C "$ROOT_DIR" checkout "$ref" 2>/dev/null
    fi
    # For compare builds (or any build), patch max_rows if requested
    if [ -n "$MAX_ROWS_LOG2" ] && [ "$label" = "compare" ]; then
        patch_max_rows "$MAX_ROWS_LOG2"
    fi
    echo -e "${GREEN}[$label] Building CLI (release + jemalloc-stats)...${NC}"
    cargo build --release -p cli --features jemalloc-stats --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1
    cp "$ROOT_DIR/target/release/cli" "$OUTPUT_DIR/cli-$label"
    echo -e "${GREEN}[$label] Binary ready.${NC}"
    # Restore patched files so checkout is clean
    if [ -n "$MAX_ROWS_LOG2" ] && [ "$label" = "compare" ]; then
        restore_max_rows
    fi
}

# Run prove for one size, one binary, collecting results.
bench_size() {
    local cli=$1
    local label=$2
    local cycles_k=$3
    local elf="$OUTPUT_DIR/elfs/fib_${cycles_k}k.elf"

    # Only pass --max-rows-log2 for the current build (which has the flag).
    # Compare builds have max_rows patched at compile time instead.
    local max_rows_flag=""
    if [ -n "$MAX_ROWS_LOG2" ] && [ "$label" = "current" ]; then
        max_rows_flag="--max-rows-log2 $MAX_ROWS_LOG2"
    fi

    local heap_file="$OUTPUT_DIR/results/${label}_${cycles_k}k_heap.txt"
    local time_file="$OUTPUT_DIR/results/${label}_${cycles_k}k_time.txt"
    rm -f "$heap_file" "$time_file"

    for i in $(seq 1 "$RUNS"); do
        local stdout_tmp="$OUTPUT_DIR/tmp_stdout.txt"
        # shellcheck disable=SC2086
        "$cli" prove "$elf" -o "$OUTPUT_DIR/tmp_proof.bin" --time $max_rows_flag \
            > "$stdout_tmp" 2>/dev/null

        local heap_mb
        heap_mb=$(grep -o 'Peak heap: [0-9]*' "$stdout_tmp" | awk '{print $3}')
        local time_s
        time_s=$(grep -o 'Proving time: [0-9.]*' "$stdout_tmp" | awk '{print $3}')

        [ -n "$heap_mb" ] && echo "$heap_mb" >> "$heap_file"
        [ -n "$time_s" ] && echo "$time_s" >> "$time_file"

        if [ "$RUNS" -gt 1 ]; then
            echo -e "  ${YELLOW}[$label] ${cycles_k}k run $i/$RUNS: ${time_s}s, ${heap_mb} MB${NC}"
        fi

        rm -f "$OUTPUT_DIR/tmp_proof.bin" "$stdout_tmp"
    done
}

# --- Setup --------------------------------------------------------------------

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/elfs" "$OUTPUT_DIR/results"

echo -e "${BOLD}=== Memory Scaling Benchmark ===${NC}"
echo "  Sizes (k cycles): $SIZES"
echo "  Max rows log2:    ${MAX_ROWS_LOG2:-default}"
echo "  Runs per size:    $RUNS"
echo "  Compare ref:      ${COMPARE_REF:-none}"
echo ""

# --- Generate ELFs ------------------------------------------------------------

echo -e "${GREEN}Generating fib ELFs...${NC}"
for k in $SIZES; do
    generate_fib_elf "$k" "$OUTPUT_DIR/elfs/fib_${k}k.elf"
done
echo ""

# --- Build binaries -----------------------------------------------------------

build_cli "current"
if [ -n "$COMPARE_REF" ]; then
    build_cli "compare" "$COMPARE_REF"
    git -C "$ROOT_DIR" checkout "$CURRENT_BRANCH" 2>/dev/null
fi
echo ""

# --- Run benchmarks -----------------------------------------------------------

run_all_sizes() {
    local cli=$1
    local label=$2
    echo -e "${BOLD}--- Benchmarking: $label ---${NC}"
    for k in $SIZES; do
        bench_size "$cli" "$label" "$k"
    done
}

run_all_sizes "$OUTPUT_DIR/cli-current" "current"
if [ -n "$COMPARE_REF" ]; then
    run_all_sizes "$OUTPUT_DIR/cli-compare" "compare"
fi

# --- Print results ------------------------------------------------------------

print_table() {
    local label=$1
    echo -e "\n${BOLD}  $label${NC}"
    printf "  %-12s %12s %12s %12s\n" "Program" "Time (s)" "Heap (MB)" "Delta (MB)"
    printf "  %-12s %12s %12s %12s\n" "-------" "--------" "---------" "----------"

    local prev_heap=0
    for k in $SIZES; do
        local heap_file="$OUTPUT_DIR/results/${label}_${k}k_heap.txt"
        local time_file="$OUTPUT_DIR/results/${label}_${k}k_time.txt"

        local heap_val="N/A"
        local time_val="N/A"
        local delta="—"

        if [ -f "$heap_file" ]; then
            if [ "$RUNS" -gt 1 ]; then
                heap_val=$(median "$heap_file")
            else
                heap_val=$(cat "$heap_file")
            fi
        fi
        if [ -f "$time_file" ]; then
            if [ "$RUNS" -gt 1 ]; then
                time_val=$(median "$time_file")
            else
                time_val=$(cat "$time_file")
            fi
        fi

        if [ "$prev_heap" -gt 0 ] 2>/dev/null && [ "$heap_val" != "N/A" ]; then
            delta="+$(( heap_val - prev_heap ))"
        fi

        printf "  %-12s %12s %12s %12s\n" "${k}k" "$time_val" "$heap_val" "$delta"

        if [ "$heap_val" != "N/A" ]; then
            prev_heap=$heap_val
        fi
    done
}

echo ""
echo -e "${BOLD}=== Results ===${NC}"
if [ -n "$MAX_ROWS_LOG2" ]; then
    echo "  Max rows: 2^${MAX_ROWS_LOG2} = $(( 1 << MAX_ROWS_LOG2 ))"
else
    echo "  Max rows: production defaults"
fi
echo "  Runs:     $RUNS"

print_table "current"
if [ -n "$COMPARE_REF" ]; then
    print_table "compare"

    # Summary: growth rate comparison
    echo ""
    echo -e "${BOLD}  Scaling comparison:${NC}"
    read -ra SIZES_ARR <<< "$SIZES"
    FIRST_K=${SIZES_ARR[0]}
    LAST_K=${SIZES_ARR[${#SIZES_ARR[@]}-1]}

    cur_first=$(cat "$OUTPUT_DIR/results/current_${FIRST_K}k_heap.txt" 2>/dev/null || echo 0)
    cur_last=$(cat "$OUTPUT_DIR/results/current_${LAST_K}k_heap.txt" 2>/dev/null || echo 0)
    cmp_first=$(cat "$OUTPUT_DIR/results/compare_${FIRST_K}k_heap.txt" 2>/dev/null || echo 0)
    cmp_last=$(cat "$OUTPUT_DIR/results/compare_${LAST_K}k_heap.txt" 2>/dev/null || echo 0)

    if [ "$cur_first" -gt 0 ] && [ "$cmp_first" -gt 0 ]; then
        cur_growth=$((cur_last - cur_first))
        cmp_growth=$((cmp_last - cmp_first))
        cycle_range=$(( (LAST_K - FIRST_K) ))
        cur_rate=$(awk "BEGIN {printf \"%.1f\", $cur_growth / $cycle_range}")
        cmp_rate=$(awk "BEGIN {printf \"%.1f\", $cmp_growth / $cycle_range}")
        echo "    current: ${FIRST_K}k→${LAST_K}k = +${cur_growth} MB  (${cur_rate} MB/k-cycle)"
        echo "    compare: ${FIRST_K}k→${LAST_K}k = +${cmp_growth} MB  (${cmp_rate} MB/k-cycle)"
    fi
fi

echo ""
echo "Raw data in $OUTPUT_DIR/results/"
