#!/bin/bash
# Run heap and timing profiles on all prepared bench programs.
# Output:
#   /tmp/bench_heap_profile/<program>/
#   /tmp/bench_timing_profile/<program>/
#
# Flags:
#   --no-heap, --no-timing  skip either profile type
#   --program <name>        run only the given program; repeat to select several
#                           (default: all programs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

HEAP=true
TIMING=true
ONLY_LIST=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-heap)   HEAP=false; shift ;;
        --no-timing) TIMING=false; shift ;;
        --program)   ONLY_LIST+=("$2"); shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# program env_var sizes
# env_var="-" means fib_iterative (prebuilt ELFs, no build step)
CONFIGS=(
    "fib_iterative   -               500k 1M 2M"
    "keccak          ITERATIONS      200 500 1000"
    "modular_exp     NUM_ITERATIONS  5000 10000 20000 40000"
    "bitwise_ops     ITERATIONS      50000 100000 200000 400000"
    "matrix_multiply SIZE            20 40 60 80"
)

if [ ${#ONLY_LIST[@]} -gt 0 ]; then
    filtered=()
    for only in "${ONLY_LIST[@]}"; do
        found=false
        for config in "${CONFIGS[@]}"; do
            read -r prog _ <<< "$config"
            if [ "$prog" = "$only" ]; then
                filtered+=("$config")
                found=true
                break
            fi
        done
        if ! $found; then
            echo "Error: unknown program '$only'" >&2
            exit 1
        fi
    done
    CONFIGS=("${filtered[@]}")
fi

GREEN='\033[0;32m'
NC='\033[0m'

FEATURES="instruments"
$HEAP && FEATURES="jemalloc-stats,instruments"

# Clear stale results so only the current run's data remains.
$HEAP   && rm -rf /tmp/bench_heap_profile
$TIMING && rm -rf /tmp/bench_timing_profile

echo -e "${GREEN}Building CLI with $FEATURES...${NC}"
cargo build --release -p cli --features "$FEATURES" \
    --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1

for config in "${CONFIGS[@]}"; do
    read -r prog env_var sizes <<< "$config"

    echo ""
    echo -e "${GREEN}=== $prog ===${NC}"

    if [ "$env_var" != "-" ]; then
        bash "$SCRIPT_DIR/build_bench_sizes.sh" "$prog" "$env_var" $sizes
    fi

    if $HEAP; then
        bash "$SCRIPT_DIR/bench_heap_profile.sh" --no-build \
            --program "$prog" --programs "$sizes"
    fi

    if $TIMING; then
        bash "$SCRIPT_DIR/bench_timing_profile.sh" --no-build \
            --program "$prog" --programs "$sizes"
    fi
done

echo ""
echo "Results:"
$HEAP   && echo "  Heap:   /tmp/bench_heap_profile/<program>/"
$TIMING && echo "  Timing: /tmp/bench_timing_profile/<program>/"
