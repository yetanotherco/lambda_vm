#!/bin/bash
# Run heap and timing profiles on all prepared bench programs.
# Output:
#   /tmp/bench_heap_profile/<program>/
#   /tmp/bench_timing_profile/<program>/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# program env_var sizes
# env_var="-" means fib_iterative (prebuilt ELFs, no build step)
CONFIGS=(
    "fib_iterative   -               500k 1M 2M"
    "keccak          ITERATIONS      200 500 1000"
    "modular_exp     NUM_ITERATIONS  5000 10000 20000 40000"
    "bitwise_ops     ITERATIONS      50000 100000 200000 400000"
    "matrix_multiply SIZE            20 40 60 80"
)

GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Building CLI with jemalloc-stats + instruments...${NC}"
cargo build --release -p cli --features jemalloc-stats,instruments \
    --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1

for config in "${CONFIGS[@]}"; do
    read -r prog env_var sizes <<< "$config"

    echo ""
    echo -e "${GREEN}=== $prog ===${NC}"

    if [ "$env_var" != "-" ]; then
        bash "$SCRIPT_DIR/build_bench_sizes.sh" "$prog" "$env_var" $sizes
    fi

    bash "$SCRIPT_DIR/bench_heap_profile.sh" --no-build \
        --program "$prog" --programs "$sizes"

    bash "$SCRIPT_DIR/bench_timing_profile.sh" --no-build \
        --program "$prog" --programs "$sizes"
done

echo ""
echo "Results:"
echo "  Heap:   /tmp/bench_heap_profile/<program>/"
echo "  Timing: /tmp/bench_timing_profile/<program>/"
