#!/bin/bash
# Lookup Code Benchmark Script
#
# This script runs benchmarks for the LogUp lookup implementation.
# It uses both Criterion (micro-benchmarks) and Hyperfine (macro-benchmarks).
#
# Usage: ./scripts/bench_lookup.sh [--quick]
#   --quick: Run fewer iterations for faster results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "LogUp Lookup Benchmarks"
echo "=============================================="
echo ""

# Check if hyperfine is installed
if ! command -v hyperfine &> /dev/null; then
    echo "Warning: hyperfine is not installed. Install with: brew install hyperfine"
    echo "Skipping hyperfine benchmarks."
    SKIP_HYPERFINE=true
fi

# Parse arguments
QUICK_MODE=false
if [[ "$1" == "--quick" ]]; then
    QUICK_MODE=true
    echo "Running in quick mode (fewer iterations)"
fi

echo ""
echo "=== Phase 1: Building release binaries ==="
echo ""
cargo build --release -p stark

echo ""
echo "=== Phase 2: Running Criterion micro-benchmarks ==="
echo ""

if [[ "$QUICK_MODE" == true ]]; then
    # Quick mode: run fewer samples
    cargo bench -p stark --bench lookup_bench -- --noplot --sample-size 10
else
    # Full mode: default sample size
    cargo bench -p stark --bench lookup_bench -- --noplot
fi

echo ""
echo "=== Phase 3: Running test-based benchmarks ==="
echo ""

if [[ "$SKIP_HYPERFINE" != true ]]; then
    echo "Measuring multi-table proof test time with hyperfine..."

    if [[ "$QUICK_MODE" == true ]]; then
        # Quick mode: 3 warmup, 5 runs
        hyperfine \
            --warmup 1 \
            --min-runs 3 \
            --export-json "$PROJECT_ROOT/bench_results_multi_table.json" \
            "cargo test -p stark test_multi_table_proof --release -- --nocapture"
    else
        # Full mode: 3 warmup, 10 runs
        hyperfine \
            --warmup 3 \
            --min-runs 10 \
            --export-json "$PROJECT_ROOT/bench_results_multi_table.json" \
            "cargo test -p stark test_multi_table_proof --release -- --nocapture"
    fi

    echo ""
    echo "Multi-table proof benchmark results saved to bench_results_multi_table.json"
fi

echo ""
echo "=== Phase 4: Running full bus_tests suite ==="
echo ""

# Time the full test suite
if [[ "$SKIP_HYPERFINE" != true ]]; then
    if [[ "$QUICK_MODE" == true ]]; then
        hyperfine \
            --warmup 1 \
            --min-runs 3 \
            --export-json "$PROJECT_ROOT/bench_results_bus_tests.json" \
            "cargo test -p stark bus_tests --release"
    else
        hyperfine \
            --warmup 3 \
            --min-runs 10 \
            --export-json "$PROJECT_ROOT/bench_results_bus_tests.json" \
            "cargo test -p stark bus_tests --release"
    fi

    echo ""
    echo "Bus tests benchmark results saved to bench_results_bus_tests.json"
fi

echo ""
echo "=== Benchmark Summary ==="
echo ""
echo "Criterion results: target/criterion/"
if [[ "$SKIP_HYPERFINE" != true ]]; then
    echo "Hyperfine results: bench_results_*.json"
fi
echo ""
echo "To view Criterion HTML reports:"
echo "  open target/criterion/report/index.html"
echo ""
echo "Benchmark complete!"
