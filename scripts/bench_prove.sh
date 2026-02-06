#!/usr/bin/env bash
#
# Prove Benchmark Script
# Benchmarks the CLI prove command on the current branch using hyperfine.
#
# Usage:
#   ./bench_prove.sh              # Benchmark all bench programs
#   ./bench_prove.sh <program>    # Benchmark a specific program (e.g. keccak)
#
# Environment variables:
#   BENCH_PROVE_RUNS      Number of hyperfine runs (default: 3)
#   BENCH_PROVE_WARMUP    Number of warmup runs (default: 0)
#   BENCH_PROVE_SECURITY  Security preset: fast|standard|maximum (default: fast)
#
# Requires: hyperfine, jq
#

set -euo pipefail

for cmd in hyperfine jq; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: $cmd is required but not installed."
        exit 1
    fi
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_prove"
BENCH_ARTIFACTS_DIR="$ROOT_DIR/executor/program_artifacts/bench"

RUNS="${BENCH_PROVE_RUNS:-3}"
WARMUP="${BENCH_PROVE_WARMUP:-0}"
SECURITY="${BENCH_PROVE_SECURITY:-fast}"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Prove Benchmark Script"
    echo "Benchmarks the CLI prove command using hyperfine."
    echo ""
    echo "Usage:"
    echo "  ./bench_prove.sh              # Benchmark all bench programs"
    echo "  ./bench_prove.sh <program>    # Benchmark a specific program (e.g. keccak)"
    echo ""
    echo "Environment variables:"
    echo "  BENCH_PROVE_RUNS      Number of runs (default: 3)"
    echo "  BENCH_PROVE_WARMUP    Number of warmup runs (default: 0)"
    echo "  BENCH_PROVE_SECURITY  Security preset (default: fast)"
    exit 0
fi

PROGRAM="${1:-}"

# Validate security preset
case "$SECURITY" in
    fast|standard|maximum) ;;
    *)
        echo -e "${RED}Error: Invalid security preset '$SECURITY'. Use fast, standard, or maximum.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}=== Prove Benchmark ===${NC}"
echo -e "${YELLOW}Runs: $RUNS | Warmup: $WARMUP | Security: $SECURITY${NC}"

# Find CLI binary
CLI="$ROOT_DIR/target/release/cli"
if [ ! -f "$CLI" ]; then
    echo -e "${RED}Error: CLI binary not found at $CLI. Build with: cargo build --release -p cli${NC}"
    exit 1
fi

# Collect ELFs to benchmark
if [ -n "$PROGRAM" ]; then
    ELF="$BENCH_ARTIFACTS_DIR/$PROGRAM.elf"
    if [ ! -f "$ELF" ]; then
        echo -e "${RED}Error: Program '$PROGRAM' not found at $ELF${NC}"
        exit 1
    fi
    ELFS=("$ELF")
else
    shopt -s nullglob
    ELFS=("$BENCH_ARTIFACTS_DIR"/*.elf)
    shopt -u nullglob
    if [ ${#ELFS[@]} -eq 0 ]; then
        echo -e "${RED}Error: No ELF files found in $BENCH_ARTIFACTS_DIR. Run: make compile-bench${NC}"
        exit 1
    fi
fi

# Setup output directory
mkdir -p "$TMP_DIR"

# Run benchmarks
for elf in "${ELFS[@]}"; do
    name=$(basename "$elf" .elf)
    proof_file="$TMP_DIR/${name}_proof.cbor"
    echo -e "${YELLOW}--- $name ---${NC}"
    hyperfine \
        --warmup "$WARMUP" \
        --runs "$RUNS" \
        --prepare "rm -f $proof_file" \
        -n "prove $name ($SECURITY)" \
        "$CLI prove $elf --output $proof_file --security $SECURITY" \
        --export-markdown "$TMP_DIR/$name.md" \
        --export-json "$TMP_DIR/$name.json"
done

# Summary
echo ""
echo -e "${GREEN}=== Results ===${NC}"
for elf in "${ELFS[@]}"; do
    name=$(basename "$elf" .elf)
    if [ -f "$TMP_DIR/$name.json" ]; then
        mean=$(jq -r '.results[0].mean' "$TMP_DIR/$name.json")
        stddev=$(jq -r '.results[0].stddev' "$TMP_DIR/$name.json")
        printf "%-20s  mean: %8.2fs  stddev: %6.2fs\n" "$name" "$mean" "$stddev"
    fi
done

echo ""
echo -e "${GREEN}Markdown reports: $TMP_DIR/*.md${NC}"
echo -e "${GREEN}JSON results:     $TMP_DIR/*.json${NC}"
