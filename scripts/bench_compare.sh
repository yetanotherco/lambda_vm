#!/usr/bin/env bash
#
# Benchmark Comparison Script
# Compares CLI execution performance between two git branches using hyperfine.
#
# Usage:
#   ./bench_compare.sh                    # Compare main vs current branch
#   ./bench_compare.sh <branch>           # Compare main vs <branch>
#   ./bench_compare.sh <branch1> <branch2> # Compare <branch1> vs <branch2>
#
# Note: Benchmarks are compiled from the current branch, not from the branches being compared.
#
# Requires: hyperfine, jq, bc
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_compare"
BENCH_ARTIFACTS_DIR="$ROOT_DIR/executor/program_artifacts/bench"

CURRENT_BRANCH=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD)
# Parse arguments
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    echo "Benchmark Comparison Script"
    echo "Compares CLI execution performance between two git branches using hyperfine."
    echo ""
    echo "Usage:"
    echo "  ./bench_compare.sh                    # Compare main vs current branch"
    echo "  ./bench_compare.sh <branch>           # Compare main vs <branch>"
    echo "  ./bench_compare.sh <branch1> <branch2> # Compare <branch1> vs <branch2>"
    echo ""
    echo "Note: Benchmarks are compiled from the current branch, not from the branches being compared."
    echo ""
    echo "Requires: hyperfine, jq, bc"
    exit 0
fi

# Cleanup on failure - restore original branch
trap 'echo -e "${YELLOW}Restoring original branch...${NC}"; git -C "$ROOT_DIR" checkout "$CURRENT_BRANCH" 2>/dev/null' EXIT


if [ $# -eq 0 ]; then
    BASE_BRANCH="main"
    COMPARE_BRANCH="$CURRENT_BRANCH"
elif [ $# -eq 1 ]; then
    BASE_BRANCH="main"
    COMPARE_BRANCH="$1"
else
    BASE_BRANCH="$1"
    COMPARE_BRANCH="$2"
fi

echo -e "${GREEN}=== Benchmark Comparison Script ===${NC}"
echo -e "${YELLOW}Comparing: $BASE_BRANCH vs $COMPARE_BRANCH${NC}"

# Validate
if [ "$BASE_BRANCH" = "$COMPARE_BRANCH" ]; then
    echo -e "${RED}Error: Cannot compare a branch against itself.${NC}"
    exit 1
fi

if ! git -C "$ROOT_DIR" diff-index --quiet HEAD; then
    echo -e "${RED}Error: You have uncommitted changes. Please commit or stash them first.${NC}"
    exit 1
fi

if ! git -C "$ROOT_DIR" rev-parse --verify "$BASE_BRANCH" &>/dev/null; then
    echo -e "${RED}Error: Branch '$BASE_BRANCH' does not exist.${NC}"
    exit 1
fi

if ! git -C "$ROOT_DIR" rev-parse --verify "$COMPARE_BRANCH" &>/dev/null; then
    echo -e "${RED}Error: Branch '$COMPARE_BRANCH' does not exist.${NC}"
    exit 1
fi

# Setup
rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"

# Build both branches
for branch in "$BASE_BRANCH" "$COMPARE_BRANCH"; do
    bin_name=$([ "$branch" = "$BASE_BRANCH" ] && echo "base" || echo "compare")
    echo -e "${GREEN}Switching to $branch...${NC}"
    git -C "$ROOT_DIR" checkout "$branch"
    echo -e "${GREEN}Building CLI for $branch...${NC}"
    cargo build --release -p cli --manifest-path "$ROOT_DIR/Cargo.toml"
    cp "$ROOT_DIR/target/release/cli" "$TMP_DIR/cli-$bin_name"
done

# Return to original branch and compile benchmarks
echo -e "${GREEN}Switching back to $CURRENT_BRANCH...${NC}"
git -C "$ROOT_DIR" checkout "$CURRENT_BRANCH"
echo -e "${GREEN}Compiling benchmark programs...${NC}"
make -C "$ROOT_DIR" compile-bench

# Run benchmarks
echo -e "${GREEN}=== Running Benchmarks ===${NC}"
for elf in "$BENCH_ARTIFACTS_DIR"/*.elf; do
    name=$(basename "$elf" .elf)
    echo -e "${YELLOW}--- $name ---${NC}"
    hyperfine -N --warmup 5 --runs 10 \
        -n "$BASE_BRANCH" "$TMP_DIR/cli-base $elf" \
        -n "$COMPARE_BRANCH" "$TMP_DIR/cli-compare $elf" \
        --export-json "$TMP_DIR/$name.json"
done

# Summary
echo -e "${GREEN}=== Summary ===${NC}"
echo -e "Speedup = $BASE_BRANCH / $COMPARE_BRANCH (>1 means $COMPARE_BRANCH is faster)"
echo ""
printf "%-20s %10s\n" "Benchmark" "Speedup"
echo "-------------------------------"
for json in "$TMP_DIR"/*.json; do
    name=$(basename "$json" .json)
    base=$(jq -r '.results[0].mean' "$json")
    compare=$(jq -r '.results[1].mean' "$json")
    if [ -z "$compare" ] || [ "$compare" = "0" ]; then
        speedup="N/A"
    else
        speedup=$(echo "scale=2; $base / $compare" | bc)
    fi
    printf "%-20s %9sx\n" "$name" "$speedup"
done
