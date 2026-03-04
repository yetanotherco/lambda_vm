#!/bin/bash
# Benchmark: Lambda VM vs SP1 v6 — Fibonacci proving time comparison.
#
# Usage: ./bench_vs/run.sh [-n 1000 50000 100000] [--lambda-only | --sp1-only]
#
# Without -n, runs the default series: 1000 10000 100000 300000
# With -n, runs the specified values (space-separated): -n 1000 50000
#
# Prerequisites:
#   - Lambda VM CLI built: cargo build --release -p cli
#   - SP1 toolchain installed: curl -L https://sp1up.succinct.xyz | bash && sp1up
#   - Rust nightly toolchain: rustup toolchain install nightly

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_fib"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# --- Defaults ----------------------------------------------------------------
DEFAULT_SERIES=(1000 10000 100000 300000 600000 1000000)
SERIES=()
RUN_LAMBDA=true
RUN_SP1=true

# --- Parse args --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        -n) shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SERIES+=("$1"); shift
            done ;;
        --lambda-only) RUN_SP1=false; shift ;;
        --sp1-only) RUN_LAMBDA=false; shift ;;
        -h|--help)
            echo "Usage: $0 [-n N1 N2 ...] [--lambda-only | --sp1-only]"
            echo ""
            echo "  -n N1 N2 ...    Fibonacci iteration counts (space-separated)"
            echo "                  Default series: ${DEFAULT_SERIES[*]}"
            echo "  --lambda-only   Only run Lambda VM benchmark"
            echo "  --sp1-only      Only run SP1 benchmark"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ${#SERIES[@]} -eq 0 ]; then
    SERIES=("${DEFAULT_SERIES[@]}")
fi

echo -e "${BOLD}=== Fibonacci Benchmark: Lambda VM vs SP1 v6 ===${NC}"
echo -e "Series: ${YELLOW}${SERIES[*]}${NC}"
echo ""

rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"

# --- Pre-build ---------------------------------------------------------------

CLI="$ROOT_DIR/target/release/cli"
LAMBDA_DIR="$SCRIPT_DIR/lambda/fibonacci"
TARGET_SPEC="$ROOT_DIR/executor/programs/riscv64im-lambda-vm-elf.json"

if $RUN_LAMBDA && [ ! -f "$CLI" ]; then
    echo -e "${YELLOW}[Lambda VM] CLI not found, building...${NC}"
    cargo build --release -p cli --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1
fi

SP1_BIN=""
if $RUN_SP1; then
    SP1_DIR="$SCRIPT_DIR/sp1/fibonacci"
    echo -e "${GREEN}[SP1 v6] Building fibonacci prover...${NC}"
    (cd "$SP1_DIR" && cargo build --release 2>&1 | tail -5)
    SP1_BIN="$SP1_DIR/target/release/fibonacci-script"
    if [ ! -f "$SP1_BIN" ]; then
        echo -e "${RED}[SP1 v6] Build failed — fibonacci-script binary not found${NC}"
        exit 1
    fi
fi

# --- Run one benchmark --------------------------------------------------------

# Arrays to collect results for the summary table
declare -a RESULT_N RESULT_LAMBDA RESULT_SP1

run_one() {
    local N=$1
    echo ""
    echo -e "${BOLD}--- n=${N} ---${NC}"

    local lambda_time=""
    local sp1_time=""
    local sp1_cycles=""

    if $RUN_LAMBDA; then
        echo -e "  ${GREEN}[Lambda VM] Building (n=${N})...${NC}"
        (cd "$LAMBDA_DIR" && BENCH_N="$N" cargo +nightly build --release \
            --target "$TARGET_SPEC" \
            -Z build-std=core -Z build-std-features=compiler-builtins-mem 2>&1 | tail -1)
        LAMBDA_ELF="$LAMBDA_DIR/target/riscv64im-lambda-vm-elf/release/fibonacci-bench"

        echo -e "  ${GREEN}[Lambda VM] Proving...${NC}"
        LAMBDA_OUTPUT=$("$CLI" prove "$LAMBDA_ELF" -o "$TMP_DIR/lambda_proof.bin" --time 2>/dev/null)
        lambda_time=$(echo "$LAMBDA_OUTPUT" | grep -o 'Proving time: [0-9.]*s' | grep -o '[0-9.]*')
        echo -e "  Lambda VM: ${BOLD}${lambda_time}s${NC}"
    fi

    if $RUN_SP1; then
        echo -e "  ${GREEN}[SP1 v6] Proving...${NC}"
        SP1_OUTPUT=$("$SP1_BIN" "$N" 2>/dev/null)
        sp1_time=$(echo "$SP1_OUTPUT" | grep -o 'Proving time: [0-9.]*s' | grep -o '[0-9.]*')
        sp1_cycles=$(echo "$SP1_OUTPUT" | grep -o 'Cycles: [0-9]*' | grep -o '[0-9]*')
        echo -e "  SP1 v6:    ${BOLD}${sp1_time}s${NC} (${sp1_cycles} cycles)"
    fi

    RESULT_N+=("$N")
    RESULT_LAMBDA+=("${lambda_time:-n/a}")
    RESULT_SP1+=("${sp1_time:-n/a}")
}

# --- Run series ---------------------------------------------------------------

for N in "${SERIES[@]}"; do
    run_one "$N"
done

# --- Summary table ------------------------------------------------------------

echo ""
echo -e "${BOLD}=== Summary ===${NC}"
echo -e "Program: Fibonacci (u64 wrapping)"
echo ""

# Header
if $RUN_LAMBDA && $RUN_SP1; then
    printf "  %-10s  %12s  %12s  %8s\n" "n" "Lambda VM" "SP1 v6" "Ratio"
    printf "  %-10s  %12s  %12s  %8s\n" "---" "---------" "------" "-----"
elif $RUN_LAMBDA; then
    printf "  %-10s  %12s\n" "n" "Lambda VM"
    printf "  %-10s  %12s\n" "---" "---------"
else
    printf "  %-10s  %12s\n" "n" "SP1 v6"
    printf "  %-10s  %12s\n" "---" "------"
fi

for i in "${!RESULT_N[@]}"; do
    n="${RESULT_N[$i]}"
    lt="${RESULT_LAMBDA[$i]}"
    st="${RESULT_SP1[$i]}"

    if $RUN_LAMBDA && $RUN_SP1; then
        if [ "$lt" != "n/a" ] && [ "$st" != "n/a" ]; then
            RATIO=$(LC_NUMERIC=C awk "BEGIN {printf \"%.1fx\", $lt / $st}")
            if (( $(LC_NUMERIC=C awk "BEGIN {print ($lt > $st)}") )); then
                RATIO="${RED}${RATIO}${NC}"
            else
                RATIO="${GREEN}${RATIO}${NC}"
            fi
            printf "  %-10s  %11ss  %11ss  " "$n" "$lt" "$st"
            echo -e "$RATIO"
        else
            printf "  %-10s  %12s  %12s  %8s\n" "$n" "${lt}s" "${st}s" "-"
        fi
    elif $RUN_LAMBDA; then
        printf "  %-10s  %11ss\n" "$n" "$lt"
    else
        printf "  %-10s  %11ss\n" "$n" "$st"
    fi
done

echo ""
echo -e "Green ratio = Lambda VM faster, Red = SP1 faster"
echo "Raw data in $TMP_DIR/"
