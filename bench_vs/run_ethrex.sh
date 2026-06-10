#!/usr/bin/env bash
# Benchmark: Lambda VM proving an empty ethrex block.
#
# Usage: ./bench_vs/run_ethrex.sh [--report-dir DIR] [--no-color]
#
# Prerequisites:
#   - Lambda VM CLI build dependencies available
#   - Sysroot present at /opt/lambda-vm-sysroot (run `make prepare-sysroot` first)
#   - Rust stable + nightly-2026-02-01 installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_ethrex"
REPORT_DIR=""
NO_COLOR=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# --- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --report-dir)
            if [[ $# -lt 2 ]]; then echo "--report-dir requires an argument"; exit 1; fi
            REPORT_DIR=$2
            shift 2
            ;;
        --no-color)
            NO_COLOR=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--report-dir DIR] [--no-color]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if $NO_COLOR; then
    RED=''
    GREEN=''
    YELLOW=''
    BOLD=''
    NC=''
fi

mkdir -p "$TMP_DIR"
rm -rf "${TMP_DIR:?}"/*

if [ -n "$REPORT_DIR" ]; then
    mkdir -p "$REPORT_DIR/raw"
fi

extract_proving_time() {
    sed -nE '/Proving time: [0-9.]+s/ {
        s/.*Proving time: ([0-9.]+)s.*/\1/
        p
        q
    }'
}

extract_cycles() {
    sed -nE '/Cycles: [0-9]+/ {
        s/.*Cycles: ([0-9]+).*/\1/
        p
        q
    }'
}

# --- Pre-build --------------------------------------------------------------

CLI="$ROOT_DIR/target/release/cli"
ETHREX_ELF="$ROOT_DIR/executor/program_artifacts/rust/ethrex.elf"
ETHREX_INPUT="$ROOT_DIR/executor/tests/ethrex_empty_block.bin"
echo -e "${BOLD}=== Ethrex Empty Block Benchmark: Lambda VM ===${NC}"
echo ""

echo -e "${GREEN}[Lambda VM] Building CLI...${NC}"
cargo build --release -p cli --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -5

if [ -f "$ETHREX_ELF" ]; then
    echo -e "${YELLOW}[Lambda VM] Using pre-existing ethrex.elf at $ETHREX_ELF${NC}"
else
    echo -e "${GREEN}[Lambda VM] Building ethrex guest ELF...${NC}"
    make -C "$ROOT_DIR" executor/program_artifacts/rust/ethrex.elf 2>&1 | tail -5
fi

if [ ! -f "$ETHREX_ELF" ]; then
    echo -e "${RED}[Lambda VM] Build failed — ethrex.elf not found at $ETHREX_ELF${NC}"
    exit 1
fi

if [ ! -f "$ETHREX_INPUT" ]; then
    echo -e "${RED}Input file not found: $ETHREX_INPUT${NC}"
    exit 1
fi

# --- Run benchmark ---------------------------------------------------
echo ""
echo -e "${BOLD}--- Proving empty ethrex block ---${NC}"

proof_file="$TMP_DIR/ethrex_empty_block.proof"
stderr_file="$TMP_DIR/ethrex_empty_block.stderr"

echo -e "  ${GREEN}[Lambda VM] Proving...${NC}"
if ! lambda_output=$("$CLI" prove "$ETHREX_ELF" \
        -o "$proof_file" \
        --private-input "$ETHREX_INPUT" \
        --time --cycles 2>"$stderr_file"); then
    echo -e "  ${RED}[Lambda VM] FAILED:${NC}"
    cat "$stderr_file"
    exit 1
fi
rm -f "$proof_file"

lambda_time=$(printf "%s\n" "$lambda_output" | extract_proving_time)
lambda_cycles=$(printf "%s\n" "$lambda_output" | extract_cycles)

if [ -z "$lambda_time" ]; then
    echo -e "  ${RED}[Lambda VM] FAILED: could not parse proving time${NC}"
    printf "%s\n" "$lambda_output"
    exit 1
fi
if [ -z "$lambda_cycles" ]; then
    lambda_cycles="n/a"
fi

if [ "$lambda_cycles" != "n/a" ]; then
    echo -e "  Lambda VM: ${BOLD}${lambda_time}s${NC} (${lambda_cycles} cycles)"
else
    echo -e "  Lambda VM: ${BOLD}${lambda_time}s${NC}"
fi

if [ -n "$REPORT_DIR" ]; then
    printf "%s\n" "$lambda_output" > "$REPORT_DIR/raw/ethrex_empty_block.stdout"
    cp "$stderr_file" "$REPORT_DIR/raw/ethrex_empty_block.stderr"
fi

# --- Summary table ----------------------------------------------------------

echo ""
echo -e "${BOLD}=== Summary ===${NC}"
echo -e "Program: ethrex empty block"
echo ""

printf "  %-22s  %14s  %14s\n" "Program" "Lambda (s)" "Lambda cycles"
printf "  %-22s  %14s  %14s\n" "----------------------" "----------" "-------------"
printf "  %-22s  %13ss  %14s\n" "ethrex empty block" "$lambda_time" "$lambda_cycles"

echo ""
echo -e "Timing window covers single-shot end-to-end proving; excludes verification."
echo "Raw data in $TMP_DIR/"

# --- Machine-readable report ------------------------------------------------

if [ -n "$REPORT_DIR" ]; then
    {
        echo "program=ethrex_empty_block"
        echo "input_file=$ETHREX_INPUT"
        echo "timing_window=single_shot_end_to_end_prove_no_verify"
        echo "ethrex_empty_block_time_s=$lambda_time"
        echo "ethrex_empty_block_cycles=$lambda_cycles"
    } > "$REPORT_DIR/ethrex_metrics.txt"

    {
        echo "# Ethrex Empty Block — Lambda VM"
        echo
        echo "Timing window: \`single-shot end-to-end prove\` (excludes verification)."
        echo
        echo "| Program | Lambda VM (s) | Lambda cycles |"
        echo "|---------|--------------:|--------------:|"
        printf "| ethrex empty block | %s | %s |\n" "$lambda_time" "$lambda_cycles"
    } > "$REPORT_DIR/ethrex_summary.md"
fi
