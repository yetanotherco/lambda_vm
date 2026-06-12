#!/usr/bin/env bash
# Benchmark: Lambda VM proving ethrex blocks.
#
# Proves each block in BLOCKS (an ethrex guest ELF + a serialized ProgramInput
# private input) and reports single-shot end-to-end proving time and cycle count.
# Add a block by appending a "label|input_basename" entry to BLOCKS — the input
# file must live in executor/tests/.
#
# Usage: ./bench_vs/run_ethrex.sh [--report-dir DIR] [--rebuild-elf] [--no-color]
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
REBUILD_ELF=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# Blocks to benchmark: "label|input_basename" (input lives in executor/tests/).
BLOCKS=(
    "ethrex empty block|ethrex_empty_block.bin"
    "ethrex 1 tx|ethrex_simple_tx.bin"
    "ethrex 10 txs|ethrex_10_transfers.bin"
)

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
        --rebuild-elf)
            REBUILD_ELF=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--report-dir DIR] [--rebuild-elf] [--no-color]"
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

# slugify a label into a metrics-key/filename-safe token
slugify() {
    printf "%s" "$1" | tr '[:upper:] ' '[:lower:]_' | tr -cd '[:alnum:]_'
}

# --- Pre-build --------------------------------------------------------------

CLI="$ROOT_DIR/target/release/cli"
ETHREX_ELF="$ROOT_DIR/executor/program_artifacts/rust/ethrex.elf"
echo -e "${BOLD}=== Ethrex Block Benchmarks: Lambda VM ===${NC}"
echo ""

echo -e "${GREEN}[Lambda VM] Building CLI...${NC}"
cargo build --release -p cli --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -5

if $REBUILD_ELF; then
    echo -e "${GREEN}[Lambda VM] Rebuilding ethrex guest ELF...${NC}"
    make -B -C "$ROOT_DIR" executor/program_artifacts/rust/ethrex.elf
elif [ -f "$ETHREX_ELF" ]; then
    echo -e "${YELLOW}[Lambda VM] Using pre-existing ethrex.elf at $ETHREX_ELF${NC}"
else
    echo -e "${GREEN}[Lambda VM] Building ethrex guest ELF...${NC}"
    make -C "$ROOT_DIR" executor/program_artifacts/rust/ethrex.elf 2>&1 | tail -5
fi

if [ ! -f "$ETHREX_ELF" ]; then
    echo -e "${RED}[Lambda VM] Build failed — ethrex.elf not found at $ETHREX_ELF${NC}"
    exit 1
fi

# --- Run benchmarks ---------------------------------------------------------

# Parallel arrays of results, indexed alongside BLOCKS.
labels=()
times=()
cycles_arr=()

for entry in "${BLOCKS[@]}"; do
    label=${entry%%|*}
    input_basename=${entry##*|}
    input_path="$ROOT_DIR/executor/tests/$input_basename"
    slug=$(slugify "$label")

    echo ""
    echo -e "${BOLD}--- Proving ${label} ---${NC}"

    if [ ! -f "$input_path" ]; then
        echo -e "${RED}Input file not found: $input_path${NC}"
        exit 1
    fi

    proof_file="$TMP_DIR/$slug.proof"
    stderr_file="$TMP_DIR/$slug.stderr"

    echo -e "  ${GREEN}[Lambda VM] Proving...${NC}"
    if ! lambda_output=$("$CLI" prove "$ETHREX_ELF" \
            -o "$proof_file" \
            --private-input "$input_path" \
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
        printf "%s\n" "$lambda_output" > "$REPORT_DIR/raw/$slug.stdout"
        cp "$stderr_file" "$REPORT_DIR/raw/$slug.stderr"
    fi

    labels+=("$label")
    times+=("$lambda_time")
    cycles_arr+=("$lambda_cycles")
done

# --- Summary table ----------------------------------------------------------

echo ""
echo -e "${BOLD}=== Summary ===${NC}"
echo ""

printf "  %-22s  %14s  %14s\n" "Program" "Lambda (s)" "Lambda cycles"
printf "  %-22s  %14s  %14s\n" "----------------------" "----------" "-------------"
for i in "${!labels[@]}"; do
    printf "  %-22s  %13ss  %14s\n" "${labels[$i]}" "${times[$i]}" "${cycles_arr[$i]}"
done

echo ""
echo -e "Timing window covers single-shot end-to-end proving; excludes verification."
echo "Raw data in $TMP_DIR/"

# --- Machine-readable report ------------------------------------------------

if [ -n "$REPORT_DIR" ]; then
    {
        echo "timing_window=single_shot_end_to_end_prove_no_verify"
        for i in "${!labels[@]}"; do
            slug=$(slugify "${labels[$i]}")
            echo "${slug}_time_s=${times[$i]}"
            echo "${slug}_cycles=${cycles_arr[$i]}"
        done
    } > "$REPORT_DIR/ethrex_metrics.txt"

    {
        echo "# Ethrex Block Benchmarks — Lambda VM"
        echo
        echo "Timing window: \`single-shot end-to-end prove\` (excludes verification)."
        echo
        echo "| Program | Lambda VM (s) | Lambda cycles |"
        echo "|---------|--------------:|--------------:|"
        for i in "${!labels[@]}"; do
            printf "| %s | %s | %s |\n" "${labels[$i]}" "${times[$i]}" "${cycles_arr[$i]}"
        done
    } > "$REPORT_DIR/ethrex_summary.md"
fi
