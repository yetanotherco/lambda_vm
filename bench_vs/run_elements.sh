#!/bin/bash
# Compare main-trace field elements: Lambda VM vs SP1 v6 vs SP1 v5 — Fibonacci.
#
# "Field elements to prove" = sum over all AIR tables of (padded_rows × num_columns).
# More elements = larger trace = more work for the prover.
#
# Usage: ./bench_vs/run_elements.sh [-n 100000 200000 400000]
#
# Prerequisites:
#   - Lambda VM CLI build dependencies available
#   - SP1 toolchain installed (sp1up)
#   - Rust stable installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_elements_fib"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Default series ---------------------------------------------------------
DEFAULT_SERIES=(100000 200000 400000)
SERIES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                SERIES+=("$1")
                shift
            done
            ;;
        -h|--help)
            echo "Usage: $0 [-n N1 N2 ...]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ ${#SERIES[@]} -eq 0 ]; then
    SERIES=("${DEFAULT_SERIES[@]}")
fi

mkdir -p "$TMP_DIR"
rm -rf "$TMP_DIR"/*

# --- Paths ------------------------------------------------------------------
CLI="$ROOT_DIR/target/release/cli"
LAMBDA_DIR="$SCRIPT_DIR/lambda/fibonacci"
TARGET_SPEC="$ROOT_DIR/executor/programs/riscv64im-lambda-vm-elf.json"
LAMBDA_ELF="$LAMBDA_DIR/target/riscv64im-lambda-vm-elf/release/fibonacci-bench"

SP1_V6_DIR="$SCRIPT_DIR/sp1/fibonacci"
SP1_V6_BIN="$SP1_V6_DIR/target/release/fibonacci-script"

SP1_V5_DIR="$SCRIPT_DIR/sp1_v5/fibonacci"
SP1_V5_BIN="$SP1_V5_DIR/target/release/fibonacci-script-v5"

write_u64_le() {
    local value=$1
    local output_path=$2

    python3 - "$value" "$output_path" <<'PY'
import struct
import sys

value = int(sys.argv[1])
path = sys.argv[2]

with open(path, "wb") as fh:
    fh.write(struct.pack("<Q", value))
PY
}

extract_elements() {
    sed -nE '/Elements: [0-9]+/ {
        s/.*Elements: ([0-9]+).*/\1/
        p
        q
    }'
}

# Format Lambda/SP1 ratio as +3.6x (Lambda bigger) or -0.21x (Lambda smaller).
format_ratio() {
    local num=$1 den=$2
    if [ "$num" = "n/a" ] || [ "$den" = "n/a" ]; then
        echo "n/a"
        return
    fi
    python3 -c "
r = $num / $den
print(f'+{r:.1f}x' if r >= 1 else f'-{r:.2f}x')
"
}

# Format an integer with thousands separators.
fmt_num() {
    [ "$1" = "n/a" ] && { echo "n/a"; return; }
    python3 -c "print(f'{$1:,}')"
}

# --- Build ------------------------------------------------------------------
echo -e "${BOLD}=== Building all provers ===${NC}"

echo -e "${GREEN}[Lambda VM] Building CLI...${NC}"
cargo build --release -p cli --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -5

echo -e "${GREEN}[Lambda VM] Building fibonacci ELF...${NC}"
(
    cd "$LAMBDA_DIR" && \
    cargo +nightly-2026-02-01 build --release \
        --target "$TARGET_SPEC" \
        -Z build-std=core \
        -Z build-std-features=compiler-builtins-mem \
        -Z json-target-spec 2>&1 | tail -5
)
if [ ! -f "$LAMBDA_ELF" ]; then
    echo -e "${RED}[Lambda VM] Build failed — fibonacci-bench ELF not found${NC}"
    exit 1
fi

echo -e "${GREEN}[SP1 v6] Building fibonacci prover...${NC}"
(cd "$SP1_V6_DIR" && cargo build --release 2>&1 | tail -5)
if [ ! -f "$SP1_V6_BIN" ]; then
    echo -e "${RED}[SP1 v6] Build failed — fibonacci-script binary not found${NC}"
    exit 1
fi

echo -e "${GREEN}[SP1 v5] Building guest ELF (riscv32)...${NC}"
SP1_V5_GUEST_ELF="$SP1_V5_DIR/target/riscv32im-succinct-zkvm-elf/release/fibonacci-program-v5"
# Replicate sp1-build 5.2.4 flags, omitting the two broken llvm-args
# (-misched-prera/postra-direction) that are invalid in the updated LLVM.
# CARGO_ENCODED_RUSTFLAGS uses ASCII field-separator (0x1f) between entries.
_SEP=$'\x1f'
_V5_FLAGS="-C${_SEP}passes=lower-atomic${_SEP}-C${_SEP}link-arg=-Ttext=0x00201000${_SEP}-C${_SEP}link-arg=--image-base=0x00200800${_SEP}-C${_SEP}panic=abort${_SEP}--cfg${_SEP}getrandom_backend=\"custom\""
(
    cd "$SP1_V5_DIR/program" && \
    CARGO_ENCODED_RUSTFLAGS="$_V5_FLAGS" \
    CFLAGS_riscv32im_succinct_zkvm_elf="-D__ILP32__" \
    RUSTC_BOOTSTRAP=1 \
    cargo +succinct build --release \
        --target riscv32im-succinct-zkvm-elf \
        -Ztrim-paths 2>&1 | tail -5
)
if [ ! -f "$SP1_V5_GUEST_ELF" ]; then
    echo -e "${RED}[SP1 v5] Build failed — fibonacci-program-v5 ELF not found${NC}"
    exit 1
fi

echo -e "${GREEN}[SP1 v5] Building fibonacci prover...${NC}"
(cd "$SP1_V5_DIR/script" && cargo build --release 2>&1 | tail -5)
if [ ! -f "$SP1_V5_BIN" ]; then
    echo -e "${RED}[SP1 v5] Build failed — fibonacci-script-v5 binary not found${NC}"
    exit 1
fi

echo ""
echo -e "${BOLD}=== Counting field elements: Lambda VM vs SP1 v6 vs SP1 v5 ===${NC}"
echo -e "Series: ${YELLOW}${SERIES[*]}${NC}"
echo ""

# --- Results arrays ---------------------------------------------------------
RESULT_N=()
RESULT_LAMBDA=()
RESULT_SP1_V6=()
RESULT_SP1_V5=()

for n in "${SERIES[@]}"; do
    echo -e "${BOLD}--- n = $n iterations ---${NC}"
    input_file="$TMP_DIR/input_${n}.bin"
    write_u64_le "$n" "$input_file"

    # Lambda VM — count-elements subcommand (no proof)
    echo -e "  ${GREEN}[Lambda VM] Counting elements...${NC}"
    lambda_out=$("$CLI" count-elements "$LAMBDA_ELF" --private-input "$input_file" 2>/dev/null)
    lambda_elems=$(printf "%s\n" "$lambda_out" | extract_elements)
    if [ -z "$lambda_elems" ]; then
        echo -e "  ${RED}[Lambda VM] FAILED to parse element count${NC}"
        echo "$lambda_out"
        lambda_elems="n/a"
    else
        echo -e "  Lambda VM: ${BOLD}${lambda_elems}${NC} elements"
    fi

    # SP1 v6 — prove and extract element count
    echo -e "  ${GREEN}[SP1 v6] Proving to extract element count...${NC}"
    sp1v6_out_file="$TMP_DIR/sp1v6_${n}.out"
    if "$SP1_V6_BIN" "$n" > "$sp1v6_out_file" 2>&1; then
        sp1v6_elems=$(extract_elements < "$sp1v6_out_file")
        if [ -z "$sp1v6_elems" ]; then
            echo -e "  ${RED}[SP1 v6] FAILED to parse element count${NC}"
            sp1v6_elems="n/a"
        else
            echo -e "  SP1 v6:    ${BOLD}${sp1v6_elems}${NC} elements"
        fi
    else
        echo -e "  ${RED}[SP1 v6] FAILED${NC}"
        cat "$sp1v6_out_file"
        sp1v6_elems="n/a"
    fi

    # SP1 v5 — prove and extract element count
    echo -e "  ${GREEN}[SP1 v5] Proving to extract element count...${NC}"
    sp1v5_out_file="$TMP_DIR/sp1v5_${n}.out"
    if "$SP1_V5_BIN" "$SP1_V5_GUEST_ELF" "$n" > "$sp1v5_out_file" 2>&1; then
        sp1v5_elems=$(extract_elements < "$sp1v5_out_file")
        if [ -z "$sp1v5_elems" ]; then
            echo -e "  ${RED}[SP1 v5] FAILED to parse element count${NC}"
            sp1v5_elems="n/a"
        else
            echo -e "  SP1 v5:    ${BOLD}${sp1v5_elems}${NC} elements"
        fi
    else
        echo -e "  ${RED}[SP1 v5] FAILED${NC}"
        cat "$sp1v5_out_file"
        sp1v5_elems="n/a"
    fi

    RESULT_N+=("$n")
    RESULT_LAMBDA+=("$lambda_elems")
    RESULT_SP1_V6+=("$sp1v6_elems")
    RESULT_SP1_V5+=("$sp1v5_elems")
    echo ""
done

# --- Summary table ----------------------------------------------------------
echo -e "${BOLD}=== Summary: Main-Trace Field Elements (rows × cols, all tables) ===${NC}"
echo -e "  Program : Fibonacci (u64 wrapping)"
echo -e "  Metric  : sum of padded_rows × num_columns across all AIR tables"
echo -e "  Fields  : Lambda VM = Goldilocks 64-bit | SP1 = BabyBear 32-bit"
echo ""

W_N=13; W_E=18; W_R=11
printf "  %-${W_N}s  %${W_E}s  %${W_E}s  %${W_E}s  %${W_R}s  %${W_R}s\n" \
    "Fibonacci n" "Lambda VM" "SP1 v6" "SP1 v5" "vs SP1 v6" "vs SP1 v5"
printf "  %-${W_N}s  %${W_E}s  %${W_E}s  %${W_E}s  %${W_R}s  %${W_R}s\n" \
    "-------------" "------------------" "------------------" "------------------" "-----------" "-----------"

for i in "${!RESULT_N[@]}"; do
    lam="${RESULT_LAMBDA[$i]}"
    v6="${RESULT_SP1_V6[$i]}"
    v5="${RESULT_SP1_V5[$i]}"
    printf "  %-${W_N}s  %${W_E}s  %${W_E}s  %${W_E}s  %${W_R}s  %${W_R}s\n" \
        "$(fmt_num "${RESULT_N[$i]}")" \
        "$(fmt_num "$lam")" \
        "$(fmt_num "$v6")" \
        "$(fmt_num "$v5")" \
        "$(format_ratio "$lam" "$v6")" \
        "$(format_ratio "$lam" "$v5")"
done
echo ""
echo -e "  +Nx = Lambda has N× more elements than SP1"
echo -e "  -Nx = Lambda has N× fewer elements than SP1"
echo ""
echo "Raw outputs in $TMP_DIR/"
