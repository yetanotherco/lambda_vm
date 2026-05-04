#!/usr/bin/env bash
# Compare main-trace field elements: Lambda VM vs SP1 v6 vs SP1 v5 vs ZisK — Fibonacci.
#
# "Field elements to prove" = sum over all AIR tables of (padded_rows × num_columns).
# More elements = larger trace = more work for the prover.
#
# Usage: ./bench_vs/run_elements.sh [-n 100000 200000 400000]
#                                   [--lambda-only | --sp1-only | --zisk-only]
#                                   [--report-dir DIR] [--no-color]
#
# Prerequisites:
#   - Lambda VM CLI build dependencies available
#   - SP1 toolchain installed (sp1up)
#   - ZisK toolchain installed (ziskup) — provides ~/.zisk/provingKey/ and the
#     +zisk Rust toolchain used to cross-compile the guest. macOS uses emulator
#     mode automatically; only Linux x86_64 supports the asm backend.
#   - Rust stable installed
#   - python3 installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_elements_fib"
REPORT_DIR=""
NO_COLOR=false

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# --- Default series ---------------------------------------------------------
DEFAULT_SERIES=(100000 200000 400000)
SERIES=()
RUN_LAMBDA=true
RUN_SP1=true
RUN_ZISK=true

while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                SERIES+=("$1")
                shift
            done
            ;;
        --lambda-only)
            RUN_SP1=false
            RUN_ZISK=false
            shift
            ;;
        --sp1-only)
            RUN_LAMBDA=false
            RUN_ZISK=false
            shift
            ;;
        --zisk-only)
            RUN_LAMBDA=false
            RUN_SP1=false
            shift
            ;;
        --no-sp1)
            RUN_SP1=false
            shift
            ;;
        --no-zisk)
            RUN_ZISK=false
            shift
            ;;
        --no-lambda)
            RUN_LAMBDA=false
            shift
            ;;
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
            echo "Usage: $0 [-n N1 N2 ...] [--lambda-only | --sp1-only | --zisk-only]"
            echo "          [--no-lambda | --no-sp1 | --no-zisk] [--report-dir DIR] [--no-color]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if ! $RUN_LAMBDA && ! $RUN_SP1 && ! $RUN_ZISK; then
    echo "At least one prover must be enabled"
    exit 1
fi

if [ ${#SERIES[@]} -eq 0 ]; then
    SERIES=("${DEFAULT_SERIES[@]}")
fi

for value in "${SERIES[@]}"; do
    if ! [[ "$value" =~ ^[0-9]+$ ]] || [ "$value" -le 0 ]; then
        echo "Invalid series value: $value (must be a positive integer)"
        exit 1
    fi
done

if $NO_COLOR; then
    BOLD=''
    GREEN=''
    YELLOW=''
    RED=''
    NC=''
fi

mkdir -p "$TMP_DIR"
rm -rf "$TMP_DIR"/*

if [ -n "$REPORT_DIR" ]; then
    mkdir -p "$REPORT_DIR/raw"
fi

# --- Paths ------------------------------------------------------------------
CLI="$ROOT_DIR/target/release/cli"
LAMBDA_DIR="$SCRIPT_DIR/lambda/fibonacci"
TARGET_SPEC="$ROOT_DIR/executor/programs/riscv64im-lambda-vm-elf.json"
LAMBDA_ELF="$LAMBDA_DIR/target/riscv64im-lambda-vm-elf/release/fibonacci-bench"

SP1_V6_DIR="$SCRIPT_DIR/sp1/fibonacci"
SP1_V6_BIN="$SP1_V6_DIR/target/release/fibonacci-script"

SP1_V5_DIR="$SCRIPT_DIR/sp1_v5/fibonacci"
SP1_V5_BIN="$SP1_V5_DIR/target/release/fibonacci-script-v5"
SP1_V5_GUEST_ELF="$SP1_V5_DIR/target/riscv32im-succinct-zkvm-elf/release/fibonacci-program-v5"

ZISK_DIR="$SCRIPT_DIR/zisk/fibonacci"
ZISK_HOST_BIN="$ZISK_DIR/target/release/fibonacci-zisk-host"

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

extract_aux_elements() {
    sed -nE '/Aux elements \(EF-cols\): [0-9]+/ {
        s/.*Aux elements \(EF-cols\): ([0-9]+).*/\1/
        p
        q
    }'
}

# Format Lambda/SP1 ratio as +3.6x (Lambda bigger) or -3.6x (Lambda smaller).
# Always shows how many times larger the bigger side is, with sign indicating direction.
format_ratio() {
    local num=$1 den=$2
    if [ "$num" = "n/a" ] || [ "$den" = "n/a" ]; then
        echo "n/a"
        return
    fi
    python3 -c "
den = $den
if den == 0: print('n/a')
else:
    r = $num / den
    print(f'+{r:.1f}x' if r >= 1 else f'-{1/r:.1f}x')
"
}

# Format an integer with thousands separators.
fmt_num() {
    [ "$1" = "n/a" ] && { echo "n/a"; return; }
    python3 -c "print(f'{$1:,}')"
}

# Verify everything ZisK needs before triggering the (slow) C++ build of zisk-sdk.
# Exits with a copy-pasteable install command if anything is missing.
check_zisk_prereqs() {
    local missing=()
    local fix_lines=()

    # 1. ziskup-installed toolchain (cargo-zisk binary + +zisk Rust toolchain)
    if ! command -v cargo-zisk >/dev/null 2>&1 && [ ! -x "$HOME/.zisk/bin/cargo-zisk" ]; then
        missing+=("ziskup (cargo-zisk binary + +zisk Rust toolchain)")
        fix_lines+=("curl -L https://raw.githubusercontent.com/0xPolygonHermez/zisk/main/ziskup/install.sh | bash")
        fix_lines+=("ziskup    # choose option 1 (default) to install the proving key")
    fi

    # 2. Proving key (populated by ziskup option 1)
    if [ ! -d "$HOME/.zisk/provingKey" ] || [ -z "$(ls -A "$HOME/.zisk/provingKey" 2>/dev/null)" ]; then
        missing+=("proving key at ~/.zisk/provingKey/ (run \`ziskup\` and choose option 1)")
    fi

    # 3. System C++ headers/libs needed by proofman-starks-lib-c
    case "$(uname -s)" in
        Darwin)
            if ! command -v brew >/dev/null 2>&1; then
                missing+=("Homebrew (https://brew.sh)")
            else
                # llvm@15 is required because the prebuilt cargo-zisk binary for
                # darwin-arm64 is dynamically linked against /opt/homebrew/opt/llvm@15/lib/libunwind.1.dylib.
                # Without it the binary aborts with a dyld error on the first invocation.
                local brew_missing=()
                for pkg in libomp nlohmann-json libsodium llvm@15; do
                    if ! brew list --formula "$pkg" >/dev/null 2>&1; then
                        brew_missing+=("$pkg")
                    fi
                done
                if [ "${#brew_missing[@]}" -gt 0 ]; then
                    missing+=("Homebrew formulas: ${brew_missing[*]}")
                    fix_lines+=("brew install ${brew_missing[*]}")
                fi
            fi
            ;;
        Linux)
            local linux_missing=()
            # Debian installs omp.h under /usr/lib/llvm-NN/lib/clang/NN.N.N/include or /usr/lib/gcc/.../include;
            # Ubuntu/older Debian put a symlink at /usr/include/omp.h. Use find to cover all variants.
            find /usr/lib /usr/include -maxdepth 8 -name omp.h -print -quit 2>/dev/null | grep -q . || linux_missing+=("libomp-dev")
            [ -f /usr/include/nlohmann/json.hpp ] || linux_missing+=("nlohmann-json3-dev")
            [ -f /usr/include/sodium.h ] || linux_missing+=("libsodium-dev")
            [ -f /usr/include/gmpxx.h ] || linux_missing+=("libgmp-dev")
            command -v cmake >/dev/null 2>&1 || linux_missing+=("cmake")
            command -v pkg-config >/dev/null 2>&1 || linux_missing+=("pkg-config")
            command -v g++ >/dev/null 2>&1 || linux_missing+=("build-essential")
            command -v clang >/dev/null 2>&1 || linux_missing+=("clang")
            command -v nasm >/dev/null 2>&1 || linux_missing+=("nasm")
            # mpicc ships with libopenmpi-dev and is more reliable than `ldconfig -p` —
            # Debian puts OpenMPI libs under /usr/lib/x86_64-linux-gnu/openmpi/lib/, often
            # outside the ldconfig cache, so the cache check false-flags the package.
            command -v mpicc >/dev/null 2>&1 || linux_missing+=("libopenmpi-dev" "openmpi-bin")
            # mpi-sys's bindgen needs libclang at build time; without it it falls back to
            # opaque struct fields and the build fails with "no field MPI_SOURCE on type
            # ompi_status_public_t". Auto-export LIBCLANG_PATH on first match so users don't
            # have to set it manually before `cargo build`.
            local libclang_dir=""
            for d in /usr/lib/llvm-14/lib /usr/lib/llvm-15/lib /usr/lib/llvm-16/lib \
                     /usr/lib/llvm-17/lib /usr/lib/llvm-18/lib /usr/lib/x86_64-linux-gnu; do
                if compgen -G "$d/libclang.so*" >/dev/null 2>&1 \
                    || compgen -G "$d/libclang-*.so*" >/dev/null 2>&1; then
                    libclang_dir="$d"
                    break
                fi
            done
            if [ -n "$libclang_dir" ]; then
                [ -z "${LIBCLANG_PATH:-}" ] && export LIBCLANG_PATH="$libclang_dir"
            else
                linux_missing+=("libclang-14-dev")
            fi
            if [ "${#linux_missing[@]}" -gt 0 ]; then
                missing+=("apt packages: ${linux_missing[*]}")
                fix_lines+=("sudo apt install -y ${linux_missing[*]}")
            fi
            ;;
    esac

    if [ "${#missing[@]}" -gt 0 ]; then
        echo -e "${RED}[ZisK] Missing prerequisites:${NC}"
        for item in "${missing[@]}"; do
            echo -e "  ${RED}- $item${NC}"
        done
        if [ "${#fix_lines[@]}" -gt 0 ]; then
            echo ""
            echo -e "${YELLOW}Install with:${NC}"
            for line in "${fix_lines[@]}"; do
                echo -e "  ${YELLOW}$line${NC}"
            done
        fi
        echo ""
        echo -e "${YELLOW}See bench_vs/README.md for the full setup, or skip ZisK with --lambda-only / --sp1-only.${NC}"
        exit 1
    fi
}

# --- Build ------------------------------------------------------------------
echo -e "${BOLD}=== Building all provers ===${NC}"

if $RUN_LAMBDA; then
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
fi

if $RUN_SP1; then
    echo -e "${GREEN}[SP1 v6] Building fibonacci prover...${NC}"
    (cd "$SP1_V6_DIR" && cargo build --release 2>&1 | tail -5)
    if [ ! -f "$SP1_V6_BIN" ]; then
        echo -e "${RED}[SP1 v6] Build failed — fibonacci-script binary not found${NC}"
        exit 1
    fi

    echo -e "${GREEN}[SP1 v5] Building guest ELF (riscv32)...${NC}"
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
fi

if $RUN_ZISK; then
    check_zisk_prereqs
    echo -e "${GREEN}[ZisK] Building host (first run compiles the C++ STARK lib, ~10 min)...${NC}"
    (cd "$ZISK_DIR/host" && cargo build --release 2>&1 | tail -5)
    if [ ! -f "$ZISK_HOST_BIN" ]; then
        echo -e "${RED}[ZisK] Build failed — fibonacci-zisk-host binary not found${NC}"
        echo -e "${YELLOW}See bench_vs/README.md for setup; \`./bench_vs/run_elements.sh --zisk-only\` re-runs the prereq check.${NC}"
        exit 1
    fi
fi

echo ""
echo -e "${BOLD}=== Counting field elements: Lambda VM vs SP1 v6 vs SP1 v5 vs ZisK ===${NC}"
echo -e "Series: ${YELLOW}${SERIES[*]}${NC}"
echo ""

# --- Results arrays ---------------------------------------------------------
RESULT_N=()
RESULT_LAMBDA=()
RESULT_LAMBDA_AUX=()
RESULT_SP1_V6=()
RESULT_SP1_V5=()
RESULT_SP1_V5_AUX=()
RESULT_ZISK=()

for n in "${SERIES[@]}"; do
    echo -e "${BOLD}--- n = $n iterations ---${NC}"
    input_file="$TMP_DIR/input_${n}.bin"
    write_u64_le "$n" "$input_file"

    lambda_elems="n/a"
    lambda_aux="n/a"
    sp1v6_elems="n/a"
    sp1v5_elems="n/a"
    sp1v5_aux="n/a"
    zisk_elems="n/a"

    if $RUN_LAMBDA; then
        echo -e "  ${GREEN}[Lambda VM] Counting elements...${NC}"
        lambda_out_file="$TMP_DIR/lambda_${n}.out"
        if "$CLI" count-elements "$LAMBDA_ELF" --private-input "$input_file" > "$lambda_out_file" 2>&1; then
            lambda_elems=$(extract_elements < "$lambda_out_file")
            lambda_aux=$(extract_aux_elements < "$lambda_out_file")
            if [ -z "$lambda_elems" ]; then
                echo -e "  ${RED}[Lambda VM] FAILED to parse element count${NC}"
                exit 1
            fi
            echo -e "  Lambda VM: ${BOLD}${lambda_elems}${NC} main elements"
            if [ -z "$lambda_aux" ]; then
                echo -e "  ${RED}[Lambda VM] FAILED to parse aux element count${NC}"
                exit 1
            fi
            echo -e "  Lambda VM: ${BOLD}${lambda_aux}${NC} aux elements (EF-cols)"
        else
            echo -e "  ${RED}[Lambda VM] FAILED${NC}"
            cat "$lambda_out_file"
            exit 1
        fi
        if [ -n "$REPORT_DIR" ]; then
            cp "$lambda_out_file" "$REPORT_DIR/raw/lambda_${n}.out"
        fi
    fi

    if $RUN_SP1; then
        echo -e "  ${GREEN}[SP1 v6] Proving to extract element count...${NC}"
        sp1v6_out_file="$TMP_DIR/sp1v6_${n}.out"
        if "$SP1_V6_BIN" "$n" > "$sp1v6_out_file" 2>&1; then
            sp1v6_elems=$(extract_elements < "$sp1v6_out_file")
            if [ -z "$sp1v6_elems" ]; then
                echo -e "  ${RED}[SP1 v6] FAILED to parse element count${NC}"
                exit 1
            fi
            echo -e "  SP1 v6:    ${BOLD}${sp1v6_elems}${NC} elements"
        else
            echo -e "  ${RED}[SP1 v6] FAILED${NC}"
            cat "$sp1v6_out_file"
            exit 1
        fi
        if [ -n "$REPORT_DIR" ]; then
            cp "$sp1v6_out_file" "$REPORT_DIR/raw/sp1v6_${n}.out"
        fi

        echo -e "  ${GREEN}[SP1 v5] Proving to extract element count...${NC}"
        sp1v5_out_file="$TMP_DIR/sp1v5_${n}.out"
        if "$SP1_V5_BIN" "$SP1_V5_GUEST_ELF" "$n" > "$sp1v5_out_file" 2>&1; then
            sp1v5_elems=$(extract_elements < "$sp1v5_out_file")
            sp1v5_aux=$(extract_aux_elements < "$sp1v5_out_file")
            if [ -z "$sp1v5_elems" ]; then
                echo -e "  ${RED}[SP1 v5] FAILED to parse element count${NC}"
                exit 1
            fi
            echo -e "  SP1 v5:    ${BOLD}${sp1v5_elems}${NC} main elements"
            if [ -z "$sp1v5_aux" ]; then
                echo -e "  ${RED}[SP1 v5] FAILED to parse aux element count${NC}"
                exit 1
            fi
            echo -e "  SP1 v5:    ${BOLD}${sp1v5_aux}${NC} aux elements (EF-cols)"
        else
            echo -e "  ${RED}[SP1 v5] FAILED${NC}"
            cat "$sp1v5_out_file"
            exit 1
        fi
        if [ -n "$REPORT_DIR" ]; then
            cp "$sp1v5_out_file" "$REPORT_DIR/raw/sp1v5_${n}.out"
        fi
    fi

    if $RUN_ZISK; then
        echo -e "  ${GREEN}[ZisK] Executing to extract element count...${NC}"
        zisk_out_file="$TMP_DIR/zisk_${n}.out"
        if "$ZISK_HOST_BIN" "$n" > "$zisk_out_file" 2>&1; then
            zisk_elems=$(extract_elements < "$zisk_out_file")
            if [ -z "$zisk_elems" ]; then
                echo -e "  ${RED}[ZisK] FAILED to parse element count${NC}"
                exit 1
            fi
            echo -e "  ZisK:      ${BOLD}${zisk_elems}${NC} main elements"
        else
            echo -e "  ${RED}[ZisK] FAILED${NC}"
            cat "$zisk_out_file"
            exit 1
        fi
        if [ -n "$REPORT_DIR" ]; then
            cp "$zisk_out_file" "$REPORT_DIR/raw/zisk_${n}.out"
        fi
    fi

    RESULT_N+=("$n")
    RESULT_LAMBDA+=("$lambda_elems")
    RESULT_LAMBDA_AUX+=("$lambda_aux")
    RESULT_SP1_V6+=("$sp1v6_elems")
    RESULT_SP1_V5+=("$sp1v5_elems")
    RESULT_SP1_V5_AUX+=("$sp1v5_aux")
    RESULT_ZISK+=("$zisk_elems")
    echo ""
done

# --- Summary tables ---------------------------------------------------------
W_N=13; W_E=18; W_R=11

print_main_table() {
    echo "=== Summary: Main-Trace Field Elements (base field, rows × cols) ==="
    echo "  Program : Fibonacci (u64 wrapping)"
    echo "  Metric  : sum of padded_rows × num_columns across all AIR tables"
    echo "  Fields  : Lambda VM = Goldilocks 64-bit | SP1 = BabyBear 32-bit | ZisK = Goldilocks 64-bit"
    echo ""
    printf "  %-${W_N}s  %${W_E}s  %${W_E}s  %${W_E}s  %${W_E}s  %${W_R}s  %${W_R}s  %${W_R}s\n" \
        "Fibonacci n" "Lambda VM" "SP1 v6" "SP1 v5" "ZisK" "vs SP1 v6" "vs SP1 v5" "vs ZisK"
    printf "  %-${W_N}s  %${W_E}s  %${W_E}s  %${W_E}s  %${W_E}s  %${W_R}s  %${W_R}s  %${W_R}s\n" \
        "-------------" "------------------" "------------------" "------------------" "------------------" "-----------" "-----------" "-----------"
    for i in "${!RESULT_N[@]}"; do
        lam="${RESULT_LAMBDA[$i]}"
        v6="${RESULT_SP1_V6[$i]}"
        v5="${RESULT_SP1_V5[$i]}"
        zk="${RESULT_ZISK[$i]}"
        printf "  %-${W_N}s  %${W_E}s  %${W_E}s  %${W_E}s  %${W_E}s  %${W_R}s  %${W_R}s  %${W_R}s\n" \
            "$(fmt_num "${RESULT_N[$i]}")" \
            "$(fmt_num "$lam")" \
            "$(fmt_num "$v6")" \
            "$(fmt_num "$v5")" \
            "$(fmt_num "$zk")" \
            "$(format_ratio "$lam" "$v6")" \
            "$(format_ratio "$lam" "$v5")" \
            "$(format_ratio "$lam" "$zk")"
    done
    echo ""
    echo "  +Nx = Lambda has N× more elements than the other prover"
    echo "  -Nx = the other prover has N× more elements than Lambda"
    echo ""
}

print_aux_table() {
    echo "=== Summary: Aux-Trace Field Elements (EF columns × rows) ==="
    echo "  Metric  : sum of padded_rows × committed_EF_columns across all AIR tables"
    echo "  Unit    : EF columns (Lambda VM = ⌈bus_interactions/2⌉, SP1 v5 = permutation_width)"
    echo "  Fields  : Lambda VM = Goldilocks cubic EF (3 BF/EF) | SP1 v5 = BabyBear quartic EF (4 BF/EF)"
    echo "  Note    : SP1 v6 and ZisK have no committed interaction columns to compare"
    echo "            (SP1 v6 uses a GKR-based bus; ZisK rolls lookup columns into its main trace)"
    echo ""
    printf "  %-${W_N}s  %${W_E}s  %${W_E}s  %${W_R}s\n" \
        "Fibonacci n" "Lambda VM" "SP1 v5" "vs SP1 v5"
    printf "  %-${W_N}s  %${W_E}s  %${W_E}s  %${W_R}s\n" \
        "-------------" "------------------" "------------------" "-----------"
    for i in "${!RESULT_N[@]}"; do
        lam_aux="${RESULT_LAMBDA_AUX[$i]}"
        v5_aux="${RESULT_SP1_V5_AUX[$i]}"
        printf "  %-${W_N}s  %${W_E}s  %${W_E}s  %${W_R}s\n" \
            "$(fmt_num "${RESULT_N[$i]}")" \
            "$(fmt_num "$lam_aux")" \
            "$(fmt_num "$v5_aux")" \
            "$(format_ratio "$lam_aux" "$v5_aux")"
    done
    echo ""
    echo "  +Nx = Lambda has N× more aux elements than SP1 v5"
    echo "  -Nx = SP1 v5 has N× more aux elements than Lambda"
    echo ""
}

echo -e "${BOLD}$(print_main_table)${NC}"
echo -e "${BOLD}$(print_aux_table)${NC}"
echo "Raw outputs in $TMP_DIR/"

if [ -n "$REPORT_DIR" ]; then
    { print_main_table; print_aux_table; } > "$REPORT_DIR/summary.md"
    echo "Report written to $REPORT_DIR/"
fi
