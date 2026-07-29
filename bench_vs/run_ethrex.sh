#!/usr/bin/env bash
# Benchmark: Lambda VM proving ethrex blocks.
#
# Proves and verifies each block in BLOCKS (an ethrex guest ELF + a serialized
# ProgramInput private input) and reports proving time, verification time and cycle
# count. Add a block by appending a "label|input_basename|continuations|epoch_size_log2"
# entry to BLOCKS — the input file must live in executor/tests/, and is generated on
# demand when its name matches ethrex_<N>_transfers.bin.
#
# Usage: ./bench_vs/run_ethrex.sh [--report-dir DIR] [--cont-txs N] [--rebuild-elf] [--no-color]
#
# Env:
#   BENCH_FEATURES  extra cargo features for the cli build (e.g. "jemalloc-stats,prover/cuda"
#                   to bench the GPU prover path). Unset = default features.
#
# Prerequisites:
#   - Lambda VM CLI build dependencies available
#   - RISC-V sysroot: auto-provisioned by the guest ELF build (the .elf rules depend on
#     `make prepare-sysroot`). Override the location with SYSROOT_DIR (default /opt/lambda-vm-sysroot).
#   - Rust stable + nightly-2026-02-01 installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_ethrex"
REPORT_DIR=""
NO_COLOR=false
REBUILD_ELF=false
# Transfer count of the continuation block. Lower it (e.g. 10) for a quick run.
CONT_TXS=100
BENCH_FEATURES="${BENCH_FEATURES:-}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# Blocks to benchmark: "label|input_basename|continuations|epoch_size_log2"
# (input lives in executor/tests/; epoch_size_log2 is "-" when monolithic).
# Populated after arg parsing so --cont-txs can size the continuation block.
BLOCKS=()

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
        --cont-txs)
            if [[ $# -lt 2 ]]; then echo "--cont-txs requires an argument"; exit 1; fi
            case $2 in
                ''|*[!0-9]*) echo "--cont-txs must be a positive integer (got: $2)"; exit 1 ;;
            esac
            if [ "$2" -lt 1 ]; then echo "--cont-txs must be at least 1"; exit 1; fi
            CONT_TXS=$2
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--report-dir DIR] [--cont-txs N] [--rebuild-elf] [--no-color]"
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

# Epoch size for the continuation block, as log2(cycles). Pinned rather than left to the
# CLI default so a default change can't silently move the series.
#
# 22 (not the CLI's 20, and not scripts/bench_abba.sh's 20 either — the comment-triggered
# /bench-gpu and /bench-abba benches still run at 20) from a sweep on the 100-transfer
# block, RTX 5090 + Ryzen 9 7950X:
#   log2  epochs  prove GPU  prove CPU  verify  proof     peak heap CPU
#   20    40      199.1s     342.5s     30.5s   2.01 GiB  11.3 GB
#   22    10      106.4s     231.4s     10.1s   0.80 GiB  25.5 GB
# Fewer, larger epochs win on every axis except memory: the bundle carries one proof per
# epoch, so prove time, verify time and proof size all shrink with the epoch count.
#
# 22 is also the CEILING for the GPU path, not just a memory choice: a proof over more than
# 2^22 padded rows fails under `prover/cuda` — 2^23 panics at
# crypto/stark/src/constraints/evaluator.rs:242 ("R2 composition fell back to the host
# trace, but it is device-only") and 2^24 fails as `Fft("resident aux LDE failed; host aux
# trace is empty")`. It bites per proof, so a continuation epoch is affected exactly as a
# monolithic prove is: on CPU log2=23 works and is faster (fibonacci-32M: 112.1s vs 124.9s
# at 22), on GPU it aborts.
#
# Root cause measured on a 32 GB RTX 5090: at 2^23 rows the device working set is already
# ~24.5 GB, GPU composition needs more than the remaining headroom, the allocation fails
# and is swallowed into an Option (gpu_main()/gpu_aux() -> None), and the CPU fallback then
# reads a host trace that the device-only gate never materialised. VRAM exhaustion surfacing
# as a mis-gate. `LAMBDA_VM_DISABLE_GPU_COMPOSITION=1` works around it (8M monolithic then
# proves in 17.4s), at the cost of the composition speedup. A card with less VRAM breaks at
# a smaller size. Raise this only once the composition path is in the VRAM budget.
CONT_EPOCH_SIZE_LOG2=22

BLOCKS=(
    "ethrex empty block|ethrex_empty_block.bin|0|-"
    "ethrex 1 tx|ethrex_simple_tx.bin|0|-"
    "ethrex ${CONT_TXS}tx cont|ethrex_${CONT_TXS}_transfers.bin|1|$CONT_EPOCH_SIZE_LOG2"
)

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

extract_verification_time() {
    sed -nE '/Verification time: [0-9.]+s/ {
        s/.*Verification time: ([0-9.]+)s.*/\1/
        p
        q
    }'
}

extract_epochs() {
    sed -nE '/^Epochs: [0-9]+/ {
        s/^Epochs: ([0-9]+).*/\1/
        p
        q
    }'
}

# Anchored: must not match "Prover peak heap:" if that line is ever added.
extract_peak_heap_mb() {
    sed -nE '/^Peak heap: [0-9]+ MB/ {
        s/^Peak heap: ([0-9]+) MB.*/\1/
        p
        q
    }'
}

# slugify a label into a metrics-key/filename-safe token
slugify() {
    printf "%s" "$1" | tr '[:upper:] ' '[:lower:]_' | tr -cd '[:alnum:]_'
}

# Generate a missing ethrex_<N>_transfers.bin fixture (the large ones aren't committed).
ensure_input() {
    local path=$1 basename=$2 txs
    [ -f "$path" ] && return 0
    txs=$(printf "%s" "$basename" | sed -nE 's/^ethrex_([0-9]+)_transfers\.bin$/\1/p')
    if [ -z "$txs" ]; then
        echo -e "${RED}Input file not found and not generatable: $path${NC}"
        return 1
    fi
    echo -e "  ${GREEN}[Lambda VM] Generating ${txs}-transfer fixture...${NC}"
    ( cd "$ROOT_DIR/tooling/ethrex-fixtures" && cargo build --release 2>&1 | tail -3 )
    "$ROOT_DIR/tooling/ethrex-fixtures/target/release/ethrex-fixtures" "$txs" "$path" distinct
}

# --- Pre-build --------------------------------------------------------------

CLI="$ROOT_DIR/target/release/cli"
ETHREX_ELF="$ROOT_DIR/executor/program_artifacts/rust/ethrex.elf"
echo -e "${BOLD}=== Ethrex Block Benchmarks: Lambda VM ===${NC}"
echo ""

if [ -n "$BENCH_FEATURES" ]; then
    echo -e "${GREEN}[Lambda VM] Building CLI (features: $BENCH_FEATURES)...${NC}"
    cargo build --release -p cli --manifest-path "$ROOT_DIR/Cargo.toml" \
        --features "$BENCH_FEATURES" 2>&1 | tail -5
else
    echo -e "${GREEN}[Lambda VM] Building CLI...${NC}"
    cargo build --release -p cli --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -5
fi

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
verify_times=()
epochs_arr=()
peak_heaps=()
proof_sizes=()

for entry in "${BLOCKS[@]}"; do
    IFS='|' read -r label input_basename continuations epoch_size_log2 <<< "$entry"
    input_path="$ROOT_DIR/executor/tests/$input_basename"
    slug=$(slugify "$label")

    echo ""
    echo -e "${BOLD}--- Proving ${label} ---${NC}"

    if ! ensure_input "$input_path" "$input_basename"; then
        exit 1
    fi

    proof_file="$TMP_DIR/$slug.proof"
    stderr_file="$TMP_DIR/$slug.stderr"

    if [ "$continuations" = "1" ]; then
        # verify takes --continuations but not --epoch-size-log2.
        cont_args=(--continuations --epoch-size-log2 "$epoch_size_log2")
        verify_args=(--continuations)
        echo -e "  ${GREEN}[Lambda VM] Proving (continuations, epoch_size_log2=$epoch_size_log2)...${NC}"
    else
        cont_args=()
        verify_args=()
        echo -e "  ${GREEN}[Lambda VM] Proving...${NC}"
    fi

    # ${arr[@]+...}: expanding an empty array under `set -u` errors on bash 3.2 (macOS).
    if ! lambda_output=$("$CLI" prove "$ETHREX_ELF" \
            -o "$proof_file" \
            --private-input "$input_path" \
            ${cont_args[@]+"${cont_args[@]}"} \
            --time --cycles 2>"$stderr_file"); then
        echo -e "  ${RED}[Lambda VM] FAILED:${NC}"
        cat "$stderr_file"
        exit 1
    fi

    lambda_time=$(printf "%s\n" "$lambda_output" | extract_proving_time)
    lambda_cycles=$(printf "%s\n" "$lambda_output" | extract_cycles)
    lambda_epochs=$(printf "%s\n" "$lambda_output" | extract_epochs)
    lambda_peak_heap=$(printf "%s\n" "$lambda_output" | extract_peak_heap_mb)
    proof_size=$(wc -c < "$proof_file" | tr -d ' ')

    if [ -z "$lambda_time" ]; then
        echo -e "  ${RED}[Lambda VM] FAILED: could not parse proving time${NC}"
        printf "%s\n" "$lambda_output"
        exit 1
    fi
    : "${lambda_cycles:=n/a}"
    : "${lambda_epochs:=n/a}"
    : "${lambda_peak_heap:=n/a}"

    if [ "$lambda_cycles" != "n/a" ]; then
        echo -e "  Lambda VM: ${BOLD}${lambda_time}s${NC} (${lambda_cycles} cycles)"
    else
        echo -e "  Lambda VM: ${BOLD}${lambda_time}s${NC}"
    fi

    # Verify the proof we just produced, then drop it (a continuation bundle is GBs).
    verify_stderr="$TMP_DIR/$slug.verify.stderr"
    echo -e "  ${GREEN}[Lambda VM] Verifying...${NC}"
    if ! verify_output=$("$CLI" verify "$proof_file" "$ETHREX_ELF" \
            ${verify_args[@]+"${verify_args[@]}"} \
            --time 2>"$verify_stderr"); then
        echo -e "  ${RED}[Lambda VM] VERIFY FAILED:${NC}"
        cat "$verify_stderr"
        exit 1
    fi
    rm -f "$proof_file"
    lambda_verify=$(printf "%s\n" "$verify_output" | extract_verification_time)
    : "${lambda_verify:=n/a}"
    echo -e "  Verify:    ${BOLD}${lambda_verify}s${NC} (proof ${proof_size} bytes)"

    if [ -n "$REPORT_DIR" ]; then
        printf "%s\n" "$lambda_output" > "$REPORT_DIR/raw/$slug.stdout"
        printf "%s\n" "$verify_output" > "$REPORT_DIR/raw/$slug.verify.stdout"
        cp "$stderr_file" "$REPORT_DIR/raw/$slug.stderr"
    fi

    labels+=("$label")
    times+=("$lambda_time")
    cycles_arr+=("$lambda_cycles")
    verify_times+=("$lambda_verify")
    epochs_arr+=("$lambda_epochs")
    peak_heaps+=("$lambda_peak_heap")
    proof_sizes+=("$proof_size")
done

# --- Summary table ----------------------------------------------------------

echo ""
echo -e "${BOLD}=== Summary ===${NC}"
echo ""

printf "  %-22s  %12s  %12s  %14s  %8s  %10s\n" \
    "Program" "Prove (s)" "Verify (s)" "Cycles" "Epochs" "Heap (MB)"
printf "  %-22s  %12s  %12s  %14s  %8s  %10s\n" \
    "----------------------" "---------" "----------" "--------------" "------" "---------"
for i in "${!labels[@]}"; do
    printf "  %-22s  %11ss  %11ss  %14s  %8s  %10s\n" \
        "${labels[$i]}" "${times[$i]}" "${verify_times[$i]}" "${cycles_arr[$i]}" \
        "${epochs_arr[$i]}" "${peak_heaps[$i]}"
done

echo ""
echo -e "Prove is end-to-end (excludes verification); verify is timed separately."
echo -e "Heap is peak jemalloc; n/a unless the CLI was built with the jemalloc-stats feature."
echo "Raw data in $TMP_DIR/"

# --- Machine-readable report ------------------------------------------------

if [ -n "$REPORT_DIR" ]; then
    {
        echo "timing_window=end_to_end_prove_verify_timed_separately"
        echo "bench_features=${BENCH_FEATURES:-default}"
        echo "cont_epoch_size_log2=$CONT_EPOCH_SIZE_LOG2"
        for i in "${!labels[@]}"; do
            slug=$(slugify "${labels[$i]}")
            echo "${slug}_time_s=${times[$i]}"
            echo "${slug}_cycles=${cycles_arr[$i]}"
            echo "${slug}_verify_s=${verify_times[$i]}"
            echo "${slug}_epochs=${epochs_arr[$i]}"
            echo "${slug}_peak_heap_mb=${peak_heaps[$i]}"
            echo "${slug}_proof_bytes=${proof_sizes[$i]}"
        done
    } > "$REPORT_DIR/ethrex_metrics.txt"

    {
        echo "# Ethrex Block Benchmarks — Lambda VM"
        echo
        echo "Prove is end-to-end (excludes verification); verify is timed separately."
        echo "Build features: \`${BENCH_FEATURES:-default}\`."
        echo
        echo "| Program | Prove (s) | Verify (s) | Cycles | Epochs | Peak heap (MB) | Proof (bytes) |"
        echo "|---------|----------:|-----------:|-------:|-------:|---------------:|--------------:|"
        for i in "${!labels[@]}"; do
            printf "| %s | %s | %s | %s | %s | %s | %s |\n" \
                "${labels[$i]}" "${times[$i]}" "${verify_times[$i]}" "${cycles_arr[$i]}" \
                "${epochs_arr[$i]}" "${peak_heaps[$i]}" "${proof_sizes[$i]}"
        done
    } > "$REPORT_DIR/ethrex_summary.md"
fi
