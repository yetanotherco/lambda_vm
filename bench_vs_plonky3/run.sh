#!/bin/bash
# Benchmark: Lambda STARK vs Plonky3 — single-shot prove time on the shared
# Fibonacci AIR (columns = 2 * num_sequences, blowup = 2, fri_queries = 219).
#
# Usage:
#   ./bench_vs_plonky3/run.sh [--log-rows K ...] [--num-sequences N] [--runs N]
#                             [--lambda-only | --p3-only] [--report-dir DIR]
#                             [--no-p3-patch] [--scalar] [--no-color]
#
# Defaults: --log-rows 19, --num-sequences 16, --runs 3.
# With multiple --log-rows values, prints one median row per size.
#
# --scalar: disables SIMD at the target-feature level. On x86_64 drops AVX2
# and AVX-512 (Goldilocks + most of Keccak go scalar, residual SSE2 in
# p3-keccak). On aarch64 drops the SHA3 NEON extension. Triggers a rebuild
# when toggling; subsequent runs with the same RUSTFLAGS are cached.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_p3"
REPORT_DIR=""
NO_COLOR=false
NO_P3_PATCH=false
SCALAR=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

LOG_ROWS=()
NUM_SEQUENCES=16
RUNS=3
RUN_LAMBDA=true
RUN_P3=true

# --- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --log-rows)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LOG_ROWS+=("$1")
                shift
            done
            ;;
        --num-sequences)
            if [[ $# -lt 2 ]]; then echo "--num-sequences requires an argument"; exit 1; fi
            NUM_SEQUENCES=$2
            shift 2
            ;;
        --runs)
            if [[ $# -lt 2 ]]; then echo "--runs requires an argument"; exit 1; fi
            RUNS=$2
            shift 2
            ;;
        --lambda-only)
            RUN_P3=false
            shift
            ;;
        --p3-only)
            RUN_LAMBDA=false
            shift
            ;;
        --report-dir)
            if [[ $# -lt 2 ]]; then echo "--report-dir requires an argument"; exit 1; fi
            REPORT_DIR=$2
            shift 2
            ;;
        --no-p3-patch)
            NO_P3_PATCH=true
            shift
            ;;
        --scalar)
            SCALAR=true
            shift
            ;;
        --no-color)
            NO_COLOR=true
            shift
            ;;
        -h|--help)
            sed -n '2,11p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ${#LOG_ROWS[@]} -eq 0 ]; then
    LOG_ROWS=(19)
fi

if ! $RUN_LAMBDA && ! $RUN_P3; then
    echo "At least one prover must be enabled"
    exit 1
fi

if [ "$RUNS" -lt 1 ]; then
    echo "--runs must be >= 1"
    exit 1
fi

if $NO_COLOR; then
    RED=''
    GREEN=''
    YELLOW=''
    BOLD=''
    NC=''
fi

mkdir -p "$TMP_DIR"
rm -rf "$TMP_DIR"/*

if [ -n "$REPORT_DIR" ]; then
    mkdir -p "$REPORT_DIR/raw"
fi

# --- Patch toggle -----------------------------------------------------------
# The root Cargo.toml has a [patch.crates-io] block pointing at the vendored
# p3-goldilocks-patched (adds BinomiallyExtendable<3>, disables NEON). For the
# nightly we build against vanilla crates.io p3-goldilocks — we comment the
# block out and drop the `p3-degree3` feature.
CARGO_TOML="$ROOT_DIR/Cargo.toml"
CARGO_TOML_BAK=""
BUILD_FEATURE_FLAGS=()
if $NO_P3_PATCH; then
    CARGO_TOML_BAK="$CARGO_TOML.bak.p3bench.$$"
    cp "$CARGO_TOML" "$CARGO_TOML_BAK"
    # Comment the [patch.crates-io] block and its entries (until the next blank
    # line or next [section]).
    python3 - "$CARGO_TOML" <<'PY'
import sys, pathlib
path = pathlib.Path(sys.argv[1])
lines = path.read_text().splitlines(keepends=True)
out = []
in_patch = False
for ln in lines:
    stripped = ln.strip()
    if stripped == "[patch.crates-io]":
        in_patch = True
        out.append("# " + ln if not ln.startswith("#") else ln)
        continue
    if in_patch:
        if stripped.startswith("[") and stripped.endswith("]"):
            in_patch = False
            out.append(ln)
            continue
        if stripped == "":
            in_patch = False
            out.append(ln)
            continue
        out.append("# " + ln if not ln.startswith("#") else ln)
    else:
        out.append(ln)
path.write_text("".join(out))
PY
    trap 'if [ -n "$CARGO_TOML_BAK" ] && [ -f "$CARGO_TOML_BAK" ]; then mv "$CARGO_TOML_BAK" "$CARGO_TOML"; fi' EXIT INT TERM
    BUILD_FEATURE_FLAGS=(--no-default-features --features parallel)
fi

# --- Scalar (no SIMD) toggle ------------------------------------------------
# When --scalar is on, disable vector instruction sets for the build so both
# provers run against the same scalar baseline. p3-keccak keeps SSE2 residual
# on x86 — acceptable per the bench workstream (contribution is ~7%).
#   x86_64   → -avx2,-avx512f         (Goldilocks + most of Keccak go scalar)
#   aarch64  → -sha3                   (drops Keccak NEON SHA3 extension)
# Cargo caches per-RUSTFLAGS, so toggling scalar vs vector triggers a rebuild
# on first use but is cached afterwards.
SCALAR_RUSTFLAGS=""
if $SCALAR; then
    case "$(uname -m)" in
        x86_64|amd64)
            SCALAR_RUSTFLAGS="-C target-feature=-avx2,-avx512f"
            ;;
        arm64|aarch64)
            SCALAR_RUSTFLAGS="-C target-feature=-sha3"
            ;;
        *)
            echo "warning: --scalar: unknown arch $(uname -m); not pinning RUSTFLAGS" >&2
            ;;
    esac
    if [ -n "$SCALAR_RUSTFLAGS" ]; then
        if [ -n "${RUSTFLAGS:-}" ]; then
            export RUSTFLAGS="${RUSTFLAGS} ${SCALAR_RUSTFLAGS}"
        else
            export RUSTFLAGS="$SCALAR_RUSTFLAGS"
        fi
    fi
fi

# --- Build ------------------------------------------------------------------
echo -e "${BOLD}=== STARK prove benchmark: Lambda vs Plonky3 ===${NC}"
echo -e "  log-rows:       ${YELLOW}${LOG_ROWS[*]}${NC}"
echo -e "  num-sequences:  ${YELLOW}${NUM_SEQUENCES}${NC}  (columns = $((2 * NUM_SEQUENCES)))"
echo -e "  runs/size:      ${YELLOW}${RUNS}${NC}  (median reported)"
if $NO_P3_PATCH; then
    echo -e "  p3 extension:   ${YELLOW}degree 2 (vanilla, no patch)${NC}"
else
    echo -e "  p3 extension:   ${YELLOW}degree 3 (patched, matches Lambda)${NC}"
fi
if $SCALAR; then
    echo -e "  scalar mode:    ${YELLOW}on${NC}  (arch=$(uname -m), RUSTFLAGS=\"${RUSTFLAGS:-}\")"
else
    echo -e "  scalar mode:    ${YELLOW}off${NC}  (SIMD enabled, compiler default)"
fi
echo ""

echo -e "${GREEN}[build]${NC} prove_bench"
# Use the `${arr[@]+...}` expansion so `set -u` doesn't blow up when the
# feature-flag array is empty (bash 3 on macOS).
cargo build --release -p bench-vs-plonky3 --bin prove_bench \
    --manifest-path "$ROOT_DIR/Cargo.toml" \
    ${BUILD_FEATURE_FLAGS[@]+"${BUILD_FEATURE_FLAGS[@]}"} 2>&1 | tail -5

BIN="$ROOT_DIR/target/release/prove_bench"
if [ ! -x "$BIN" ]; then
    echo -e "${RED}[build] prove_bench not produced at $BIN${NC}"
    exit 1
fi

# --- Helpers ----------------------------------------------------------------
extract_proving_time() {
    sed -nE '/Proving time: [0-9.]+s/ {
        s/.*Proving time: ([0-9.]+)s.*/\1/
        p
        q
    }'
}

median_of() {
    # prints median of the given numeric arguments (rounded to 3 decimals).
    # Uses shell `sort -g` for portability (macOS awk lacks gawk's asort).
    printf '%s\n' "$@" | LC_ALL=C sort -g | LC_NUMERIC=C awk '
        { a[NR] = $0 + 0 }
        END {
            if (NR == 0) { print "n/a"; exit }
            if (NR % 2 == 1) {
                printf "%.3f\n", a[(NR + 1) / 2]
            } else {
                printf "%.3f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2
            }
        }'
}

ratio_fmt() {
    LC_NUMERIC=C awk -v num="$1" -v den="$2" 'BEGIN {
        if (den + 0 == 0) { print "n/a"; exit }
        printf "%.3f\n", num / den
    }'
}

# --- Run benchmark ----------------------------------------------------------

RESULT_LOG_ROWS=()
RESULT_ROWS=()
RESULT_LAMBDA=()
RESULT_P3=()
RESULT_RATIO=()

run_prover() {
    local prover=$1   # lambda | p3
    local log_rows=$2
    local times=()
    for run_i in $(seq 1 "$RUNS"); do
        local out_file="$TMP_DIR/${prover}_${log_rows}_${run_i}.stdout"
        if ! "$BIN" --prover "$prover" \
                --log-rows "$log_rows" \
                --num-sequences "$NUM_SEQUENCES" > "$out_file" 2>&1; then
            echo -e "  ${RED}[${prover}] FAILED on log-rows=${log_rows} run ${run_i}${NC}"
            cat "$out_file"
            exit 1
        fi
        local t
        t=$(extract_proving_time < "$out_file")
        if [ -z "$t" ]; then
            echo -e "  ${RED}[${prover}] could not parse proving time (log-rows=${log_rows}, run ${run_i})${NC}"
            cat "$out_file"
            exit 1
        fi
        times+=("$t")
        if [ -n "$REPORT_DIR" ]; then
            cp "$out_file" "$REPORT_DIR/raw/${prover}_log${log_rows}_run${run_i}.stdout"
        fi
    done
    median_of "${times[@]}"
    printf '%s\n' "${times[@]}" > "$TMP_DIR/${prover}_${log_rows}.times"
}

for lr in "${LOG_ROWS[@]}"; do
    rows=$((1 << lr))
    echo -e "${BOLD}--- log-rows=${lr}  (rows = ${rows}) ---${NC}"

    lambda_median="n/a"
    p3_median="n/a"

    if $RUN_LAMBDA; then
        echo -ne "  ${GREEN}[lambda]${NC} "
        lambda_median=$(run_prover lambda "$lr")
        echo "median ${BOLD}${lambda_median}s${NC} from $RUNS runs: $(paste -sd, "$TMP_DIR/lambda_${lr}.times")"
    fi

    if $RUN_P3; then
        echo -ne "  ${GREEN}[p3]${NC}     "
        p3_median=$(run_prover p3 "$lr")
        echo "median ${BOLD}${p3_median}s${NC} from $RUNS runs: $(paste -sd, "$TMP_DIR/p3_${lr}.times")"
    fi

    local_ratio="n/a"
    if $RUN_LAMBDA && $RUN_P3; then
        local_ratio=$(ratio_fmt "$lambda_median" "$p3_median")
    fi

    RESULT_LOG_ROWS+=("$lr")
    RESULT_ROWS+=("$rows")
    RESULT_LAMBDA+=("$lambda_median")
    RESULT_P3+=("$p3_median")
    RESULT_RATIO+=("$local_ratio")
done

# --- Summary table ----------------------------------------------------------

echo ""
echo -e "${BOLD}=== Summary ===${NC}"
if $RUN_LAMBDA && $RUN_P3; then
    printf "  %-9s  %-12s  %14s  %14s  %10s\n" "log-rows" "rows" "Lambda (s)" "P3 (s)" "L/P3"
    printf "  %-9s  %-12s  %14s  %14s  %10s\n" "--------" "----" "----------" "------" "----"
else
    printf "  %-9s  %-12s  %14s\n" "log-rows" "rows" "Time (s)"
    printf "  %-9s  %-12s  %14s\n" "--------" "----" "--------"
fi

for i in "${!RESULT_LOG_ROWS[@]}"; do
    lr="${RESULT_LOG_ROWS[$i]}"
    rows="${RESULT_ROWS[$i]}"
    lt="${RESULT_LAMBDA[$i]}"
    pt="${RESULT_P3[$i]}"
    rt="${RESULT_RATIO[$i]}"
    if $RUN_LAMBDA && $RUN_P3; then
        color=$GREEN
        if awk -v l="$lt" -v p="$pt" 'BEGIN{ exit !(l+0 > p+0) }'; then
            color=$RED
        fi
        printf "  %-9s  %-12s  %13ss  %13ss  ${color}%9sx${NC}\n" \
            "$lr" "$rows" "$lt" "$pt" "$rt"
    elif $RUN_LAMBDA; then
        printf "  %-9s  %-12s  %13ss\n" "$lr" "$rows" "$lt"
    else
        printf "  %-9s  %-12s  %13ss\n" "$lr" "$rows" "$pt"
    fi
done

echo ""
if $RUN_LAMBDA && $RUN_P3; then
    echo -e "Timing window: single-shot end-to-end prove. Ratio < 1 → Lambda faster."
fi
if $NO_P3_PATCH; then
    echo -e "${YELLOW}Note:${NC} Plonky3 was built without the degree-3 patch; Challenge type is degree-2."
    echo -e "      Lambda keeps degree-3 — extension fields differ across sides."
fi

# --- Machine-readable report ------------------------------------------------

if [ -n "$REPORT_DIR" ]; then
    {
        printf "log_rows\trows\tlambda_median_s\tp3_median_s\tratio_lambda_over_p3\truns\n"
        for i in "${!RESULT_LOG_ROWS[@]}"; do
            printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
                "${RESULT_LOG_ROWS[$i]}" \
                "${RESULT_ROWS[$i]}" \
                "${RESULT_LAMBDA[$i]}" \
                "${RESULT_P3[$i]}" \
                "${RESULT_RATIO[$i]}" \
                "$RUNS"
        done
    } > "$REPORT_DIR/results.tsv"

    {
        echo "# Lambda STARK vs Plonky3 Benchmark"
        echo
        echo "Timing window: \`single-shot end-to-end prove\` (no verification)."
        echo "num-sequences: \`$NUM_SEQUENCES\`, columns: \`$((2 * NUM_SEQUENCES))\`, blowup: 2, fri_queries: 219, grinding: 0."
        echo "runs per size: \`$RUNS\` (median reported)."
        echo "arch: \`$(uname -m)\`, scalar mode: \`$($SCALAR && echo on || echo off)\`."
        if $SCALAR && [ -n "$SCALAR_RUSTFLAGS" ]; then
            echo "RUSTFLAGS: \`$SCALAR_RUSTFLAGS\`."
        fi
        if $NO_P3_PATCH; then
            echo
            echo "> Plonky3 built without the vendored degree-3 patch: Challenge type is degree-2 (vanilla crates.io p3-goldilocks 0.5.2). Lambda still uses degree 3."
        fi
        echo
        echo "| log-rows | rows | Lambda (s) | P3 (s) | Lambda / P3 |"
        echo "|---------:|-----:|-----------:|-------:|------------:|"
        for i in "${!RESULT_LOG_ROWS[@]}"; do
            printf "| %s | %s | %s | %s | %s |\n" \
                "${RESULT_LOG_ROWS[$i]}" \
                "${RESULT_ROWS[$i]}" \
                "${RESULT_LAMBDA[$i]}" \
                "${RESULT_P3[$i]}" \
                "${RESULT_RATIO[$i]}"
        done
    } > "$REPORT_DIR/summary.md"
fi
