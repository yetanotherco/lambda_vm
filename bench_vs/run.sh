#!/bin/bash
# Benchmark: Lambda VM vs SP1 v6 — Fibonacci proving time comparison.
#
# Usage: ./bench_vs/run.sh [-n 1000 50000 100000] [--lambda-only | --sp1-only]
#                         [--report-dir DIR] [--no-color]
#
# Without -n, runs the default series: 1000 10000 100000 300000
#
# Prerequisites:
#   - Lambda VM CLI build dependencies available
#   - SP1 toolchain installed (or available in PATH for CI)
#   - Rust stable + nightly-2026-02-01 installed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_fib"
REPORT_DIR=""
NO_COLOR=false
TARGET_STEPS=500000000

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# --- Defaults ---------------------------------------------------------------
DEFAULT_SERIES=(1000 10000 100000 300000)
SERIES=()
RUN_LAMBDA=true
RUN_SP1=true

# --- Parse args -------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        -n)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SERIES+=("$1")
                shift
            done
            ;;
        --lambda-only)
            RUN_SP1=false
            shift
            ;;
        --sp1-only)
            RUN_LAMBDA=false
            shift
            ;;
        --report-dir)
            REPORT_DIR=$2
            shift 2
            ;;
        --no-color)
            NO_COLOR=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-n N1 N2 ...] [--lambda-only | --sp1-only] [--report-dir DIR] [--no-color]"
            echo ""
            echo "  -n N1 N2 ...      Fibonacci iteration counts (space-separated)"
            echo "                    Default series: ${DEFAULT_SERIES[*]}"
            echo "  --lambda-only     Only run Lambda VM benchmark"
            echo "  --sp1-only        Only run SP1 benchmark"
            echo "  --report-dir DIR  Write TSV, metrics, markdown summary, and raw outputs"
            echo "  --no-color        Disable ANSI colors"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ ${#SERIES[@]} -eq 0 ]; then
    SERIES=("${DEFAULT_SERIES[@]}")
fi

if ! $RUN_LAMBDA && ! $RUN_SP1; then
    echo "At least one prover must be enabled"
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

join_slash() {
    local joined=""
    local value
    for value in "$@"; do
        joined="${joined:+$joined/}$value"
    done
    printf "%s\n" "$joined"
}

fit_series() {
    local steps_slash=$1
    local values_slash=$2

    awk -v steps="$steps_slash" -v values="$values_slash" 'BEGIN {
        n = split(steps, xs, "/")
        m = split(values, ys, "/")
        if (n == 0 || n != m) {
            print "0 0 0.0000"
            exit
        }

        sx = 0; sy = 0; sxy = 0; sx2 = 0
        for (i = 1; i <= n; i++) {
            x = xs[i] / 1000000
            y = ys[i] + 0
            sx += x
            sy += y
            sxy += x * y
            sx2 += x * x
        }

        d = n * sx2 - sx * sx
        if (d == 0) {
            intercept = sy / n
            printf "0 %.6f 0.0000\n", intercept
            exit
        }

        slope = (n * sxy - sx * sy) / d
        intercept = (sy - slope * sx) / n

        my = sy / n
        ss_tot = 0
        ss_res = 0
        for (i = 1; i <= n; i++) {
            x = xs[i] / 1000000
            y = ys[i] + 0
            pred = slope * x + intercept
            ss_res += (y - pred) * (y - pred)
            ss_tot += (y - my) * (y - my)
        }

        r2 = (ss_tot > 0) ? 1 - ss_res / ss_tot : 0
        if (r2 < 0) {
            r2 = 0
        }

        printf "%.6f %.6f %.4f\n", slope, intercept, r2
    }'
}

project_series() {
    local slope=$1
    local intercept=$2
    local target_steps=$3

    awk -v slope="$slope" -v intercept="$intercept" -v target="$target_steps" 'BEGIN {
        projected = slope * (target / 1000000) + intercept
        if (projected < 0) {
            projected = 0
        }
        printf "%.3f\n", projected
    }'
}

format_hours() {
    local seconds=$1
    awk -v value="$seconds" 'BEGIN { printf "%.2f\n", value / 3600 }'
}

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

echo -e "${BOLD}=== Fibonacci Benchmark: Lambda VM vs SP1 v6 ===${NC}"
echo -e "Series: ${YELLOW}${SERIES[*]}${NC}"
echo ""

# --- Pre-build --------------------------------------------------------------

CLI="$ROOT_DIR/target/release/cli"
LAMBDA_DIR="$SCRIPT_DIR/lambda/fibonacci"
TARGET_SPEC="$ROOT_DIR/executor/programs/riscv64im-lambda-vm-elf.json"
LAMBDA_ELF="$LAMBDA_DIR/target/riscv64im-lambda-vm-elf/release/fibonacci-bench"

if $RUN_LAMBDA; then
    echo -e "${GREEN}[Lambda VM] Building CLI...${NC}"
    cargo build --release -p cli --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -5
fi

if $RUN_LAMBDA; then
    echo -e "${GREEN}[Lambda VM] Building fibonacci prover...${NC}"
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

# --- Run benchmark series ---------------------------------------------------

RESULT_N=()
RESULT_LAMBDA=()
RESULT_SP1=()
RESULT_SP1_CYCLES=()
RESULT_RATIO=()

LAMBDA_STEPS=()
LAMBDA_TIMES=()
SP1_STEPS=()
SP1_TIMES=()

if [ -n "$REPORT_DIR" ]; then
    printf "n\tlambda_time_s\tsp1_time_s\tsp1_cycles\tratio_lambda_over_sp1\n" > "$REPORT_DIR/results.tsv"
fi

run_one() {
    local n=$1
    local lambda_time="n/a"
    local sp1_time="n/a"
    local sp1_cycles="n/a"
    local ratio="n/a"

    echo ""
    echo -e "${BOLD}--- n=${n} ---${NC}"

    if $RUN_LAMBDA; then
        local input_file="$TMP_DIR/lambda_${n}.bin"
        local proof_file="$TMP_DIR/lambda_${n}.proof"
        local stderr_file="$TMP_DIR/lambda_${n}.stderr"
        write_u64_le "$n" "$input_file"

        echo -e "  ${GREEN}[Lambda VM] Proving...${NC}"
        local lambda_output
        if ! lambda_output=$("$CLI" prove "$LAMBDA_ELF" -o "$proof_file" --private-input "$input_file" --time 2>"$stderr_file"); then
            echo -e "  ${RED}[Lambda VM] FAILED:${NC}"
            cat "$stderr_file"
            exit 1
        fi
        rm -f "$proof_file"

        lambda_time=$(echo "$lambda_output" | grep -o 'Proving time: [0-9.]*s' | grep -o '[0-9.]*')
        if [ -z "$lambda_time" ]; then
            echo -e "  ${RED}[Lambda VM] FAILED: could not parse proving time${NC}"
            printf "%s\n" "$lambda_output"
            exit 1
        fi

        echo -e "  Lambda VM: ${BOLD}${lambda_time}s${NC}"
        LAMBDA_STEPS+=("$n")
        LAMBDA_TIMES+=("$lambda_time")

        if [ -n "$REPORT_DIR" ]; then
            printf "%s\n" "$lambda_output" > "$REPORT_DIR/raw/lambda_${n}.stdout"
            cp "$stderr_file" "$REPORT_DIR/raw/lambda_${n}.stderr"
        fi
    fi

    if $RUN_SP1; then
        echo -e "  ${GREEN}[SP1 v6] Proving...${NC}"
        local sp1_output_file="$TMP_DIR/sp1_${n}.stdout"
        if ! "$SP1_BIN" "$n" > "$sp1_output_file" 2>&1; then
            echo -e "  ${RED}[SP1 v6] FAILED:${NC}"
            cat "$sp1_output_file"
            exit 1
        fi

        sp1_time=$(grep -o 'Proving time: [0-9.]*s' "$sp1_output_file" | grep -o '[0-9.]*')
        sp1_cycles=$(grep -o 'Cycles: [0-9]*' "$sp1_output_file" | grep -o '[0-9]*')
        if [ -z "$sp1_time" ] || [ -z "$sp1_cycles" ]; then
            echo -e "  ${RED}[SP1 v6] FAILED: could not parse output${NC}"
            cat "$sp1_output_file"
            exit 1
        fi

        echo -e "  SP1 v6:    ${BOLD}${sp1_time}s${NC} (${sp1_cycles} cycles)"
        SP1_STEPS+=("$n")
        SP1_TIMES+=("$sp1_time")

        if [ -n "$REPORT_DIR" ]; then
            cp "$sp1_output_file" "$REPORT_DIR/raw/sp1_${n}.stdout"
        fi
    fi

    if [ "$lambda_time" != "n/a" ] && [ "$sp1_time" != "n/a" ]; then
        ratio=$(LC_NUMERIC=C awk -v lambda="$lambda_time" -v sp1="$sp1_time" 'BEGIN { printf "%.3f", lambda / sp1 }')
    fi

    RESULT_N+=("$n")
    RESULT_LAMBDA+=("$lambda_time")
    RESULT_SP1+=("$sp1_time")
    RESULT_SP1_CYCLES+=("$sp1_cycles")
    RESULT_RATIO+=("$ratio")

    if [ -n "$REPORT_DIR" ]; then
        printf "%s\t%s\t%s\t%s\t%s\n" "$n" "$lambda_time" "$sp1_time" "$sp1_cycles" "$ratio" >> "$REPORT_DIR/results.tsv"
    fi
}

for n in "${SERIES[@]}"; do
    run_one "$n"
done

# --- Projection -------------------------------------------------------------

LAMBDA_SLOPE=""
LAMBDA_INTERCEPT=""
LAMBDA_R2=""
LAMBDA_PROJECTED_S=""
LAMBDA_PROJECTED_H=""

SP1_SLOPE=""
SP1_INTERCEPT=""
SP1_R2=""
SP1_PROJECTED_S=""
SP1_PROJECTED_H=""

compute_projection() {
    local label=$1
    local steps_slash=$2
    local times_slash=$3
    local slope intercept r2 projected_s projected_h

    if [ -z "$steps_slash" ] || [ -z "$times_slash" ]; then
        return 0
    fi

    read -r slope intercept r2 <<< "$(fit_series "$steps_slash" "$times_slash")"
    projected_s=$(project_series "$slope" "$intercept" "$TARGET_STEPS")
    projected_h=$(format_hours "$projected_s")

    case "$label" in
        lambda)
            LAMBDA_SLOPE=$slope
            LAMBDA_INTERCEPT=$intercept
            LAMBDA_R2=$r2
            LAMBDA_PROJECTED_S=$projected_s
            LAMBDA_PROJECTED_H=$projected_h
            ;;
        sp1)
            SP1_SLOPE=$slope
            SP1_INTERCEPT=$intercept
            SP1_R2=$r2
            SP1_PROJECTED_S=$projected_s
            SP1_PROJECTED_H=$projected_h
            ;;
    esac
}

if $RUN_LAMBDA && [ ${#LAMBDA_STEPS[@]} -gt 0 ]; then
    compute_projection "lambda" "$(join_slash "${LAMBDA_STEPS[@]}")" "$(join_slash "${LAMBDA_TIMES[@]}")"
fi
if $RUN_SP1 && [ ${#SP1_STEPS[@]} -gt 0 ]; then
    compute_projection "sp1" "$(join_slash "${SP1_STEPS[@]}")" "$(join_slash "${SP1_TIMES[@]}")"
fi

# --- Summary table ----------------------------------------------------------

echo ""
echo -e "${BOLD}=== Summary ===${NC}"
echo -e "Program: Fibonacci (u64 wrapping)"
echo ""

if $RUN_LAMBDA && $RUN_SP1; then
    printf "  %-10s  %12s  %12s  %12s  %8s\n" "n" "Lambda VM" "SP1 v6" "SP1 cycles" "Ratio"
    printf "  %-10s  %12s  %12s  %12s  %8s\n" "---" "---------" "------" "----------" "-----"
elif $RUN_LAMBDA; then
    printf "  %-10s  %12s\n" "n" "Lambda VM"
    printf "  %-10s  %12s\n" "---" "---------"
else
    printf "  %-10s  %12s  %12s\n" "n" "SP1 v6" "SP1 cycles"
    printf "  %-10s  %12s  %12s\n" "---" "------" "----------"
fi

for i in "${!RESULT_N[@]}"; do
    n="${RESULT_N[$i]}"
    lambda_time="${RESULT_LAMBDA[$i]}"
    sp1_time="${RESULT_SP1[$i]}"
    sp1_cycles="${RESULT_SP1_CYCLES[$i]}"
    ratio="${RESULT_RATIO[$i]}"

    if $RUN_LAMBDA && $RUN_SP1; then
        if [ "$ratio" != "n/a" ]; then
            ratio_colored=$(LC_NUMERIC=C awk -v ratio="$ratio" 'BEGIN { printf "%.1fx", ratio }')
            if (( $(LC_NUMERIC=C awk -v lambda="$lambda_time" -v sp1="$sp1_time" 'BEGIN { print (lambda > sp1) }') )); then
                ratio_colored="${RED}${ratio_colored}${NC}"
            else
                ratio_colored="${GREEN}${ratio_colored}${NC}"
            fi
            printf "  %-10s  %11ss  %11ss  %12s  " "$n" "$lambda_time" "$sp1_time" "$sp1_cycles"
            echo -e "$ratio_colored"
        else
            printf "  %-10s  %12s  %12s  %12s  %8s\n" "$n" "${lambda_time}s" "${sp1_time}s" "$sp1_cycles" "-"
        fi
    elif $RUN_LAMBDA; then
        printf "  %-10s  %11ss\n" "$n" "$lambda_time"
    else
        printf "  %-10s  %11ss  %12s\n" "$n" "$sp1_time" "$sp1_cycles"
    fi
done

echo ""
if $RUN_LAMBDA && $RUN_SP1; then
    echo -e "Green ratio = Lambda VM faster, Red = SP1 faster"
fi
echo "Raw data in $TMP_DIR/"

if [ -n "$LAMBDA_PROJECTED_S" ] || [ -n "$SP1_PROJECTED_S" ]; then
    echo ""
    echo -e "${BOLD}=== Linear Projection to 500M Steps ===${NC}"
    if [ -n "$LAMBDA_PROJECTED_S" ]; then
        echo "  Lambda VM: ${LAMBDA_PROJECTED_S}s (${LAMBDA_PROJECTED_H}h), R²=${LAMBDA_R2}"
    fi
    if [ -n "$SP1_PROJECTED_S" ]; then
        echo "  SP1 v6:    ${SP1_PROJECTED_S}s (${SP1_PROJECTED_H}h), R²=${SP1_R2}"
    fi
fi

# --- Machine-readable report ------------------------------------------------

if [ -n "$REPORT_DIR" ]; then
    {
        echo "target_steps=$TARGET_STEPS"
        echo "series=$(join_slash "${RESULT_N[@]}")"
        echo "lambda_times=$(join_slash "${RESULT_LAMBDA[@]}")"
        echo "sp1_times=$(join_slash "${RESULT_SP1[@]}")"
        echo "sp1_cycles=$(join_slash "${RESULT_SP1_CYCLES[@]}")"
        echo "ratios=$(join_slash "${RESULT_RATIO[@]}")"
        if [ -n "$LAMBDA_PROJECTED_S" ]; then
            echo "lambda_slope_s_per_1m=$LAMBDA_SLOPE"
            echo "lambda_intercept_s=$LAMBDA_INTERCEPT"
            echo "lambda_r2=$LAMBDA_R2"
            echo "lambda_projected_time_s=$LAMBDA_PROJECTED_S"
            echo "lambda_projected_time_h=$LAMBDA_PROJECTED_H"
        fi
        if [ -n "$SP1_PROJECTED_S" ]; then
            echo "sp1_slope_s_per_1m=$SP1_SLOPE"
            echo "sp1_intercept_s=$SP1_INTERCEPT"
            echo "sp1_r2=$SP1_R2"
            echo "sp1_projected_time_s=$SP1_PROJECTED_S"
            echo "sp1_projected_time_h=$SP1_PROJECTED_H"
        fi
    } > "$REPORT_DIR/metrics.txt"

    {
        echo "# Lambda VM vs SP1 v6 Benchmark"
        echo
        echo "| n | Lambda VM (s) | SP1 v6 (s) | SP1 cycles | Ratio |"
        echo "|--:|--------------:|-----------:|-----------:|------:|"
        for i in "${!RESULT_N[@]}"; do
            printf "| %s | %s | %s | %s | %s |\n" \
                "${RESULT_N[$i]}" \
                "${RESULT_LAMBDA[$i]}" \
                "${RESULT_SP1[$i]}" \
                "${RESULT_SP1_CYCLES[$i]}" \
                "${RESULT_RATIO[$i]}"
        done
        echo
        echo "## Linear Projection to 500M Steps"
        echo
        echo "| Prover | Slope (s / 1M steps) | Intercept (s) | R² | Projected @ 500M (s) | Projected @ 500M (h) |"
        echo "|--------|----------------------:|--------------:|---:|---------------------:|---------------------:|"
        if [ -n "$LAMBDA_PROJECTED_S" ]; then
            printf "| Lambda VM | %s | %s | %s | %s | %s |\n" \
                "$LAMBDA_SLOPE" \
                "$LAMBDA_INTERCEPT" \
                "$LAMBDA_R2" \
                "$LAMBDA_PROJECTED_S" \
                "$LAMBDA_PROJECTED_H"
        fi
        if [ -n "$SP1_PROJECTED_S" ]; then
            printf "| SP1 v6 | %s | %s | %s | %s | %s |\n" \
                "$SP1_SLOPE" \
                "$SP1_INTERCEPT" \
                "$SP1_R2" \
                "$SP1_PROJECTED_S" \
                "$SP1_PROJECTED_H"
        fi
    } > "$REPORT_DIR/summary.md"
fi
