#!/bin/bash
# Per-phase timing profile across program sizes.
# Shows how each proving phase and sub-operation scales with program size.
#
# Usage: bench_timing_profile.sh [--no-build] [--programs "1M 2M 4M 8M"]
#
# Requires: instruments feature.
# Timing is deterministic with instruments, so 1 run per size is enough.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
ELF_DIR="$ROOT_DIR/executor/program_artifacts/asm"

GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

BUILD=true
PROGRAMS="500k 1M 2M 4M"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build) BUILD=false; shift ;;
        --programs) PROGRAMS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

suffix_to_steps() {
    case $1 in
        160k) echo 160000 ;; 250k) echo 250000 ;; 372k) echo 372000 ;;
        500k) echo 500000 ;; 1M) echo 1000000 ;; 1200k) echo 1200000 ;;
        2M) echo 2000000 ;; 4M) echo 4000000 ;; 8M) echo 8000000 ;;
        16M) echo 16000000 ;; 32M) echo 32000000 ;; 64M) echo 64000000 ;; 128M) echo 128000000 ;;
        *) echo "Unknown: $1" >&2; exit 1 ;;
    esac
}


if $BUILD; then
    echo -e "${GREEN}Building CLI with instruments...${NC}"
    cargo build --release -p cli --features instruments \
        --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1
fi
CLI="$ROOT_DIR/target/release/cli"

# Parse instruments stderr into key=value pairs.
parse_timing() {
    awk '
    function secs(    s) {
        if (match($0, /[0-9]+\.?[0-9]*s/)) {
            s = substr($0, RSTART, RLENGTH - 1)
            return s
        }
        return ""
    }
    /^  Execute / { v = secs(); if (v) print "execute=" v }
    /^  Trace build/ { v = secs(); if (v) print "trace_build=" v }
    /^  AIR construction/ { v = secs(); if (v) print "air=" v }
    /^  Pre-pass/ { v = secs(); if (v) print "prepass=" v }
    /^  Round 1 / { v = secs(); if (v) print "round1=" v }
    /Main trace commits/ { v = secs(); if (v) print "main_commits=" v }
    /Aux trace build/ { v = secs(); if (v) print "aux_build=" v }
    /Aux trace commit/ { v = secs(); if (v) print "aux_commit=" v }
    /Rounds 2/ { v = secs(); if (v) print "rounds24=" v }
    /Main expand_(pool|columns)_to_lde/ { v = secs(); if (v) print "main_lde=" v }
    /Aux expand_(pool|columns)_to_lde/  { v = secs(); if (v) print "aux_lde=" v }
    /Main commit \(Merkle\)/            { v = secs(); if (v) print "main_merkle=" v }
    /Aux commit \(Merkle\)/             { v = secs(); if (v) print "aux_merkle=" v }
    /R1  expand_(pool|columns)_to_lde/ { v = secs(); if (v) print "r1_lde=" v }
    /R2  evaluate/           { v = secs(); if (v) print "r2_evaluate=" v }
    /R2  decompose/          { v = secs(); if (v) print "r2_decompose=" v }
    /R2  commit_comp/        { v = secs(); if (v) print "r2_commit_comp=" v }
    /R3  OOD/                { v = secs(); if (v) print "r3_ood=" v }
    /R4  deep_comp/          { v = secs(); if (v) print "r4_deep_comp=" v }
    /R4  interpolate/        { v = secs(); if (v) print "r4_interp=" v }
    /R4  fri::commit/        { v = secs(); if (v) print "r4_fri_commit=" v }
    /R4  queries/            { v = secs(); if (v) print "r4_queries=" v }
    /^  Total FFT/    { v = secs(); if (v) print "total_fft=" v }
    /^  Total Merkle/ { v = secs(); if (v) print "total_merkle=" v }
    /^  TOTAL /       { v = secs(); if (v) print "total=" v }
    ' "$1"
}

for size in $PROGRAMS; do
    ELF="$ELF_DIR/fib_iterative_${size}.elf"
    [ -f "$ELF" ] || { echo "Missing: $ELF"; continue; }
    steps=$(suffix_to_steps "$size")
    echo -e "${GREEN}Running fib_iterative_${size}...${NC}"

    STDERR="$TMP_DIR/${size}_stderr.txt"
    "$CLI" prove "$ELF" -o "$TMP_DIR/proof.bin" 2>"$STDERR" >/dev/null
    rm -f "$TMP_DIR/proof.bin"

    echo "steps=$steps" > "$TMP_DIR/${size}_data.txt"
    parse_timing "$STDERR" >> "$TMP_DIR/${size}_data.txt"
done

# --- Display ---------------------------------------------------------------

get_val() {
    grep "^${2}=" "$1" 2>/dev/null | cut -d= -f2
}

print_section() {
    local title=$1; shift

    echo ""
    echo -e "  ${BOLD}${title}${NC}"
    printf "  %-30s" ""
    for size in $PROGRAMS; do printf " %9s" "$size"; done
    echo ""
    printf "  %-30s" ""
    for size in $PROGRAMS; do printf " %9s" "─────────"; done
    echo ""

    for spec in "$@"; do
        local label="${spec%%:*}"
        local key="${spec#*:}"
        printf "  %-30s" "$label"
        for size in $PROGRAMS; do
            local DATA="$TMP_DIR/${size}_data.txt"
            local val
            val=$(get_val "$DATA" "$key")
            if [ -n "$val" ]; then
                printf " %8.2fs" "$val"
            else
                printf " %9s" "-"
            fi
        done
        echo ""
    done
}

echo ""
echo -e "${BOLD}=== TIMING PROFILE ACROSS SIZES ===${NC}"

TABLE_K="${TABLE_PARALLELISM:-default (cores/3)}"
echo -e "  TABLE_PARALLELISM=$TABLE_K"

print_section "Phase (wall time)" \
    "TOTAL:total" \
    "Execute:execute" \
    "Trace build:trace_build" \
    "AIR construction:air" \
    "Pre-pass:prepass" \
    "Round 1:round1" \
    "Rounds 2-4:rounds24"

print_section "Round 1 breakdown (CPU sum)" \
    "Main LDE:main_lde" \
    "Main Merkle:main_merkle" \
    "Aux build (wall):aux_build" \
    "Aux LDE:aux_lde" \
    "Aux Merkle:aux_merkle"

print_section "Rounds 2-4 sub-ops (CPU sum)" \
    "R1 LDE (reconstruct):r1_lde" \
    "R2 decompose+extend:r2_decompose" \
    "R2 evaluate:r2_evaluate" \
    "R2 commit comp poly:r2_commit_comp" \
    "R3 OOD evaluation:r3_ood" \
    "R4 deep comp evals:r4_deep_comp" \
    "R4 interpolate+evaluate:r4_interp" \
    "R4 FRI commit:r4_fri_commit" \
    "R4 queries & openings:r4_queries"

print_section "Cross-round totals (CPU sum)" \
    "Total FFT:total_fft" \
    "Total Merkle:total_merkle"

# --- Growth rate -----------------------------------------------------------

echo ""
echo -e "${BOLD}=== GROWTH RATE (per 1M steps) ===${NC}"
echo ""

for spec in \
    "TOTAL:total" \
    "Trace build:trace_build" \
    "Round 1:round1" \
    "Rounds 2-4:rounds24" \
    "Main LDE:main_lde" \
    "Aux LDE:aux_lde" \
    "R1 LDE (reconstruct):r1_lde" \
    "R2 decompose+extend:r2_decompose" \
    "R3 OOD evaluation:r3_ood" \
    "Total FFT:total_fft" \
    "Total Merkle:total_merkle"
do
    label="${spec%%:*}"
    key="${spec#*:}"

    PAIRS=""
    for size in $PROGRAMS; do
        DATA="$TMP_DIR/${size}_data.txt"
        [ -f "$DATA" ] || continue
        steps=$(get_val "$DATA" "steps")
        val=$(get_val "$DATA" "$key")
        [ -z "$val" ] && continue
        [ -z "$steps" ] && continue
        steps_m=$(awk -v s="$steps" 'BEGIN {printf "%.6f", s / 1000000}')
        PAIRS="$PAIRS $steps_m $val"
    done

    echo "$PAIRS" | awk -v label="$label" '{
        n = NF / 2
        if (n < 2) { printf "  %-30s  (insufficient data)\n", label; next }
        for (i = 0; i < n; i++) {
            x[i] = $(2*i+1); y[i] = $(2*i+2)
        }
        sx=0; sy=0; sxx=0; sxy=0
        for (i=0;i<n;i++) { sx+=x[i]; sy+=y[i]; sxx+=x[i]*x[i]; sxy+=x[i]*y[i] }
        denom = n*sxx - sx*sx
        if (denom == 0) { printf "  %-30s  (constant)\n", label; next }
        b = (n*sxy - sx*sy) / denom
        a = (sy - b*sx) / n
        ym = sy/n; ss_tot=0; ss_res=0
        for (i=0;i<n;i++) { p=a+b*x[i]; ss_res+=(y[i]-p)^2; ss_tot+=(y[i]-ym)^2 }
        r2 = (ss_tot>0) ? 1 - ss_res/ss_tot : 1
        printf "  %-30s  %+.2fs/M steps  (base: %.2fs, R\xc2\xb2=%.3f)\n", label, b, a, r2
    }'
done

echo ""
