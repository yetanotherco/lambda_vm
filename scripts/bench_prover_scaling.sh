#!/bin/bash
# Prover scaling benchmark across fib_iterative sizes.
# Shows how per-phase timing or heap grows with program size, plus linear regression.
#
# Usage: bench_prover_scaling.sh <time|heap> [--sizes "500k 1M 2M 4M"] [--runs N]
#
# time mode: builds with `instruments` only (clean timings, no heap data).
# heap mode: builds with `instruments,jemalloc-stats` (timings + per-phase heap;
#            jemalloc-stats adds a few percent of timing overhead).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_prover_scaling"
ELF_DIR="$ROOT_DIR/executor/program_artifacts/asm"

GREEN='\033[0;32m'
BOLD='\033[1m'
NC='\033[0m'

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <time|heap> [--sizes \"...\"] [--runs N]" >&2
    exit 1
fi
MODE=$1; shift
case $MODE in
    time) FEATURES="instruments" ;;
    heap) FEATURES="instruments,jemalloc-stats" ;;
    *) echo "Unknown mode: $MODE (expected time|heap)" >&2; exit 1 ;;
esac

SIZES="500k 1M 2M 4M"
RUNS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --sizes) SIZES="$2"; shift 2 ;;
        --runs)  RUNS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

suffix_to_steps() {
    case $1 in
        160k) echo 160000 ;; 250k) echo 250000 ;; 372k) echo 372000 ;;
        500k) echo 500000 ;; 1M) echo 1000000 ;; 1200k) echo 1200000 ;;
        2M) echo 2000000 ;; 4M) echo 4000000 ;; 8M) echo 8000000 ;;
        16M) echo 16000000 ;; 32M) echo 32000000 ;; 64M) echo 64000000 ;; 128M) echo 128000000 ;;
        *) echo "Unknown size: $1" >&2; exit 1 ;;
    esac
}

rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"

echo -e "${GREEN}Building CLI with features: ${FEATURES}...${NC}"
cargo build --release -p cli --features "$FEATURES" \
    --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1
CLI="$ROOT_DIR/target/release/cli"

# Parse timing (seconds) and heap (MB) from one run's stdout + stderr.
# Emits key=value lines to stdout.
parse_run() {
    local stdout=$1 stderr=$2
    awk '
    function secs(    s) {
        if (match($0, /[0-9]+\.?[0-9]*s/)) {
            s = substr($0, RSTART, RLENGTH - 1)
            return s
        }
        return ""
    }
    /^  Execute /            { v = secs(); if (v) print "t_execute="     v }
    /^  Trace build/         { v = secs(); if (v) print "t_trace_build=" v }
    /^  AIR construction/    { v = secs(); if (v) print "t_air="         v }
    /^  Pre-pass/            { v = secs(); if (v) print "t_prepass="     v }
    /^  Round 1 /            { v = secs(); if (v) print "t_round1="      v }
    /Main trace commits/     { v = secs(); if (v) print "t_main_commits="v }
    /Rounds 2/               { v = secs(); if (v) print "t_rounds24="    v }
    /Main expand_columns_to_lde/{ v = secs(); if (v) print "t_main_lde=" v }
    /Aux expand_columns_to_lde/ { v = secs(); if (v) print "t_aux_lde="  v }
    /Main commit \(Merkle\)/ { v = secs(); if (v) print "t_main_merkle=" v }
    /Aux commit \(Merkle\)/  { v = secs(); if (v) print "t_aux_merkle="  v }
    /^  Total FFT/           { v = secs(); if (v) print "t_total_fft="   v }
    /^  Total Merkle/        { v = secs(); if (v) print "t_total_merkle="v }
    /^  TOTAL /              { v = secs(); if (v) print "t_total="       v }
    /After execute/          { print "h_execute="      $(NF-1) }
    /After trace build/      { print "h_trace_build="  $(NF-1) }
    /After AIR/              { print "h_air="          $(NF-1) }
    /After pool alloc/       { print "h_pool_alloc="   $(NF-1) }
    /After main commits/     { print "h_main_commits=" $(NF-1) }
    # No "After aux build"/"After aux commit" rows: aux build and aux commit are
    # fused into the per-table scheduler, so with k tables in flight there is no
    # single moment at which either has finished, and the prover no longer takes
    # those snapshots. "Aux trace build"/"Aux trace commit" timing rows are gone
    # for the same reason. "After main commits" and "Peak heap" still bracket
    # the fused region.
    ' "$stderr"

    grep -o 'Peak heap: [0-9]*' "$stdout" | awk '{print "peak=" $3}'
    grep -o 'Proving time: [0-9.]*' "$stdout" | awk '{print "wall=" $3}'
}

get_val() {
    awk -F= -v k="$2" '$1==k {print $2; exit}' "$1" 2>/dev/null
}

# Pick the run whose wall-clock is the median across RUNS; copy its data file.
# Writes final per-size data to ${size}_data.txt.
select_median_run() {
    local size=$1
    if [ "$RUNS" -eq 1 ]; then
        cp "$TMP_DIR/${size}_run_1.txt" "$TMP_DIR/${size}_data.txt"
        return
    fi
    # Collect (wall, run_index), sort by wall, pick middle
    local pairs=""
    for i in $(seq 1 "$RUNS"); do
        local w
        w=$(get_val "$TMP_DIR/${size}_run_${i}.txt" wall)
        pairs+=" $w $i"
    done
    local median_idx
    median_idx=$(echo "$pairs" | tr ' ' '\n' | awk 'NF' | paste -d' ' - - | \
        sort -n | awk -v n="$RUNS" 'NR == int((n+1)/2) {print $2}')
    cp "$TMP_DIR/${size}_run_${median_idx}.txt" "$TMP_DIR/${size}_data.txt"
}

for size in $SIZES; do
    ELF="$ELF_DIR/fib_iterative_${size}.elf"
    [ -f "$ELF" ] || { echo "Missing: $ELF"; continue; }
    steps=$(suffix_to_steps "$size")
    echo -e "${GREEN}Running fib_iterative_${size} (${RUNS} runs)...${NC}"

    for i in $(seq 1 "$RUNS"); do
        STDOUT="$TMP_DIR/${size}_run_${i}_stdout.txt"
        STDERR="$TMP_DIR/${size}_run_${i}_stderr.txt"
        "$CLI" prove "$ELF" -o "$TMP_DIR/proof.bin" --time >"$STDOUT" 2>"$STDERR"
        rm -f "$TMP_DIR/proof.bin"
        {
            echo "steps=$steps"
            parse_run "$STDOUT" "$STDERR"
        } > "$TMP_DIR/${size}_run_${i}.txt"
    done
    select_median_run "$size"
done

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

print_row() {
    local label=$1 key=$2 unit=$3
    printf "  %-26s" "$label"
    for size in $SIZES; do
        local data="$TMP_DIR/${size}_data.txt"
        if [ ! -f "$data" ]; then
            printf " %10s" "-"; continue
        fi
        local v
        v=$(get_val "$data" "$key")
        if [ -z "$v" ]; then
            printf " %10s" "-"
        elif [ "$unit" = "s" ]; then
            printf " %9.2fs" "$v"
        else
            printf " %10d" "$v"
        fi
    done
    echo ""
}

print_header() {
    printf "  %-26s" ""
    for size in $SIZES; do printf " %10s" "$size"; done
    echo ""
    printf "  %-26s" ""
    for size in $SIZES; do printf " %10s" "──────────"; done
    echo ""
}

echo ""
echo -e "${BOLD}=== TIMING (seconds) ===${NC}"
print_header
print_row "Execute"                t_execute     s
print_row "Trace build"            t_trace_build s
print_row "AIR construction"       t_air         s
print_row "Pre-pass"               t_prepass     s
print_row "Round 1"                t_round1      s
print_row "  Main trace commits"   t_main_commits s
print_row "    Main LDE"           t_main_lde    s
print_row "    Main Merkle"        t_main_merkle s
print_row "    Aux LDE"            t_aux_lde     s
print_row "    Aux Merkle"         t_aux_merkle  s
print_row "Rounds 2-4"             t_rounds24    s
print_row "Total FFT (all rounds)" t_total_fft   s
print_row "Total Merkle"           t_total_merkle s
print_row "TOTAL"                  t_total       s

if [[ "$MODE" == "heap" ]]; then
    echo ""
    echo -e "${BOLD}=== HEAP (MB absolute) ===${NC}"
    print_header
    print_row "After execute"          h_execute      mb
    print_row "After trace build"      h_trace_build  mb
    print_row "After AIR construction" h_air          mb
    print_row "After pool alloc"       h_pool_alloc   mb
    print_row "After main commits"     h_main_commits mb
    print_row "Peak heap"              peak           mb
fi

# ---------------------------------------------------------------------------
# Linear regression per metric: y = a + b * (steps / 1M)
# ---------------------------------------------------------------------------

regress() {
    local label=$1 key=$2 unit=$3
    local pairs=""
    for size in $SIZES; do
        local data="$TMP_DIR/${size}_data.txt"
        [ -f "$data" ] || continue
        local steps v
        steps=$(get_val "$data" steps)
        v=$(get_val "$data" "$key")
        [ -z "$v" ] && continue
        [ -z "$steps" ] && continue
        local steps_m
        steps_m=$(awk -v s="$steps" 'BEGIN {printf "%.6f", s / 1000000}')
        pairs+=" $steps_m $v"
    done
    echo "$pairs" | awk -v label="$label" -v unit="$unit" '{
        n = NF / 2
        if (n < 2) { printf "  %-26s  (insufficient data)\n", label; next }
        for (i = 0; i < n; i++) { x[i] = $(2*i+1); y[i] = $(2*i+2) }
        sx=0; sy=0; sxx=0; sxy=0
        for (i=0;i<n;i++) { sx+=x[i]; sy+=y[i]; sxx+=x[i]*x[i]; sxy+=x[i]*y[i] }
        d = n*sxx - sx*sx
        if (d == 0) { printf "  %-26s  (constant)\n", label; next }
        b = (n*sxy - sx*sy) / d
        a = (sy - b*sx) / n
        ym = sy/n; ss_tot=0; ss_res=0
        for (i=0;i<n;i++) { p=a+b*x[i]; ss_res+=(y[i]-p)^2; ss_tot+=(y[i]-ym)^2 }
        r2 = (ss_tot>0) ? 1 - ss_res/ss_tot : 1
        if (unit == "s")
            printf "  %-26s  %+7.2fs/M  (base %6.2fs, R\xc2\xb2=%.3f)\n", label, b, a, r2
        else
            printf "  %-26s  %+7.0f MB/M  (base %6.0f MB, R\xc2\xb2=%.3f)\n", label, b, a, r2
    }'
}

echo ""
echo -e "${BOLD}=== TIMING GROWTH (per 1M steps) ===${NC}"
regress "Execute"          t_execute     s
regress "Trace build"      t_trace_build s
regress "AIR construction" t_air         s
regress "Pre-pass"         t_prepass     s
regress "Round 1"          t_round1      s
regress "Rounds 2-4"       t_rounds24    s
regress "Total FFT"        t_total_fft   s
regress "Total Merkle"     t_total_merkle s
regress "TOTAL"            t_total       s

if [[ "$MODE" == "heap" ]]; then
    echo ""
    echo -e "${BOLD}=== HEAP GROWTH (per 1M steps) ===${NC}"
    regress "After execute"          h_execute      mb
    regress "After trace build"      h_trace_build  mb
    regress "After AIR construction" h_air          mb
    regress "After pool alloc"       h_pool_alloc   mb
    regress "After main commits"     h_main_commits mb
    regress "Peak heap"              peak           mb
fi

echo ""
echo "Raw data: $TMP_DIR/"
