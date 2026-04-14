#!/bin/bash
# Per-phase heap profile across program sizes.
# Shows where heap grows and how each phase scales with program size.
#
# Usage: bench_heap_profile.sh [--no-build] [--programs "500k 1M 2M 4M"]
#
# Requires: instruments + jemalloc-stats features.
# Peak heap is deterministic, so 1 run per size is enough.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_heap_profile"
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

rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"

if $BUILD; then
    echo -e "${GREEN}Building CLI with instruments + jemalloc-stats...${NC}"
    cargo build --release -p cli --features jemalloc-stats,instruments \
        --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -1
fi
CLI="$ROOT_DIR/target/release/cli"

# Phase labels we parse from stderr (order matters)
PHASES="execute trace_build air pool_alloc main_commits aux_build aux_commit"

for size in $PROGRAMS; do
    ELF="$ELF_DIR/fib_iterative_${size}.elf"
    [ -f "$ELF" ] || { echo "Missing: $ELF"; continue; }
    steps=$(suffix_to_steps "$size")
    echo -e "${GREEN}Running fib_iterative_${size}...${NC}"

    STDERR="$TMP_DIR/${size}_stderr.txt"
    STDOUT="$TMP_DIR/${size}_stdout.txt"
    "$CLI" prove "$ELF" -o "$TMP_DIR/proof.bin" --time >"$STDOUT" 2>"$STDERR"
    rm -f "$TMP_DIR/proof.bin"

    # Parse absolute heap values (second-to-last column) from HEAP PROFILE section
    HEAP_VALS=$(awk '
        /After execute/          { printf "execute=%s\n",      $(NF-1) }
        /After trace build/      { printf "trace_build=%s\n",  $(NF-1) }
        /After AIR/              { printf "air=%s\n",          $(NF-1) }
        /after pool alloc/       { printf "pool_alloc=%s\n",   $(NF-1) }
        /after main commits/     { printf "main_commits=%s\n", $(NF-1) }
        /after aux build/        { printf "aux_build=%s\n",    $(NF-1) }
        /after aux commit/       { printf "aux_commit=%s\n",   $(NF-1) }
    ' "$STDERR")

    PEAK=$(grep -o 'Peak heap: [0-9]*' "$STDOUT" | awk '{print $3}')
    echo "steps=$steps" > "$TMP_DIR/${size}_data.txt"
    echo "peak=$PEAK" >> "$TMP_DIR/${size}_data.txt"
    echo "$HEAP_VALS" >> "$TMP_DIR/${size}_data.txt"
done

echo ""
echo -e "${BOLD}=== HEAP PROFILE ACROSS SIZES ===${NC}"
echo ""

# Print table: phases as rows, sizes as columns
# Header
printf "  %-22s" "Phase (delta MB)"
for size in $PROGRAMS; do printf " %10s" "$size"; done
echo ""
printf "  %-22s" "──────────────────────"
for size in $PROGRAMS; do printf " %10s" "──────────"; done
echo ""

# For each phase, print the delta
prev_phase=""
for phase in $PHASES; do
    case $phase in
        execute)      label="Execute" ;;
        trace_build)  label="Trace build" ;;
        air)          label="AIR construction" ;;
        pool_alloc)   label="Pool allocation" ;;
        main_commits) label="Main commits" ;;
        aux_build)    label="Aux build" ;;
        aux_commit)   label="Aux commit" ;;
    esac

    printf "  %-22s" "$label"
    for size in $PROGRAMS; do
        DATA="$TMP_DIR/${size}_data.txt"
        [ -f "$DATA" ] || { printf " %10s" "N/A"; continue; }
        cur=$(grep "^${phase}=" "$DATA" | cut -d= -f2)
        if [ -z "$cur" ]; then
            printf " %10s" "N/A"
        else
            # Get previous phase value to compute delta
            if [ -z "$prev_phase" ]; then
                delta="$cur"
            else
                prev_val=$(grep "^${prev_phase}=" "$DATA" | cut -d= -f2)
                delta=$((cur - prev_val))
            fi
            printf " %+10d" "$delta"
        fi
    done
    echo ""
    prev_phase=$phase
done

# Total/peak row
printf "  %-22s" "──────────────────────"
for size in $PROGRAMS; do printf " %10s" "──────────"; done
echo ""
printf "  %-22s" "Peak heap"
for size in $PROGRAMS; do
    DATA="$TMP_DIR/${size}_data.txt"
    peak=$(grep "^peak=" "$DATA" | cut -d= -f2)
    printf " %10s" "${peak:-N/A}"
done
echo ""

# Linear regression per phase
echo ""
echo -e "${BOLD}=== GROWTH RATE PER PHASE (MB per 1M steps) ===${NC}"
echo ""

for phase in $PHASES; do
    case $phase in
        execute)      label="Execute" ;;
        trace_build)  label="Trace build" ;;
        air)          label="AIR construction" ;;
        pool_alloc)   label="Pool allocation" ;;
        main_commits) label="Main commits" ;;
        aux_build)    label="Aux build" ;;
        aux_commit)   label="Aux commit" ;;
    esac

    # Collect (steps_M, delta) pairs
    PAIRS=""
    prev_phase_key=""
    case $phase in
        execute)      prev_phase_key="" ;;
        trace_build)  prev_phase_key="execute" ;;
        air)          prev_phase_key="trace_build" ;;
        pool_alloc)   prev_phase_key="air" ;;
        main_commits) prev_phase_key="pool_alloc" ;;
        aux_build)    prev_phase_key="main_commits" ;;
        aux_commit)   prev_phase_key="aux_build" ;;
    esac

    for size in $PROGRAMS; do
        DATA="$TMP_DIR/${size}_data.txt"
        [ -f "$DATA" ] || continue
        steps=$(grep "^steps=" "$DATA" | cut -d= -f2)
        cur=$(grep "^${phase}=" "$DATA" | cut -d= -f2)
        [ -z "$cur" ] && continue
        if [ -z "$prev_phase_key" ]; then
            delta="$cur"
        else
            prev_val=$(grep "^${prev_phase_key}=" "$DATA" | cut -d= -f2)
            delta=$((cur - prev_val))
        fi
        steps_m=$(awk "BEGIN {printf \"%.2f\", $steps / 1000000}")
        PAIRS="$PAIRS $steps_m $delta"
    done

    # Linear regression: delta = a + b * steps_M
    echo "$PAIRS" | awk -v label="$label" '{
        n = NF / 2
        if (n < 2) { printf "  %-22s  (insufficient data)\n", label; next }
        for (i = 0; i < n; i++) {
            x[i] = $(2*i+1); y[i] = $(2*i+2)
        }
        sx=0; sy=0; sxx=0; sxy=0
        for (i=0;i<n;i++) { sx+=x[i]; sy+=y[i]; sxx+=x[i]*x[i]; sxy+=x[i]*y[i] }
        b = (n*sxy - sx*sy) / (n*sxx - sx*sx)
        a = (sy - b*sx) / n
        # R²
        ym = sy/n; ss_tot=0; ss_res=0
        for (i=0;i<n;i++) { p=a+b*x[i]; ss_res+=(y[i]-p)^2; ss_tot+=(y[i]-ym)^2 }
        r2 = (ss_tot>0) ? 1 - ss_res/ss_tot : 1
        printf "  %-22s  %+.0f MB/M steps  (base: %.0f MB, R²=%.3f)\n", label, b, a, r2
    }'
done

# Extrapolation
echo ""
echo -e "${BOLD}=== EXTRAPOLATED PEAK HEAP ===${NC}"
echo ""

# Collect (steps_M, peak) for regression
PEAK_PAIRS=""
for size in $PROGRAMS; do
    DATA="$TMP_DIR/${size}_data.txt"
    [ -f "$DATA" ] || continue
    steps=$(grep "^steps=" "$DATA" | cut -d= -f2)
    peak=$(grep "^peak=" "$DATA" | cut -d= -f2)
    [ -z "$peak" ] && continue
    steps_m=$(awk "BEGIN {printf \"%.2f\", $steps / 1000000}")
    PEAK_PAIRS="$PEAK_PAIRS $steps_m $peak"
done

echo "$PEAK_PAIRS" | awk '{
    n = NF / 2
    if (n < 2) { print "  (insufficient data)"; next }
    for (i = 0; i < n; i++) { x[i] = $(2*i+1); y[i] = $(2*i+2) }
    sx=0; sy=0; sxx=0; sxy=0
    for (i=0;i<n;i++) { sx+=x[i]; sy+=y[i]; sxx+=x[i]*x[i]; sxy+=x[i]*y[i] }
    b = (n*sxy - sx*sy) / (n*sxx - sx*sx)
    a = (sy - b*sx) / n
    ym = sy/n; ss_tot=0; ss_res=0
    for (i=0;i<n;i++) { p=a+b*x[i]; ss_res+=(y[i]-p)^2; ss_tot+=(y[i]-ym)^2 }
    r2 = (ss_tot>0) ? 1 - ss_res/ss_tot : 1
    printf "  Model: peak = %.0f + %.0f * steps_M  (R²=%.4f)\n\n", a, b, r2
    targets[0]=8; targets[1]=16; targets[2]=32; targets[3]=64
    labels[0]="8M"; labels[1]="16M"; labels[2]="32M"; labels[3]="64M"
    for (t=0; t<4; t++) {
        pred = a + b * targets[t]
        printf "  fib_iterative_%-6s  ~%.0f MB  (~%.0f GB)\n", labels[t], pred, pred/1024
    }
}'

echo ""
echo "Raw data: $TMP_DIR/"
