#!/bin/bash
set -euo pipefail

WEBHOOK_URL="$1"

if [ -z "$WEBHOOK_URL" ]; then
    echo "SLACK_WEBHOOK not configured, skipping notification."
    exit 0
fi

# All overridable so the GPU nightly can reuse this publisher; the defaults reproduce the
# CPU nightly's message byte for byte.
ARTIFACT_DIR="${BENCH_ARTIFACT_DIR:-bench_vs_artifacts}"
SLACK_HEADER="${BENCH_SLACK_HEADER:-Lambda VM vs SP1 v6 - Nightly Benchmark}"
DEVICE_LABEL="${BENCH_SLACK_DEVICE:-CPU}"
PROGRAM_LABEL="${BENCH_SLACK_PROGRAM:-Fibonacci}"
# Set both to render the ethrex GPU-vs-CPU section (same workload, same host, two builds).
GPU_DIR="${BENCH_GPU_DIR:-}"
CPU_DIR="${BENCH_CPU_DIR:-}"
# Suppress the extrapolated-projection block. Set it whenever the run measures only one
# prover: a lone "projected at 500M cycles" number gets read against the other nightly's
# number, which was fitted over different sizes on a different machine. The fit itself is
# still in the artifact for whoever wants it.
NO_PROJECTION="${BENCH_SLACK_NO_PROJECTION:-}"

METRICS_FILE="$ARTIFACT_DIR/metrics.txt"

HAVE_FIB=false
[ -f "$METRICS_FILE" ] && HAVE_FIB=true
HAVE_GPU=false
[ -n "$GPU_DIR" ] && [ -f "$GPU_DIR/ethrex_metrics.txt" ] && HAVE_GPU=true

if ! $HAVE_FIB && ! $HAVE_GPU; then
    curl -X POST "$WEBHOOK_URL" \
        -H 'Content-Type: application/json; charset=utf-8' \
        --data '{"blocks":[{"type":"header","text":{"type":"plain_text","text":"'"$SLACK_HEADER"'"}},{"type":"section","text":{"type":"mrkdwn","text":":x: Benchmark failed - no metrics found. Check the workflow logs."}}]}'
    exit 0
fi

parse_metric() {
    { grep "^${1}=" "$METRICS_FILE" || true; } | cut -d= -f2-
}

# Read a key from an arbitrary metrics file: parse_from <file> <key>
parse_from() {
    { grep "^${2}=" "$1" 2>/dev/null || true; } | cut -d= -f2-
}

FIB_SECTION=""
PROJ_SECTION=""
if $HAVE_FIB; then
TARGET_STEPS_SERIES=$(parse_metric "target_steps_series")
LAMBDA_TIMES=$(parse_metric "lambda_times")
SP1_TIMES=$(parse_metric "sp1_times")
RATIOS=$(parse_metric "ratios")
TARGET_CYCLES=$(parse_metric "target_cycles")
TARGET_CYCLES="${TARGET_CYCLES:-500000000}"
LAMBDA_PROJECTED_H=$(parse_metric "lambda_projected_time_h")
SP1_PROJECTED_H=$(parse_metric "sp1_projected_time_h")
LAMBDA_R2=$(parse_metric "lambda_r2")
SP1_R2=$(parse_metric "sp1_r2")

IFS='/' read -ra STEPS_ARR <<< "${TARGET_STEPS_SERIES:-}"
IFS='/' read -ra LAMBDA_ARR <<< "${LAMBDA_TIMES:-}"
IFS='/' read -ra SP1_ARR <<< "${SP1_TIMES:-}"
IFS='/' read -ra RATIO_ARR <<< "${RATIOS:-}"

RESULTS_MRKDWN=""
for i in "${!STEPS_ARR[@]}"; do
    steps="${STEPS_ARR[$i]}"
    [ -z "$steps" ] && continue
    lambda_t="${LAMBDA_ARR[$i]:-n/a}"
    sp1_t="${SP1_ARR[$i]:-n/a}"
    ratio="${RATIO_ARR[$i]:-n/a}"
    steps_m=$(LC_NUMERIC=C awk -v s="$steps" 'BEGIN { printf "%dM", s/1000000 }')
    if [ "$ratio" != "n/a" ]; then
        ratio_fmt=$(LC_NUMERIC=C awk -v r="$ratio" 'BEGIN { printf "%.2fx", r }')
        line="*${steps_m} steps:* Lambda ${lambda_t}s / SP1 ${sp1_t}s - ratio ${ratio_fmt}"
    else
        line="*${steps_m} steps:* Lambda ${lambda_t}s / SP1 ${sp1_t}s"
    fi
    if [ -n "$RESULTS_MRKDWN" ]; then
        RESULTS_MRKDWN="${RESULTS_MRKDWN}\\n${line}"
    else
        RESULTS_MRKDWN="$line"
    fi
done

if [ -z "$RESULTS_MRKDWN" ]; then
    RESULTS_MRKDWN="(no data)"
fi
FIB_SECTION=',{"type":"divider"},{"type":"section","text":{"type":"mrkdwn","text":"'"$RESULTS_MRKDWN"'"}}'

if [ -z "$NO_PROJECTION" ] && { [ -n "$LAMBDA_PROJECTED_H" ] || [ -n "$SP1_PROJECTED_H" ]; }; then
    TARGET_M=$(LC_NUMERIC=C awk -v c="$TARGET_CYCLES" 'BEGIN { printf "%dM", c/1000000 }')
    PROJ_MRKDWN="Projected @ ${TARGET_M} cycles:"
    if [ -n "$LAMBDA_PROJECTED_H" ]; then
        line="*Lambda VM:* ${LAMBDA_PROJECTED_H}h"
        [ -n "$LAMBDA_R2" ] && line="$line (R2=${LAMBDA_R2})"
        PROJ_MRKDWN="${PROJ_MRKDWN}\\n${line}"
    fi
    if [ -n "$SP1_PROJECTED_H" ]; then
        line="*SP1 v6:* ${SP1_PROJECTED_H}h"
        [ -n "$SP1_R2" ] && line="$line (R2=${SP1_R2})"
        PROJ_MRKDWN="${PROJ_MRKDWN}\\n${line}"
    fi
    PROJ_SECTION=',{"type":"divider"},{"type":"header","text":{"type":"plain_text","text":"Linear Projection"}},{"type":"section","text":{"type":"mrkdwn","text":"'"$PROJ_MRKDWN"'"}}'
fi
fi

ETHREX_METRICS_FILE="$ARTIFACT_DIR/ethrex_metrics.txt"
ETHREX_SECTION=""
if [ -f "$ETHREX_METRICS_FILE" ]; then
    # Render one "*<label>:* <time>s (<cycles> cycles)" line per block.
    ethrex_line() {
        local label=$1 key=$2 t c
        t=$(grep "^${key}_time_s=" "$ETHREX_METRICS_FILE" | cut -d= -f2-)
        c=$(grep "^${key}_cycles=" "$ETHREX_METRICS_FILE" | cut -d= -f2-)
        [ -z "$t" ] && return 0
        local line="*${label}:* ${t}s"
        if [ -n "$c" ] && [ "$c" != "n/a" ]; then
            line="${line} (${c} cycles)"
        fi
        printf '%s' "$line"
    }
    EMPTY_LINE=$(ethrex_line "Empty block" "ethrex_empty_block")
    TX_LINE=$(ethrex_line "1 tx" "ethrex_1_tx")
    ETHREX_MRKDWN=""
    [ -n "$EMPTY_LINE" ] && ETHREX_MRKDWN="$EMPTY_LINE"
    [ -n "$TX_LINE" ] && ETHREX_MRKDWN="${ETHREX_MRKDWN:+$ETHREX_MRKDWN\n}$TX_LINE"
    if [ -n "$ETHREX_MRKDWN" ]; then
        ETHREX_SECTION=',{"type":"divider"},{"type":"header","text":{"type":"plain_text","text":"Lambda VM - Ethrex"}},{"type":"section","text":{"type":"mrkdwn","text":"'"$ETHREX_MRKDWN"'"}}'
    fi
fi

GPU_SECTION=""
if $HAVE_GPU; then
    GM="$GPU_DIR/ethrex_metrics.txt"
    CM="$CPU_DIR/ethrex_metrics.txt"
    # One block per slug present on the GPU side, with the CPU/GPU speedup. Verification
    # never touches CUDA, so its ratio is a control: it should sit near 1.00, and a drift
    # means the host was noisy and the prove speedup from this run is suspect.
    GPU_MRKDWN=""
    add_line() {
        if [ -n "$GPU_MRKDWN" ]; then GPU_MRKDWN="${GPU_MRKDWN}\\n${1}"; else GPU_MRKDWN="$1"; fi
    }
    ratio() {  # $1=cpu $2=gpu -> "N.NNx" or empty
        [ -z "$1" ] || [ -z "$2" ] && return 0
        [ "$1" = "n/a" ] || [ "$2" = "n/a" ] && return 0
        LC_NUMERIC=C awk -v c="$1" -v g="$2" 'BEGIN { if (g+0 > 0) printf "%.2fx", c/g }'
    }
    for slug in $(grep -oE '^[a-z0-9_]+_time_s=' "$GM" | sed 's/_time_s=$//'); do
        gp=$(parse_from "$GM" "${slug}_time_s"); cp=$(parse_from "$CM" "${slug}_time_s")
        gv=$(parse_from "$GM" "${slug}_verify_s"); cv=$(parse_from "$CM" "${slug}_verify_s")
        cyc=$(parse_from "$GM" "${slug}_cycles"); ep=$(parse_from "$GM" "${slug}_epochs")
        heap=$(parse_from "$GM" "${slug}_peak_heap_mb")
        pbytes=$(parse_from "$GM" "${slug}_proof_bytes")
        [ -z "$gp" ] && continue
        label=$(printf '%s' "$slug" | tr '_' ' ')
        sp=$(ratio "$cp" "$gp")
        line="*${label}* prove: GPU ${gp}s / CPU ${cp:-n/a}s"
        [ -n "$sp" ] && line="${line} - *${sp}*"
        add_line "$line"
        vr=$(ratio "$cv" "$gv")
        vline="   verify (CPU path, control): ${gv:-n/a}s / ${cv:-n/a}s"
        [ -n "$vr" ] && vline="${vline} - ratio ${vr}"
        add_line "$vline"
        det="   ${cyc:-n/a} cycles"
        [ -n "$ep" ] && [ "$ep" != "n/a" ] && det="${det} · ${ep} epochs"
        [ -n "$heap" ] && [ "$heap" != "n/a" ] && det="${det} · ${heap} MB peak heap"
        [ -n "$pbytes" ] && det="${det} · $(LC_NUMERIC=C awk -v b="$pbytes" 'BEGIN { printf "%.2f GiB proof", b/1073741824 }')"
        add_line "$det"
    done
    if [ -n "$GPU_MRKDWN" ]; then
        add_line " "
        add_line "_Absolute times carry ±10-20% host noise across nightly rentals — trust the ratio._"
        GPU_SECTION=',{"type":"divider"},{"type":"header","text":{"type":"plain_text","text":"Ethrex continuations - GPU vs CPU (same host)"}},{"type":"section","text":{"type":"mrkdwn","text":"'"$GPU_MRKDWN"'"}}'
    fi
fi

curl -X POST "$WEBHOOK_URL" \
    -H 'Content-Type: application/json; charset=utf-8' \
    --data '{"blocks":[{"type":"header","text":{"type":"plain_text","text":"'"$SLACK_HEADER"'"}},{"type":"context","elements":[{"type":"mrkdwn","text":"*Program:* '"$PROGRAM_LABEL"'  ·  *Device:* '"$DEVICE_LABEL"'"}]}'"$FIB_SECTION$PROJ_SECTION$ETHREX_SECTION$GPU_SECTION"']}'
