#!/bin/bash
set -euo pipefail

WEBHOOK_URL="$1"

if [ -z "$WEBHOOK_URL" ]; then
    echo "SLACK_WEBHOOK not configured, skipping notification."
    exit 0
fi

METRICS_FILE="bench_vs_artifacts/metrics.txt"

if [ ! -f "$METRICS_FILE" ]; then
    curl -X POST "$WEBHOOK_URL" \
        -H 'Content-Type: application/json; charset=utf-8' \
        --data '{"blocks":[{"type":"header","text":{"type":"plain_text","text":"Lambda VM vs SP1 v6 - Nightly Benchmark"}},{"type":"section","text":{"type":"mrkdwn","text":":x: Benchmark failed - no metrics found. Check the workflow logs."}}]}'
    exit 0
fi

parse_metric() {
    { grep "^${1}=" "$METRICS_FILE" || true; } | cut -d= -f2-
}

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

PROJ_SECTION=""
if [ -n "$LAMBDA_PROJECTED_H" ] || [ -n "$SP1_PROJECTED_H" ]; then
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

ETHREX_METRICS_FILE="bench_vs_artifacts/ethrex_metrics.txt"
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

curl -X POST "$WEBHOOK_URL" \
    -H 'Content-Type: application/json; charset=utf-8' \
    --data '{"blocks":[{"type":"header","text":{"type":"plain_text","text":"Lambda VM vs SP1 v6 - Nightly Benchmark"}},{"type":"context","elements":[{"type":"mrkdwn","text":"*Program:* Fibonacci  ·  *Device:* CPU"}]},{"type":"divider"},{"type":"section","text":{"type":"mrkdwn","text":"'"$RESULTS_MRKDWN"'"}}'"$PROJ_SECTION$ETHREX_SECTION"']}'
