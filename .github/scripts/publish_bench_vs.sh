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

# --- Plonky3 section (optional) --------------------------------------------
# Built when `bench_vs_artifacts/p3/headline/metrics.txt` exists.

p3_parse() {
    local file=$1
    local key=$2
    { grep "^${key}=" "$file" 2>/dev/null || true; } | cut -d= -f2-
}

p3_fmt_seconds() {
    LC_NUMERIC=C awk -v s="$1" 'BEGIN {
        if (s == "") { print "n/a"; exit }
        if (s + 0 < 1) printf "%.0fms", s * 1000
        else printf "%.3fs", s
    }'
}

p3_fmt_mb() {
    LC_NUMERIC=C awk -v b="$1" 'BEGIN {
        if (b == "") { print "n/a"; exit }
        printf "%.1f MB", b / (1024 * 1024)
    }'
}

p3_fmt_gb() {
    LC_NUMERIC=C awk -v kb="$1" 'BEGIN {
        if (kb == "") { print "n/a"; exit }
        printf "%.2f GB", kb / (1024 * 1024)
    }'
}

p3_fmt_ratio_pair() {
    LC_NUMERIC=C awk -v a="$1" -v b="$2" 'BEGIN {
        if (a == "" || b == "" || b + 0 == 0) { print "n/a"; exit }
        printf "%.2fx", a / b
    }'
}

P3_SECTION=""
P3_HEADLINE_FILE="bench_vs_artifacts/p3/headline/metrics.txt"
if [ -f "$P3_HEADLINE_FILE" ]; then
    H_LOG_ROWS=$(p3_parse "$P3_HEADLINE_FILE" "log_rows_series")
    H_COLS=$(p3_parse "$P3_HEADLINE_FILE" "columns")
    H_BLOWUP=$(p3_parse "$P3_HEADLINE_FILE" "blowup")
    H_QUERIES=$(p3_parse "$P3_HEADLINE_FILE" "fri_queries")
    H_ROWS=$(p3_parse "$P3_HEADLINE_FILE" "rows_series")
    H_LAMBDA_PROVE=$(p3_parse "$P3_HEADLINE_FILE" "lambda_prove_medians")
    H_P3_PROVE=$(p3_parse "$P3_HEADLINE_FILE" "p3_prove_medians")
    H_LAMBDA_VERIFY=$(p3_parse "$P3_HEADLINE_FILE" "lambda_verify_medians")
    H_P3_VERIFY=$(p3_parse "$P3_HEADLINE_FILE" "p3_verify_medians")
    H_LAMBDA_PROOF=$(p3_parse "$P3_HEADLINE_FILE" "lambda_proof_size_medians")
    H_P3_PROOF=$(p3_parse "$P3_HEADLINE_FILE" "p3_proof_size_medians")
    H_LAMBDA_RSS=$(p3_parse "$P3_HEADLINE_FILE" "lambda_peak_rss_medians")
    H_P3_RSS=$(p3_parse "$P3_HEADLINE_FILE" "p3_peak_rss_medians")
    H_RATIO=$(p3_parse "$P3_HEADLINE_FILE" "ratios_lambda_over_p3")

    H_ROWS_FMT=$(LC_NUMERIC=C awk -v r="$H_ROWS" 'BEGIN {
        if (r == "") { print "n/a"; exit }
        if (r + 0 >= 1000000) printf "%.1fM", r / 1000000
        else if (r + 0 >= 1000) printf "%.0fK", r / 1000
        else printf "%d", r
    }')

    PROOF_RATIO=$(p3_fmt_ratio_pair "$H_LAMBDA_PROOF" "$H_P3_PROOF")
    RSS_RATIO=$(p3_fmt_ratio_pair "$H_LAMBDA_RSS" "$H_P3_RSS")
    PROVE_RATIO_FMT=$(LC_NUMERIC=C awk -v r="$H_RATIO" 'BEGIN {
        if (r == "" || r == "n/a") { print "n/a"; exit }
        printf "%.2fx", r
    }')

    P3_HEADLINE_MRKDWN="*log_rows=${H_LOG_ROWS} (${H_ROWS_FMT} rows · ${H_COLS} cols · blowup=${H_BLOWUP} · ${H_QUERIES} queries)*"
    P3_HEADLINE_MRKDWN="${P3_HEADLINE_MRKDWN}\\n*Lambda:* $(p3_fmt_seconds "$H_LAMBDA_PROVE") prove · $(p3_fmt_seconds "$H_LAMBDA_VERIFY") verify · $(p3_fmt_mb "$H_LAMBDA_PROOF") proof · $(p3_fmt_gb "$H_LAMBDA_RSS") RSS"
    P3_HEADLINE_MRKDWN="${P3_HEADLINE_MRKDWN}\\n*Plonky3:* $(p3_fmt_seconds "$H_P3_PROVE") prove · $(p3_fmt_seconds "$H_P3_VERIFY") verify · $(p3_fmt_mb "$H_P3_PROOF") proof · $(p3_fmt_gb "$H_P3_RSS") RSS"
    P3_HEADLINE_MRKDWN="${P3_HEADLINE_MRKDWN}\\n*Ratio L/P3:* ${PROVE_RATIO_FMT} prove · ${PROOF_RATIO} proof · ${RSS_RATIO} RSS"

    P3_SECTION=',{"type":"divider"},{"type":"header","text":{"type":"plain_text","text":"Lambda VM vs Plonky3 - Headline"}},{"type":"section","text":{"type":"mrkdwn","text":"'"$P3_HEADLINE_MRKDWN"'"}}'
fi

curl -X POST "$WEBHOOK_URL" \
    -H 'Content-Type: application/json; charset=utf-8' \
    --data '{"blocks":[{"type":"header","text":{"type":"plain_text","text":"Lambda VM vs SP1 v6 - Nightly Benchmark"}},{"type":"divider"},{"type":"section","text":{"type":"mrkdwn","text":"'"$RESULTS_MRKDWN"'"}}'"$PROJ_SECTION$P3_SECTION"']}'
