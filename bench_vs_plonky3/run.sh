#!/bin/bash
# Benchmark: Lambda STARK vs Plonky3 — single-shot prove time on the shared
# Fibonacci AIR (columns = 2 * num_sequences, blowup = 2, fri_queries = 219).
#
# Usage:
#   ./bench_vs_plonky3/run.sh [--log-rows K ...] [--num-sequences N] [--runs N]
#                             [--lambda-only | --p3-only] [--report-dir DIR]
#                             [--scalar] [--breakdown] [--no-color]
#
# Defaults: --log-rows 19, --num-sequences 16, --runs 10.
# With multiple --log-rows values, prints one stats row per size.
#
# --scalar: on x86_64 drops AVX2 / AVX-512 so Goldilocks runs scalar. The MMCS
# itself is already scalar (single-input tiny_keccak via Keccak256Hash) regardless
# of this flag — its SIMD lanes were removed in the config. Triggers a rebuild
# when toggling; subsequent runs with the same RUSTFLAGS are cached.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="$(mktemp -d -t bench_p3.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
REPORT_DIR=""
NO_COLOR=false
SCALAR=false
BREAKDOWN=false

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

LOG_ROWS=()
NUM_SEQUENCES=16
RUNS=10
BLOWUP=2
FRI_QUERIES=219
GRINDING=0
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
        --scalar)
            SCALAR=true
            shift
            ;;
        --breakdown)
            BREAKDOWN=true
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

if [ -n "$REPORT_DIR" ]; then
    mkdir -p "$REPORT_DIR/raw"
fi

# --- Scalar (no SIMD) toggle ------------------------------------------------
# When --scalar is on, disable AVX2/AVX-512 so Goldilocks field arithmetic runs
# scalar for an apples-to-apples comparison against Lambda STARK. The MMCS Keccak
# is already scalar regardless of this flag (see plonky3_config.rs).
# Cargo caches per-RUSTFLAGS, so toggling scalar vs vector triggers a rebuild
# on first use but is cached afterwards.
SCALAR_RUSTFLAGS=""
SCALAR_ACTIVE=false
if $SCALAR; then
    case "$(uname -m)" in
        x86_64|amd64)
            SCALAR_RUSTFLAGS="-C target-feature=-avx2,-avx512f"
            SCALAR_ACTIVE=true
            ;;
        *)
            echo "warning: --scalar: only supported on x86_64; host is $(uname -m), not pinning RUSTFLAGS" >&2
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
echo -e "  runs/size:      ${YELLOW}${RUNS}${NC}  (median + CV reported)"
echo -e "  p3 extension:   ${YELLOW}upstream CubicTrinomialExtensionField (x^3 - x - 1)${NC}"
echo -e "  p3 mmcs:        ${YELLOW}scalar Keccak256 (val_packing_width=1, hash_lanes=1)${NC}"
echo -e "  proof params:   ${YELLOW}blowup=${BLOWUP}, queries=${FRI_QUERIES}, grinding=${GRINDING}${NC}"
if $BREAKDOWN; then
    echo -e "  breakdown:      ${YELLOW}on${NC}  (Lambda instruments + P3 tracing spans)"
else
    echo -e "  breakdown:      ${YELLOW}off${NC}"
fi
if $SCALAR_ACTIVE; then
    echo -e "  scalar mode:    ${YELLOW}on${NC}  (arch=$(uname -m), RUSTFLAGS=\"${RUSTFLAGS:-}\")"
elif $SCALAR; then
    echo -e "  scalar mode:    ${YELLOW}requested (unsupported on $(uname -m))${NC}  (SIMD enabled, compiler default)"
else
    echo -e "  scalar mode:    ${YELLOW}off${NC}  (SIMD enabled, compiler default)"
fi
echo ""

echo -e "${GREEN}[build]${NC} prove_bench"
BUILD_ARGS=(build --release -p bench-vs-plonky3 --bin prove_bench --manifest-path "$ROOT_DIR/Cargo.toml")
if $BREAKDOWN; then
    BUILD_ARGS+=(--features instruments)
fi
cargo "${BUILD_ARGS[@]}" 2>&1 | tail -5

# Resolve the actual target directory via cargo metadata so we find the binary
# whether cargo used ./target/ (default) or a custom CARGO_TARGET_DIR.
TARGET_DIR=$(cargo metadata --manifest-path "$ROOT_DIR/Cargo.toml" \
    --format-version 1 --no-deps 2>/dev/null \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])' \
    2>/dev/null || echo "$ROOT_DIR/target")
BIN="$TARGET_DIR/release/prove_bench"
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

extract_metrics_line() {
    sed -n '/^METRICS	/ {
        p
        q
    }'
}

extract_audit_line() {
    sed -n '/^AUDIT	/ {
        p
        q
    }'
}

metric_value() {
    local line=$1
    local key=$2
    printf '%s\n' "$line" | tr '\t' '\n' | LC_ALL=C awk -F= -v key="$key" '$1 == key { print $2; exit }'
}

median_of() {
    # prints median of the given numeric arguments.
    # Uses shell `sort -g` for portability (macOS awk lacks gawk's asort).
    printf '%s\n' "$@" | LC_ALL=C sort -g | LC_NUMERIC=C awk '
        { a[NR] = $0 + 0 }
        END {
            if (NR == 0) { print "n/a"; exit }
            if (NR % 2 == 1) {
                printf "%.6f\n", a[(NR + 1) / 2]
            } else {
                printf "%.6f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2
            }
        }'
}

ratio_fmt() {
    LC_NUMERIC=C awk -v num="$1" -v den="$2" 'BEGIN {
        if (den + 0 == 0) { print "n/a"; exit }
        printf "%.3f\n", num / den
    }'
}

median_file() {
    LC_ALL=C sort -g "$1" | LC_NUMERIC=C awk '
        { a[NR] = $0 + 0 }
        END {
            if (NR == 0) { print "n/a"; exit }
            if (NR % 2 == 1) printf "%.6f\n", a[(NR + 1) / 2]
            else printf "%.6f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2
        }'
}

cv_pct_file() {
    LC_NUMERIC=C awk '
        { s += $1; ss += $1 * $1; n++ }
        END {
            if (n == 0) { print "n/a"; exit }
            if (n < 2) { print "n/a"; exit }
            m = s / n
            v = (ss - n * m * m) / (n - 1)
            if (v < 0) v = 0
            sd = sqrt(v)
            if (m == 0) print "n/a"
            else printf "%.2f\n", sd * 100 / m
        }' "$1"
}

fmt0() {
    LC_NUMERIC=C awk -v v="$1" 'BEGIN { if (v == "n/a") print v; else printf "%.0f\n", v }'
}

metric_file_for() {
    local metrics_file=$1
    local key=$2
    local out_file=$3
    : > "$out_file"
    while IFS= read -r line; do
        local value
        value=$(metric_value "$line" "$key")
        if [ -n "$value" ] && [ "$value" != "n/a" ]; then
            printf '%s\n' "$value" >> "$out_file"
        fi
    done < "$metrics_file"
}

median_metric() {
    local prover=$1
    local log_rows=$2
    local key=$3
    local file="$TMP_DIR/${prover}_${log_rows}_${key}.values"
    metric_file_for "$TMP_DIR/${prover}_${log_rows}.metrics" "$key" "$file"
    if [ ! -s "$file" ]; then
        printf "n/a\n"
    else
        median_file "$file"
    fi
}

# --- Run benchmark ----------------------------------------------------------

RESULT_LOG_ROWS=()
RESULT_ROWS=()
RESULT_LAMBDA=()
RESULT_P3=()
RESULT_RATIO=()
RESULT_LAMBDA_CV=()
RESULT_P3_CV=()
RESULT_LAMBDA_VERIFY=()
RESULT_P3_VERIFY=()
RESULT_LAMBDA_PROOF_SIZE=()
RESULT_P3_PROOF_SIZE=()
RESULT_LAMBDA_RSS=()
RESULT_P3_RSS=()

run_prover() {
    local prover=$1   # lambda | p3
    local log_rows=$2
    local times=()
    local metrics_file="$TMP_DIR/${prover}_${log_rows}.metrics"
    local audit_file="$TMP_DIR/${prover}_${log_rows}.audits"
    local breakdown_file="$TMP_DIR/${prover}_${log_rows}.breakdown"
    : > "$metrics_file"
    : > "$audit_file"
    : > "$breakdown_file"
    for run_i in $(seq 1 "$RUNS"); do
        local out_file="$TMP_DIR/${prover}_${log_rows}_${run_i}.stdout"
        local run_args=(--prover "$prover" --log-rows "$log_rows" --num-sequences "$NUM_SEQUENCES" --blowup "$BLOWUP" --queries "$FRI_QUERIES" --grinding "$GRINDING")
        if $BREAKDOWN; then
            run_args+=(--breakdown)
        fi
        if ! "$BIN" "${run_args[@]}" > "$out_file" 2>&1; then
            echo -e "  ${RED}[${prover}] FAILED on log-rows=${log_rows} run ${run_i}${NC}" >&2
            cat "$out_file" >&2
            exit 1
        fi
        local audit_line
        audit_line=$(extract_audit_line < "$out_file")
        if [ -n "$audit_line" ]; then
            printf 'run=%s\t%s\n' "$run_i" "$audit_line" >> "$audit_file"
        fi
        local metrics_line
        metrics_line=$(extract_metrics_line < "$out_file")
        if [ -z "$metrics_line" ]; then
            echo -e "  ${RED}[${prover}] could not parse metrics (log-rows=${log_rows}, run ${run_i})${NC}" >&2
            cat "$out_file" >&2
            exit 1
        fi
        printf '%s\n' "$metrics_line" >> "$metrics_file"
        if $BREAKDOWN; then
            sed -n "s/^BREAKDOWN	/BREAKDOWN	run=${run_i}	/p" "$out_file" >> "$breakdown_file"
        fi

        local t
        t=$(metric_value "$metrics_line" prove_s)
        if [ -z "$t" ]; then
            t=$(extract_proving_time < "$out_file")
        fi
        times+=("$t")
        if [ -n "$REPORT_DIR" ]; then
            cp "$out_file" "$REPORT_DIR/raw/${prover}_log${log_rows}_run${run_i}.stdout"
        fi
    done
    printf '%s\n' "${times[@]}" > "$TMP_DIR/${prover}_${log_rows}.times"
    median_of "${times[@]}"
}

for lr in "${LOG_ROWS[@]}"; do
    rows=$((1 << lr))
    echo -e "${BOLD}--- log-rows=${lr}  (rows = ${rows}) ---${NC}"

    lambda_median="n/a"
    p3_median="n/a"
    lambda_cv="n/a"
    p3_cv="n/a"
    lambda_verify="n/a"
    p3_verify="n/a"
    lambda_proof_size="n/a"
    p3_proof_size="n/a"
    lambda_rss="n/a"
    p3_rss="n/a"

    if $RUN_LAMBDA; then
        echo -ne "  ${GREEN}[lambda]${NC} "
        lambda_median=$(run_prover lambda "$lr")
        lambda_cv=$(cv_pct_file "$TMP_DIR/lambda_${lr}.times")
        lambda_verify=$(median_metric lambda "$lr" verify_s)
        lambda_proof_size=$(median_metric lambda "$lr" proof_size_bytes)
        lambda_rss=$(median_metric lambda "$lr" peak_rss_kb)
        echo -e "prove median ${BOLD}${lambda_median}s${NC} (CV ${lambda_cv}%), verify ${lambda_verify}s, proof $(fmt0 "$lambda_proof_size") B, rss $(fmt0 "$lambda_rss") KB"
    fi

    if $RUN_P3; then
        echo -ne "  ${GREEN}[p3]${NC}     "
        p3_median=$(run_prover p3 "$lr")
        p3_cv=$(cv_pct_file "$TMP_DIR/p3_${lr}.times")
        p3_verify=$(median_metric p3 "$lr" verify_s)
        p3_proof_size=$(median_metric p3 "$lr" proof_size_bytes)
        p3_rss=$(median_metric p3 "$lr" peak_rss_kb)
        echo -e "prove median ${BOLD}${p3_median}s${NC} (CV ${p3_cv}%), verify ${p3_verify}s, proof $(fmt0 "$p3_proof_size") B, rss $(fmt0 "$p3_rss") KB"
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
    RESULT_LAMBDA_CV+=("$lambda_cv")
    RESULT_P3_CV+=("$p3_cv")
    RESULT_LAMBDA_VERIFY+=("$lambda_verify")
    RESULT_P3_VERIFY+=("$p3_verify")
    RESULT_LAMBDA_PROOF_SIZE+=("$lambda_proof_size")
    RESULT_P3_PROOF_SIZE+=("$p3_proof_size")
    RESULT_LAMBDA_RSS+=("$lambda_rss")
    RESULT_P3_RSS+=("$p3_rss")
done

# --- Summary table ----------------------------------------------------------

echo ""
echo -e "${BOLD}=== Summary ===${NC}"
if $RUN_LAMBDA && $RUN_P3; then
    printf "  %-9s  %-12s  %14s  %9s  %14s  %9s  %10s\n" "log-rows" "rows" "Lambda (s)" "L CV%" "P3 (s)" "P3 CV%" "L/P3"
    printf "  %-9s  %-12s  %14s  %9s  %14s  %9s  %10s\n" "--------" "----" "----------" "-----" "------" "------" "----"
else
    printf "  %-9s  %-12s  %14s  %9s\n" "log-rows" "rows" "Time (s)" "CV%"
    printf "  %-9s  %-12s  %14s  %9s\n" "--------" "----" "--------" "---"
fi

for i in "${!RESULT_LOG_ROWS[@]}"; do
    lr="${RESULT_LOG_ROWS[$i]}"
    rows="${RESULT_ROWS[$i]}"
    lt="${RESULT_LAMBDA[$i]}"
    pt="${RESULT_P3[$i]}"
    rt="${RESULT_RATIO[$i]}"
    lcv="${RESULT_LAMBDA_CV[$i]}"
    pcv="${RESULT_P3_CV[$i]}"
    if $RUN_LAMBDA && $RUN_P3; then
        color=$GREEN
        verdict="Lambda faster"
        if awk -v l="$lt" -v p="$pt" 'BEGIN{ exit !(l+0 > p+0) }'; then
            color=$RED
            verdict="P3 faster"
        fi
        printf "  %-9s  %-12s  %13ss  %8s%%  %13ss  %8s%%  ${color}%9sx${NC}  (${color}%s${NC})\n" \
            "$lr" "$rows" "$lt" "$lcv" "$pt" "$pcv" "$rt" "$verdict"
    elif $RUN_LAMBDA; then
        printf "  %-9s  %-12s  %13ss  %8s%%\n" "$lr" "$rows" "$lt" "$lcv"
    else
        printf "  %-9s  %-12s  %13ss  %8s%%\n" "$lr" "$rows" "$pt" "$pcv"
    fi
done

echo ""
if $RUN_LAMBDA && $RUN_P3; then
    echo -e "Timing window: prove only for the ratio. Verify, proof size, RSS and throughput are reported separately."
fi

# --- Machine-readable report ------------------------------------------------

if [ -n "$REPORT_DIR" ]; then
    # Slash-joined helpers for metrics.txt (mirrors the format used by
    # bench_vs/run.sh).
    join_slash() {
        local joined=""
        for value in "$@"; do
            joined="${joined:+$joined/}$value"
        done
        printf "%s\n" "$joined"
    }

    {
        printf "log_rows\trows\tlambda_prove_median_s\tlambda_prove_cv_pct\tlambda_verify_median_s\tlambda_proof_size_bytes_median\tlambda_peak_rss_kb_median\tp3_prove_median_s\tp3_prove_cv_pct\tp3_verify_median_s\tp3_proof_size_bytes_median\tp3_peak_rss_kb_median\tratio_lambda_over_p3\truns\n"
        for i in "${!RESULT_LOG_ROWS[@]}"; do
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                "${RESULT_LOG_ROWS[$i]}" \
                "${RESULT_ROWS[$i]}" \
                "${RESULT_LAMBDA[$i]}" \
                "${RESULT_LAMBDA_CV[$i]}" \
                "${RESULT_LAMBDA_VERIFY[$i]}" \
                "${RESULT_LAMBDA_PROOF_SIZE[$i]}" \
                "${RESULT_LAMBDA_RSS[$i]}" \
                "${RESULT_P3[$i]}" \
                "${RESULT_P3_CV[$i]}" \
                "${RESULT_P3_VERIFY[$i]}" \
                "${RESULT_P3_PROOF_SIZE[$i]}" \
                "${RESULT_P3_RSS[$i]}" \
                "${RESULT_RATIO[$i]}" \
                "$RUNS"
        done
    } > "$REPORT_DIR/results.tsv"

    {
        printf "workload\tprover\tlog_rows\trows\tnum_sequences\tmain_cols\taux_cols\ttables\tlogup\tblowup\tfri_queries\tgrinding\tprove_s\tverify_s\tproof_size_bytes\tpeak_rss_kb\trows_per_sec\tcells_per_sec\n"
        for lr in "${RESULT_LOG_ROWS[@]}"; do
            for prover in lambda p3; do
                metrics_file="$TMP_DIR/${prover}_${lr}.metrics"
                if [ ! -f "$metrics_file" ]; then
                    continue
                fi
                while IFS= read -r line; do
                    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                        "$(metric_value "$line" workload)" \
                        "$(metric_value "$line" prover)" \
                        "$(metric_value "$line" log_rows)" \
                        "$(metric_value "$line" rows)" \
                        "$(metric_value "$line" num_sequences)" \
                        "$(metric_value "$line" main_cols)" \
                        "$(metric_value "$line" aux_cols)" \
                        "$(metric_value "$line" tables)" \
                        "$(metric_value "$line" logup)" \
                        "$(metric_value "$line" blowup)" \
                        "$(metric_value "$line" fri_queries)" \
                        "$(metric_value "$line" grinding)" \
                        "$(metric_value "$line" prove_s)" \
                        "$(metric_value "$line" verify_s)" \
                        "$(metric_value "$line" proof_size_bytes)" \
                        "$(metric_value "$line" peak_rss_kb)" \
                        "$(metric_value "$line" rows_per_sec)" \
                        "$(metric_value "$line" cells_per_sec)"
                done < "$metrics_file"
            done
        done
    } > "$REPORT_DIR/raw_metrics.tsv"

    # Raw AUDIT lines per run, one row per prover×log_rows×run. Lets the reader
    # confirm in retrospect that val_packing_width=1, hash_lanes=1, etc.
    {
        printf "run\taudit_line\n"
        for lr in "${RESULT_LOG_ROWS[@]}"; do
            for prover in lambda p3; do
                audit_file="$TMP_DIR/${prover}_${lr}.audits"
                if [ -f "$audit_file" ]; then
                    cat "$audit_file"
                fi
            done
        done
    } > "$REPORT_DIR/raw_audits.tsv"

    if $BREAKDOWN; then
        {
            printf "run\tworkload\tprover\tlog_rows\trows\tphase\tms\ttable\ttable_rows\tspan\n"
            for lr in "${RESULT_LOG_ROWS[@]}"; do
                for prover in lambda p3; do
                    breakdown_file="$TMP_DIR/${prover}_${lr}.breakdown"
                    if [ ! -f "$breakdown_file" ]; then
                        continue
                    fi
                    while IFS= read -r line; do
                        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
                            "$(metric_value "$line" run)" \
                            "$(metric_value "$line" workload)" \
                            "$(metric_value "$line" prover)" \
                            "$(metric_value "$line" log_rows)" \
                            "$(metric_value "$line" rows)" \
                            "$(metric_value "$line" phase)" \
                            "$(metric_value "$line" ms)" \
                            "$(metric_value "$line" table)" \
                            "$(metric_value "$line" table_rows)" \
                            "$(metric_value "$line" span)"
                    done < "$breakdown_file"
                done
            done
        } > "$REPORT_DIR/breakdown.tsv"
    fi

    # Capture commit + timestamp so the artifact is self-describing.
    git_sha="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    git_dirty="clean"
    if ! git -C "$ROOT_DIR" diff --quiet HEAD -- 2>/dev/null; then
        git_dirty="dirty"
    fi
    timestamp_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    {
        echo "timestamp_utc=$timestamp_utc"
        echo "git_sha=$git_sha"
        echo "git_tree=$git_dirty"
        echo "arch=$(uname -m)"
        echo "num_sequences=$NUM_SEQUENCES"
        echo "columns=$((2 * NUM_SEQUENCES))"
        echo "blowup=$BLOWUP"
        echo "fri_queries=$FRI_QUERIES"
        echo "grinding=$GRINDING"
        echo "runs_per_size=$RUNS"
        if $BREAKDOWN; then
            echo "breakdown=on"
        else
            echo "breakdown=off"
        fi
        echo "p3_extension=upstream_cubic_trinomial"
        echo "p3_mmcs=scalar_keccak256"
        if $SCALAR_ACTIVE; then
            echo "scalar=on"
            echo "rustflags=$SCALAR_RUSTFLAGS"
        elif $SCALAR; then
            echo "scalar=requested_unsupported"
        else
            echo "scalar=off"
        fi
        echo "timing_window=prove_only_ratio_verify_size_rss_reported_separately"
        echo "log_rows_series=$(join_slash "${RESULT_LOG_ROWS[@]}")"
        echo "rows_series=$(join_slash "${RESULT_ROWS[@]}")"
        echo "lambda_prove_medians=$(join_slash "${RESULT_LAMBDA[@]}")"
        echo "p3_prove_medians=$(join_slash "${RESULT_P3[@]}")"
        echo "lambda_verify_medians=$(join_slash "${RESULT_LAMBDA_VERIFY[@]}")"
        echo "p3_verify_medians=$(join_slash "${RESULT_P3_VERIFY[@]}")"
        echo "lambda_proof_size_medians=$(join_slash "${RESULT_LAMBDA_PROOF_SIZE[@]}")"
        echo "p3_proof_size_medians=$(join_slash "${RESULT_P3_PROOF_SIZE[@]}")"
        echo "lambda_peak_rss_medians=$(join_slash "${RESULT_LAMBDA_RSS[@]}")"
        echo "p3_peak_rss_medians=$(join_slash "${RESULT_P3_RSS[@]}")"
        echo "ratios_lambda_over_p3=$(join_slash "${RESULT_RATIO[@]}")"
    } > "$REPORT_DIR/metrics.txt"
fi
