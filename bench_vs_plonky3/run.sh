#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="$(mktemp -d -t bench_p3.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

LOG_ROWS=()
NUM_SEQUENCES=16
RUNS=10
BLOWUP=2
FRI_QUERIES=219
GRINDING=0
RUN_LAMBDA=true
RUN_P3=true
REPORT_DIR=""
NO_SIMD=true
BREAKDOWN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-rows)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LOG_ROWS+=("$1")
                shift
            done
            ;;
        --num-sequences)
            NUM_SEQUENCES="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --blowup)
            BLOWUP="$2"
            shift 2
            ;;
        --queries)
            FRI_QUERIES="$2"
            shift 2
            ;;
        --grinding)
            GRINDING="$2"
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
            REPORT_DIR="$2"
            shift 2
            ;;
        --native-simd)
            NO_SIMD=false
            shift
            ;;
        --no-simd|--scalar)
            NO_SIMD=true
            shift
            ;;
        --breakdown)
            BREAKDOWN=true
            shift
            ;;
        -h|--help)
            sed -n '1,80p' "$0" | sed -n 's/^# //p'
            exit 0
            ;;
        *)
            echo "unknown option: $1" >&2
            exit 2
            ;;
    esac
done

if [ ${#LOG_ROWS[@]} -eq 0 ]; then
    LOG_ROWS=(19)
fi

if [ "$RUNS" -lt 1 ]; then
    echo "--runs must be >= 1" >&2
    exit 2
fi

if [ -n "$REPORT_DIR" ]; then
    mkdir -p "$REPORT_DIR/raw"
fi

if $NO_SIMD; then
    case "$(uname -m)" in
        x86_64|amd64)
            SIMD_FLAGS="-C target-feature=-avx2,-avx512f"
            export RUSTFLAGS="${RUSTFLAGS:-} ${SIMD_FLAGS}"
            ;;
        *)
            echo "warning: --no-simd cannot force Plonky3 Goldilocks scalar on $(uname -m) without patching Plonky3" >&2
            ;;
    esac
fi

echo "=== Lambda STARK vs Plonky3 Fibonacci ==="
echo "log_rows: ${LOG_ROWS[*]}"
echo "num_sequences: $NUM_SEQUENCES (main_cols=$((2 * NUM_SEQUENCES)))"
echo "runs: $RUNS"
echo "params: blowup=$BLOWUP queries=$FRI_QUERIES grinding=$GRINDING"
if $NO_SIMD; then
    echo "simd: requested off (RUSTFLAGS='${RUSTFLAGS:-}')"
else
    echo "simd: native compiler defaults"
fi
echo

CARGO_FEATURES=""
if $BREAKDOWN; then
    CARGO_FEATURES="--features instruments"
fi
# shellcheck disable=SC2086
cargo build --release -p bench-vs-plonky3 --bin prove_bench --manifest-path "$ROOT_DIR/bench_vs_plonky3/Cargo.toml" $CARGO_FEATURES

TARGET_DIR="$(cargo metadata --manifest-path "$ROOT_DIR/bench_vs_plonky3/Cargo.toml" --format-version 1 --no-deps \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')"
BIN="$TARGET_DIR/release/prove_bench"

metric_value() {
    local line="$1"
    local key="$2"
    printf '%s\n' "$line" | tr '\t' '\n' | awk -F= -v key="$key" '$1 == key { print $2; exit }'
}

median_file() {
    LC_ALL=C sort -g "$1" | awk '
        { a[NR] = $0 + 0 }
        END {
            if (NR == 0) { print "n/a"; exit }
            if (NR % 2 == 1) printf "%.6f\n", a[(NR + 1) / 2]
            else printf "%.6f\n", (a[NR / 2] + a[NR / 2 + 1]) / 2
        }'
}

cv_pct_file() {
    awk '
        { s += $1; ss += $1 * $1; n++ }
        END {
            if (n == 0) { print "n/a"; exit }
            m = s / n
            v = (ss / n) - (m * m)
            if (v < 0) v = 0
            if (m == 0) print "n/a"
            else printf "%.2f\n", sqrt(v) * 100 / m
        }' "$1"
}

run_one() {
    local prover="$1"
    local log_rows="$2"
    local out_file="$3"
    local extra_args=()
    if $BREAKDOWN; then
        extra_args+=(--breakdown)
    fi
    "$BIN" \
        --prover "$prover" \
        --log-rows "$log_rows" \
        --num-sequences "$NUM_SEQUENCES" \
        --blowup "$BLOWUP" \
        --queries "$FRI_QUERIES" \
        --grinding "$GRINDING" \
        ${extra_args[@]+"${extra_args[@]}"} > "$out_file" 2>&1
}

run_prover() {
    local prover="$1"
    local log_rows="$2"
    local times_file="$TMP_DIR/${prover}_${log_rows}.times"
    local metrics_file="$TMP_DIR/${prover}_${log_rows}.metrics"
    : > "$times_file"
    : > "$metrics_file"

    for run_i in $(seq 1 "$RUNS"); do
        local out_file="$TMP_DIR/${prover}_${log_rows}_${run_i}.stdout"
        if ! run_one "$prover" "$log_rows" "$out_file"; then
            echo "[$prover] failed at log_rows=$log_rows run=$run_i" >&2
            cat "$out_file" >&2
            exit 1
        fi
        if [ "$run_i" -eq 1 ]; then
            grep '^AUDIT	' "$out_file" >&2 || true
        fi
        local metrics_line
        metrics_line="$(grep '^METRICS	' "$out_file" | head -1)"
        if [ -z "$metrics_line" ]; then
            echo "missing METRICS line for $prover log_rows=$log_rows run=$run_i" >&2
            cat "$out_file" >&2
            exit 1
        fi
        printf '%s\n' "$metrics_line" >> "$metrics_file"
        metric_value "$metrics_line" prove_s >> "$times_file"

        if [ -n "$REPORT_DIR" ]; then
            cp "$out_file" "$REPORT_DIR/raw/${prover}_log${log_rows}_run${run_i}.stdout"
            if $BREAKDOWN; then
                # BREAKDOWN lines are TAB-separated `key=value` fields, e.g.:
                #   BREAKDOWN<TAB>workload=fib_pair<TAB>prover=p3<TAB>log_rows=21<TAB>rows=2097152<TAB>phase=norm_fri<TAB>ms=93.4<TAB>pct=23.8<TAB>kind=norm
                # Field set varies by kind (lambda native carries table=/table_rows=;
                # p3 native carries depth=/total_ms=; norm carries pct=; agg carries
                # calls=). One awk pass splits on the first '=' of each field and
                # projects a fixed-width TSV row (missing keys -> empty).
                grep '^BREAKDOWN	' "$out_file" | awk -v run="$run_i" '
                    BEGIN { FS = "\t"; OFS = "\t" }
                    {
                        delete kv
                        for (i = 1; i <= NF; i++) {
                            eq = index($i, "=")
                            if (eq > 0) kv[substr($i, 1, eq - 1)] = substr($i, eq + 1)
                        }
                        print run, kv["workload"], kv["prover"], kv["log_rows"], kv["rows"], \
                              kv["phase"], kv["ms"], kv["total_ms"], kv["pct"], kv["calls"], \
                              kv["kind"], kv["depth"], kv["table"], kv["table_rows"]
                    }' >> "$REPORT_DIR/breakdown.tsv"
            fi
        fi
    done

    median_file "$times_file"
}

ratio() {
    awk -v a="$1" -v b="$2" 'BEGIN { if (b == 0 || b == "n/a") print "n/a"; else printf "%.3f", a / b }'
}

if [ -n "$REPORT_DIR" ]; then
    printf "log_rows\trows\tlambda_median_s\tlambda_cv_pct\tp3_median_s\tp3_cv_pct\tratio_lambda_over_p3\truns\n" > "$REPORT_DIR/results.tsv"
    if $BREAKDOWN; then
        printf "run\tworkload\tprover\tlog_rows\trows\tphase\tms\ttotal_ms\tpct\tcalls\tkind\tdepth\ttable\ttable_rows\n" > "$REPORT_DIR/breakdown.tsv"
    fi
fi

for lr in "${LOG_ROWS[@]}"; do
    rows=$((1 << lr))
    echo "--- log_rows=$lr rows=$rows ---"
    lambda_median="n/a"
    lambda_cv="n/a"
    p3_median="n/a"
    p3_cv="n/a"

    if $RUN_LAMBDA; then
        lambda_median="$(run_prover lambda "$lr")"
        lambda_cv="$(cv_pct_file "$TMP_DIR/lambda_${lr}.times")"
        echo "lambda prove median ${lambda_median}s (CV ${lambda_cv}%)"
    fi

    if $RUN_P3; then
        p3_median="$(run_prover p3 "$lr")"
        p3_cv="$(cv_pct_file "$TMP_DIR/p3_${lr}.times")"
        echo "p3     prove median ${p3_median}s (CV ${p3_cv}%)"
    fi

    r="n/a"
    if $RUN_LAMBDA && $RUN_P3; then
        r="$(ratio "$lambda_median" "$p3_median")"
        echo "lambda/p3 ratio: ${r}x"
    fi
    echo

    if [ -n "$REPORT_DIR" ]; then
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
            "$lr" "$rows" "$lambda_median" "$lambda_cv" "$p3_median" "$p3_cv" "$r" "$RUNS" \
            >> "$REPORT_DIR/results.tsv"
    fi
done
