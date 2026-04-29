#!/bin/bash
# Multi-workload prover bench suite.
#
# Wraps bench_prove.sh and runs it across one or more programs that together
# exercise every chip in the prover, instead of relying on fib alone.
#
# Default behavior runs fib only (cheap canary, ~ same as bench_prove.sh).
# Use --all / --only / --skip to opt in to the full coverage suite.
#
# Usage:
#   bench_prove_suite.sh [runs=1] [base_branch=main | --no-compare]
#                        [--all] [--only NAMES] [--skip NAMES] [--instruments]
#
# Examples:
#   bench_prove_suite.sh                          # fib only, 1 run, vs main
#   bench_prove_suite.sh 3 main                   # fib only, 3 runs, vs main
#   bench_prove_suite.sh 3 main --all             # full 5-program suite
#   bench_prove_suite.sh 3 main --only keccak,quicksort
#   bench_prove_suite.sh 3 main --skip hashmap    # all except hashmap
#
# Programs in the suite (run order):
#   fib         executor/program_artifacts/asm/fib_iterative_8M.elf
#   keccak      executor/program_artifacts/bench/keccak.elf
#   quicksort   executor/program_artifacts/bench/quicksort.elf
#   modular_exp executor/program_artifacts/bench/modular_exp.elf
#   hashmap     executor/program_artifacts/bench/hashmap.elf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TMP_DIR="/tmp/bench_prove_suite"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

# Verdict thresholds, applied to the time delta vs base_branch.
REGRESSION_FAIL_PCT=5
REGRESSION_WARN_PCT=2

# Suite definition. Run order is fib first (cheap canary).
# Bash 3.2 (macOS default) has no associative arrays, so name→ELF is a case.
SUITE_ORDER=(fib keccak quicksort modular_exp hashmap)

elf_for() {
    case "$1" in
        fib)         echo "$ROOT_DIR/executor/program_artifacts/asm/fib_iterative_8M.elf" ;;
        keccak)      echo "$ROOT_DIR/executor/program_artifacts/bench/keccak.elf" ;;
        quicksort)   echo "$ROOT_DIR/executor/program_artifacts/bench/quicksort.elf" ;;
        modular_exp) echo "$ROOT_DIR/executor/program_artifacts/bench/modular_exp.elf" ;;
        hashmap)     echo "$ROOT_DIR/executor/program_artifacts/bench/hashmap.elf" ;;
        *) return 1 ;;
    esac
}

# --- Argument parsing ------------------------------------------------------

SELECT_ALL=false
ONLY_LIST=""
SKIP_LIST=""
INSTRUMENTS_FLAG=""
POSITIONAL=()

require_value_arg() {
    local flag=$1 val=${2:-}
    case "$val" in
        ""|--*)
            echo -e "${RED}ERROR: $flag requires a non-empty list of program names (got '$val')${NC}" >&2
            exit 2
            ;;
    esac
}

while [ $# -gt 0 ]; do
    case "$1" in
        --all)         SELECT_ALL=true; shift ;;
        --only)        require_value_arg "--only" "${2:-}"; ONLY_LIST=$2; shift 2 ;;
        --skip)        require_value_arg "--skip" "${2:-}"; SKIP_LIST=$2; shift 2 ;;
        --instruments) INSTRUMENTS_FLAG="--instruments"; shift ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            POSITIONAL+=("$1"); shift
            ;;
    esac
done

RUNS=${POSITIONAL[0]:-1}
BASE_BRANCH=${POSITIONAL[1]:-main}

if [[ ! "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
    echo -e "${RED}ERROR: runs must be a positive integer (got '$RUNS')${NC}" >&2
    exit 2
fi

if [ -n "$ONLY_LIST" ] && { $SELECT_ALL || [ -n "$SKIP_LIST" ]; }; then
    echo -e "${RED}ERROR: --only is mutually exclusive with --all and --skip${NC}" >&2
    exit 2
fi

# --- Resolve which programs to run ----------------------------------------

is_known_name() {
    local n=$1
    for k in "${SUITE_ORDER[@]}"; do
        [ "$k" = "$n" ] && return 0
    done
    return 1
}

SELECTED=()
if [ -n "$ONLY_LIST" ]; then
    IFS=',' read -ra ONLY <<< "$ONLY_LIST"
    for n in "${ONLY[@]}"; do
        if ! is_known_name "$n"; then
            echo -e "${RED}ERROR: unknown program '$n'. Known: ${SUITE_ORDER[*]}${NC}" >&2
            exit 2
        fi
        SELECTED+=("$n")
    done
elif $SELECT_ALL || [ -n "$SKIP_LIST" ]; then
    SKIPS=()
    if [ -n "$SKIP_LIST" ]; then
        IFS=',' read -ra SKIPS <<< "$SKIP_LIST"
        for n in "${SKIPS[@]}"; do
            if ! is_known_name "$n"; then
                echo -e "${RED}ERROR: unknown program '$n'. Known: ${SUITE_ORDER[*]}${NC}" >&2
                exit 2
            fi
        done
    fi
    for n in "${SUITE_ORDER[@]}"; do
        skip_it=false
        for s in "${SKIPS[@]+"${SKIPS[@]}"}"; do
            if [ "$n" = "$s" ]; then skip_it=true; break; fi
        done
        $skip_it || SELECTED+=("$n")
    done
else
    SELECTED=(fib)
fi

# --- Validate ELFs ---------------------------------------------------------

MISSING=()
for n in "${SELECTED[@]}"; do
    elf=$(elf_for "$n")
    [ -f "$elf" ] || MISSING+=("$n -> $elf")
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo -e "${RED}ERROR: missing ELF artifact(s):${NC}" >&2
    for m in "${MISSING[@]}"; do echo "  $m" >&2; done
    echo "" >&2
    echo "Build them with:" >&2
    echo "  make compile-programs-asm   # for asm/*.elf (fib)" >&2
    echo "  make compile-bench          # for bench/*.elf (keccak, quicksort, modular_exp, hashmap)" >&2
    exit 1
fi

# --- Run each program ------------------------------------------------------

rm -rf "$TMP_DIR" && mkdir -p "$TMP_DIR"

CURRENT_BRANCH=$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD)
COMPARE=true
if [ "$BASE_BRANCH" = "--no-compare" ] || [ "$CURRENT_BRANCH" = "$BASE_BRANCH" ]; then
    COMPARE=false
fi

echo -e "${BOLD}Suite:${NC}     ${SELECTED[*]}"
echo -e "${BOLD}Runs:${NC}      $RUNS"
echo -e "${BOLD}Branch:${NC}    $CURRENT_BRANCH"
if $COMPARE; then
    echo -e "${BOLD}Compare:${NC}   vs $BASE_BRANCH"
else
    echo -e "${BOLD}Compare:${NC}   none"
fi
[ -n "$INSTRUMENTS_FLAG" ] && echo -e "${BOLD}Instruments:${NC} on"

for n in "${SELECTED[@]}"; do
    echo ""
    echo -e "${BOLD}========== $n ==========${NC}"
    LOG="$TMP_DIR/${n}.log"
    if ! "$SCRIPT_DIR/bench_prove.sh" "$(elf_for "$n")" "$RUNS" "$BASE_BRANCH" $INSTRUMENTS_FLAG 2>&1 | tee "$LOG"; then
        echo -e "${RED}WARN: bench_prove.sh exited non-zero for $n${NC}" >&2
    fi
    # bench_prove.sh writes to /tmp/bench_prove/instruments.txt and clobbers it
    # on the next invocation; preserve a per-program copy here.
    if [ -n "$INSTRUMENTS_FLAG" ] && [ -f "/tmp/bench_prove/instruments.txt" ]; then
        cp "/tmp/bench_prove/instruments.txt" "$TMP_DIR/${n}_instruments.txt"
    fi
done

# --- Aggregate verdict -----------------------------------------------------

echo ""
echo -e "${BOLD}===========================================${NC}"
echo -e "${BOLD}Suite verdict${NC}"
echo -e "${BOLD}===========================================${NC}"

if ! $COMPARE; then
    echo "(no comparison: same branch as base or --no-compare)"
    echo ""
    printf "  %-12s  %12s  %12s\n" "program" "time(median)" "heap(median)"
    for n in "${SELECTED[@]}"; do
        LOG="$TMP_DIR/${n}.log"
        TIME_MED=$(grep -E "current[[:space:]]+time\(mean\)" "$LOG" 2>/dev/null | sed -E 's/.*time\(median\):[[:space:]]+([0-9.]+s).*/\1/' | tail -1 || true)
        HEAP_MED=$(grep -E "current[[:space:]]+time\(mean\)" "$LOG" 2>/dev/null | sed -E 's/.*heap\(median\):[[:space:]]+([0-9]+ MB).*/\1/' | tail -1 || true)
        printf "  %-12s  %12s  %12s\n" "$n" "${TIME_MED:-N/A}" "${HEAP_MED:-N/A}"
    done
    echo ""
    echo "Per-program logs in $TMP_DIR/"
    exit 0
fi

OVERALL=PASS
PROGRAM_LINES=()

for n in "${SELECTED[@]}"; do
    LOG="$TMP_DIR/${n}.log"
    # || true: a missing pattern (e.g. crashed run, partial log) must not abort the script.
    TIME_DELTA=$(grep -E "^[[:space:]]+Time:[[:space:]]+[+-]" "$LOG" 2>/dev/null | tail -1 | sed -E 's/.*Time:[[:space:]]+([+-][0-9.]+)%.*/\1/' || true)
    HEAP_DELTA=$(grep -E "^[[:space:]]+Heap:" "$LOG" 2>/dev/null | tail -1 | sed -E 's/.*\(([+-][0-9.]+)%\).*/\1/' || true)

    if [ -z "$TIME_DELTA" ]; then
        PROGRAM_LINES+=("$(printf "  %-12s  ${YELLOW}[%s]${NC} (no comparison data — see log)" "$n" "INCONCLUSIVE")")
        case "$OVERALL" in PASS|WARN) OVERALL=INCONCLUSIVE ;; esac
        continue
    fi

    # Status: regression > FAIL_PCT = FAIL, > WARN_PCT = WARN, otherwise OK.
    STATUS=OK; COLOR=$GREEN
    if awk "BEGIN {exit !($TIME_DELTA > $REGRESSION_FAIL_PCT)}"; then
        STATUS=FAIL; COLOR=$RED; OVERALL=FAIL
    elif awk "BEGIN {exit !($TIME_DELTA > $REGRESSION_WARN_PCT)}"; then
        STATUS=WARN; COLOR=$YELLOW
        [ "$OVERALL" = PASS ] && OVERALL=WARN
    fi

    PROGRAM_LINES+=("$(printf "  %-12s  time: %+7s%%   heap: %+7s%%   ${COLOR}[%s]${NC}" \
        "$n" "$TIME_DELTA" "${HEAP_DELTA:-N/A}" "$STATUS")")
done

echo ""
printf "  %-12s  %-15s   %-15s   %s\n" "program" "time delta" "heap delta" "status"
echo "  ----------------------------------------------------------------"
for line in "${PROGRAM_LINES[@]}"; do
    echo -e "$line"
done
echo ""
echo -e "  Thresholds: WARN > +${REGRESSION_WARN_PCT}%, FAIL > +${REGRESSION_FAIL_PCT}% (regression vs $BASE_BRANCH)"
echo ""

case $OVERALL in
    PASS)         echo -e "${GREEN}${BOLD}Overall: PASS${NC}" ;;
    WARN)         echo -e "${YELLOW}${BOLD}Overall: WARN${NC}" ;;
    INCONCLUSIVE) echo -e "${YELLOW}${BOLD}Overall: INCONCLUSIVE${NC} (one or more programs produced no comparison data)" ;;
    FAIL)         echo -e "${RED}${BOLD}Overall: FAIL${NC}" ;;
esac
echo ""
echo "Per-program logs in $TMP_DIR/"

case $OVERALL in
    FAIL)         exit 1 ;;
    INCONCLUSIVE) exit 2 ;;
    *)            exit 0 ;;
esac
