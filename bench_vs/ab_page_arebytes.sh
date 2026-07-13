#!/usr/bin/env bash
# A/B/C benchmark: monolithic WITH range checks (main) vs monolithic WITHOUT them
# (this branch, bench/page-drop-arebytes) vs continuation — for 5tx and 10tx ethrex
# blocks, all in ONE session so the numbers are directly comparable.
#
# Runs the three versions × two blocks, interleaved, RUNS times each, under
# `/usr/bin/time -v`, and reports MEDIAN proving time + peak RSS with % differences.
#
# Usage (from the repo root, while on branch bench/page-drop-arebytes):
#   ./bench_vs/ab_page_arebytes.sh [--runs R] [--no-color]
#
# It builds:
#   - main's CLI in a detached git worktree at $WT   (monolithic WITH checks)
#   - this branch's CLI at target/release/cli        (monolithic WITHOUT checks)
# and proves the SAME input files (from this checkout) with both.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT="/tmp/lambda_main_wt"
TMP_DIR="/tmp/ab_page_arebytes"
RUNS=3
NO_COLOR=false

RED='\033[0;31m'; GREEN='\033[0;32m'; BOLD='\033[1m'; NC='\033[0m'

while [[ $# -gt 0 ]]; do
    case $1 in
        --runs)     RUNS=$2; shift 2 ;;
        --no-color) NO_COLOR=true; shift ;;
        -h|--help)  echo "Usage: $0 [--runs R] [--no-color]"; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
if $NO_COLOR; then RED=''; GREEN=''; BOLD=''; NC=''; fi

TIME_BIN=$(command -v time || true)
[[ -x /usr/bin/time ]] && TIME_BIN=/usr/bin/time
[[ -x "$TIME_BIN" ]] || { echo -e "${RED}/usr/bin/time (GNU time) not found (apt-get install -y time)${NC}"; exit 1; }

ELF="$ROOT_DIR/executor/program_artifacts/rust/ethrex.elf"
[[ -f "$ELF" ]] || { echo -e "${RED}ethrex.elf not found at $ELF${NC}"; exit 1; }

# Blocks: "label|input_basename|epoch_size_log2"
BLOCKS=(
    "5tx|ethrex_5_transfers.bin|22"
    "10tx|ethrex_10_transfers.bin|23"
)

mkdir -p "$TMP_DIR"; rm -rf "${TMP_DIR:?}"/*

# --- Build both binaries ----------------------------------------------------
echo -e "${GREEN}[build] this branch (monolithic WITHOUT range checks)...${NC}"
cargo build --release -p cli --features jemalloc-stats --manifest-path "$ROOT_DIR/Cargo.toml" 2>&1 | tail -2
BRANCH_CLI="$ROOT_DIR/target/release/cli"
[[ -x "$BRANCH_CLI" ]] || { echo -e "${RED}branch CLI build failed${NC}"; exit 1; }

echo -e "${GREEN}[build] main via git worktree (monolithic WITH range checks)...${NC}"
if [[ ! -d "$WT" ]]; then
    git -C "$ROOT_DIR" worktree add --detach "$WT" origin/main 2>&1 | tail -2
else
    git -C "$WT" fetch origin main 2>&1 | tail -1
    git -C "$WT" checkout --detach origin/main 2>&1 | tail -1
fi
cargo build --release -p cli --features jemalloc-stats --manifest-path "$WT/Cargo.toml" 2>&1 | tail -2
MAIN_CLI="$WT/target/release/cli"
[[ -x "$MAIN_CLI" ]] || { echo -e "${RED}main CLI build failed${NC}"; exit 1; }

echo ""
echo -e "${BOLD}main commit:${NC}   $(git -C "$WT" rev-parse --short HEAD)"
echo -e "${BOLD}branch commit:${NC} $(git -C "$ROOT_DIR" rev-parse --short HEAD)"
echo ""

# --- One measured run: echoes "time_s rss_kb" (and asserts epochs=1 for cont) ---
one_run() {
    local out="$TMP_DIR/run.$$.$RANDOM.out"
    "$TIME_BIN" -v "$@" >"$out" 2>&1 || { echo -e "${RED}FAILED:${NC}" >&2; cat "$out" >&2; exit 1; }
    local t rss ep
    t=$(sed -nE 's/.*Proving time: ([0-9.]+)s.*/\1/p' "$out" | head -1)
    rss=$(sed -nE 's/.*Maximum resident set size \(kbytes\): ([0-9]+).*/\1/p' "$out" | head -1)
    ep=$(sed -nE 's/.*Epochs: ([0-9]+).*/\1/p' "$out" | head -1)
    if [[ -n "$ep" && "$ep" != "1" ]]; then
        echo -e "${RED}continuation ran $ep epochs, not 1 — fix epoch-size-log2${NC}" >&2; exit 1
    fi
    [[ -n "$t" && -n "$rss" ]] || { echo -e "${RED}parse failed${NC}" >&2; cat "$out" >&2; exit 1; }
    printf "%s %s" "$t" "$rss"
}

median() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print "n/a";exit} if(NR%2){print a[(NR+1)/2]} else printf "%.3f\n",(a[NR/2]+a[NR/2+1])/2 }'; }

# Accumulators keyed by "block_version" -> space-separated samples.
declare -A T_SAMP R_SAMP

# Version runner: $1=block_label $2=input_path $3=N $4=version_key
run_version() {
    local label=$1 input=$2 n=$3 vkey=$4 res
    case $vkey in
        with)    res=$(one_run "$MAIN_CLI"   prove "$ELF" -o "$TMP_DIR/o.proof" --private-input "$input" --time --cycles) ;;
        without) res=$(one_run "$BRANCH_CLI" prove "$ELF" -o "$TMP_DIR/o.proof" --private-input "$input" --time --cycles) ;;
        cont)    res=$(one_run "$BRANCH_CLI" prove "$ELF" -o "$TMP_DIR/o.proof" --private-input "$input" --continuations --epoch-size-log2 "$n" --time --cycles) ;;
    esac
    local t=${res% *} rss=${res#* }
    T_SAMP["${label}_${vkey}"]+="$t "
    R_SAMP["${label}_${vkey}"]+="$rss "
    printf "    %-4s %-8s r: %ss, %d MB\n" "$label" "$vkey" "$t" "$((rss/1024))"
}

# --- Interleaved runs (each round touches all 3 versions of both blocks) ----
for ((r=1; r<=RUNS; r++)); do
    echo -e "${BOLD}--- round $r/$RUNS ---${NC}"
    for entry in "${BLOCKS[@]}"; do
        IFS='|' read -r label basename n <<< "$entry"
        input="$ROOT_DIR/executor/tests/$basename"
        [[ -f "$input" ]] || { echo -e "${RED}input not found: $input${NC}"; exit 1; }
        run_version "$label" "$input" "$n" with
        run_version "$label" "$input" "$n" without
        run_version "$label" "$input" "$n" cont
    done
done

# --- Medians + table --------------------------------------------------------
pct() { awk -v a="$1" -v b="$2" 'BEGIN{ if(a==0){print "n/a"} else printf "%+.1f%%", (b-a)/a*100 }'; }

echo ""
echo -e "${BOLD}=== Proving time (median of $RUNS runs, seconds) ===${NC}"
echo ""
printf "| %-5s | %-15s | %-16s | %-13s | %-11s | %-12s | %-11s |\n" \
    "Block" "Mono w/ checks" "Mono w/o checks" "Continuation" "w/o vs w/" "cont vs w/o" "cont vs w/"
printf "|-------|-----------------|------------------|---------------|-------------|--------------|-------------|\n"
for entry in "${BLOCKS[@]}"; do
    IFS='|' read -r label _ _ <<< "$entry"
    mw=$(printf "%s\n" ${T_SAMP["${label}_with"]}    | median)
    mo=$(printf "%s\n" ${T_SAMP["${label}_without"]} | median)
    ct=$(printf "%s\n" ${T_SAMP["${label}_cont"]}    | median)
    printf "| %-5s | %-15s | %-16s | %-13s | %-11s | %-12s | %-11s |\n" \
        "$label" "$mw" "$mo" "$ct" "$(pct "$mw" "$mo")" "$(pct "$mo" "$ct")" "$(pct "$mw" "$ct")"
done

echo ""
echo -e "${BOLD}=== Peak RSS (median of $RUNS runs, MB) ===${NC}"
echo ""
printf "| %-5s | %-15s | %-16s | %-13s |\n" "Block" "Mono w/ checks" "Mono w/o checks" "Continuation"
printf "|-------|-----------------|------------------|---------------|\n"
for entry in "${BLOCKS[@]}"; do
    IFS='|' read -r label _ _ <<< "$entry"
    mw=$(( $(printf "%s\n" ${R_SAMP["${label}_with"]}    | median) / 1024 ))
    mo=$(( $(printf "%s\n" ${R_SAMP["${label}_without"]} | median) / 1024 ))
    ct=$(( $(printf "%s\n" ${R_SAMP["${label}_cont"]}    | median) / 1024 ))
    printf "| %-5s | %-15s | %-16s | %-13s |\n" "$label" "$mw" "$mo" "$ct"
done

echo ""
echo "Legend: w/o vs w/ = removing PAGE range checks' effect on monolithic (negative = faster)."
echo "        cont vs w/ = full continuation advantage;  cont vs w/o = advantage that remains after removing the checks."
echo "Raw per-run output in $TMP_DIR/"
