#!/usr/bin/env bash
# A/B/C benchmark: monolithic WITH range checks (main) vs monolithic WITHOUT them
# (this branch, bench/page-drop-arebytes) vs continuation — for a set of ethrex
# blocks, all in ONE session so the numbers are directly comparable.
#
# Emits three tables: proving time (with % diffs), peak RSS, and page counts
# (populated / touched / untouched) — the latter from the branch binary, which
# carries the [PAGE-COUNT] instrumentation.
#
# The epoch size for each block's one-epoch continuation is computed automatically
# as ceil(log2(cycles)), floored at 18.
#
# Usage (from repo root, on branch bench/page-drop-arebytes):
#   ./bench_vs/ab_page_arebytes.sh [--runs R] [--no-color]
#
# Prereq: each block's input file must exist in executor/tests/ (generate the
# 15-tx fixture first — see the ethrex-fixtures crate).

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

# Blocks: "label|input_basename" (epoch size is computed per block).
BLOCKS=(
    "5tx|ethrex_5_transfers.bin"
    "10tx|ethrex_10_transfers.bin"
    "15tx|ethrex_15_transfers.bin"
)

mkdir -p "$TMP_DIR"; rm -rf "${TMP_DIR:?}"/*

# --- Build both binaries ----------------------------------------------------
echo -e "${GREEN}[build] this branch (monolithic WITHOUT range checks + PAGE-COUNT)...${NC}"
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

# --- Per-block epoch size (ceil(log2(cycles)), floored at 18) ---------------
declare -A NMAP
echo -e "${BOLD}Cycle counts / epoch sizes:${NC}"
for entry in "${BLOCKS[@]}"; do
    IFS='|' read -r label basename <<< "$entry"
    input="$ROOT_DIR/executor/tests/$basename"
    [[ -f "$input" ]] || { echo -e "${RED}input not found: $input (generate the fixture first)${NC}"; exit 1; }
    cyc=$("$BRANCH_CLI" execute "$ELF" --private-input "$input" --cycles 2>/dev/null | sed -nE 's/.*Cycles: ([0-9]+).*/\1/p')
    [[ -n "$cyc" ]] || { echo -e "${RED}could not read cycles for $label${NC}"; exit 1; }
    n=$(awk -v c="$cyc" 'BEGIN{n=18; while((2^n) < c) n++; print n}')
    NMAP[$label]=$n
    printf "  %-5s %12s cycles  ->  epoch-size-log2=%s\n" "$label" "$cyc" "$n"
done
echo ""

# --- One measured run: echoes "time_s rss_kb" (writes full output to last_run.out) ---
one_run() {
    local out="$TMP_DIR/last_run.out"
    "$TIME_BIN" -v "$@" >"$out" 2>&1 || { echo -e "${RED}FAILED:${NC}" >&2; cat "$out" >&2; exit 1; }
    local t rss ep
    t=$(sed -nE 's/.*Proving time: ([0-9.]+)s.*/\1/p' "$out" | head -1)
    rss=$(sed -nE 's/.*Maximum resident set size \(kbytes\): ([0-9]+).*/\1/p' "$out" | head -1)
    ep=$(sed -nE 's/.*Epochs: ([0-9]+).*/\1/p' "$out" | head -1)
    if [[ -n "$ep" && "$ep" != "1" ]]; then
        echo -e "${RED}continuation ran $ep epochs, not 1 — epoch size too small${NC}" >&2; exit 1
    fi
    [[ -n "$t" && -n "$rss" ]] || { echo -e "${RED}parse failed${NC}" >&2; cat "$out" >&2; exit 1; }
    printf "%s %s" "$t" "$rss"
}

median() { sort -n | awk '{a[NR]=$1} END{ if(NR==0){print "n/a";exit} if(NR%2){print a[(NR+1)/2]} else printf "%.3f\n",(a[NR/2]+a[NR/2+1])/2 }'; }

declare -A T_SAMP R_SAMP PC_POP PC_TOUCH PC_UNTOUCH

# $1=block_label $2=input_path $3=N $4=version_key
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
    # Capture page counts once per block from the branch monolithic run.
    if [[ "$vkey" == "without" && -z "${PC_POP[$label]:-}" ]]; then
        local line
        line=$(grep -m1 "PAGE-COUNT] populated=" "$TMP_DIR/last_run.out" || true)
        if [[ -n "$line" ]]; then
            PC_POP[$label]=$(sed -nE 's/.*populated=([0-9]+).*/\1/p' <<< "$line")
            PC_TOUCH[$label]=$(sed -nE 's/.*touched=([0-9]+).*/\1/p' <<< "$line")
            PC_UNTOUCH[$label]=$(sed -nE 's/.*untouched=([0-9]+).*/\1/p' <<< "$line")
        fi
    fi
    printf "    %-5s %-8s r: %ss, %d MB\n" "$label" "$vkey" "$t" "$((rss/1024))"
}

for ((r=1; r<=RUNS; r++)); do
    echo -e "${BOLD}--- round $r/$RUNS ---${NC}"
    for entry in "${BLOCKS[@]}"; do
        IFS='|' read -r label basename <<< "$entry"
        input="$ROOT_DIR/executor/tests/$basename"
        run_version "$label" "$input" "${NMAP[$label]}" with
        run_version "$label" "$input" "${NMAP[$label]}" without
        run_version "$label" "$input" "${NMAP[$label]}" cont
    done
done

pct() { awk -v a="$1" -v b="$2" 'BEGIN{ if(a==0){print "n/a"} else printf "%+.1f%%", (b-a)/a*100 }'; }

echo ""
echo -e "${BOLD}=== Proving time (median of $RUNS runs, seconds) ===${NC}"
echo ""
printf "| %-5s | %-15s | %-16s | %-13s | %-11s | %-12s | %-11s |\n" \
    "Block" "Mono w/ checks" "Mono w/o checks" "Continuation" "w/o vs w/" "cont vs w/o" "cont vs w/"
printf "|-------|-----------------|------------------|---------------|-------------|--------------|-------------|\n"
for entry in "${BLOCKS[@]}"; do
    IFS='|' read -r label _ <<< "$entry"
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
    IFS='|' read -r label _ <<< "$entry"
    mw=$(( $(printf "%s\n" ${R_SAMP["${label}_with"]}    | median) / 1024 ))
    mo=$(( $(printf "%s\n" ${R_SAMP["${label}_without"]} | median) / 1024 ))
    ct=$(( $(printf "%s\n" ${R_SAMP["${label}_cont"]}    | median) / 1024 ))
    printf "| %-5s | %-15s | %-16s | %-13s |\n" "$label" "$mw" "$mo" "$ct"
done

echo ""
echo -e "${BOLD}=== Page tables (deterministic; from branch binary) ===${NC}"
echo ""
printf "| %-5s | %-22s | %-22s | %-18s |\n" "Block" "Populated (monolithic)" "Touched (continuation)" "Untouched (wasted)"
printf "|-------|------------------------|------------------------|--------------------|\n"
for entry in "${BLOCKS[@]}"; do
    IFS='|' read -r label _ <<< "$entry"
    printf "| %-5s | %-22s | %-22s | %-18s |\n" \
        "$label" "${PC_POP[$label]:-?}" "${PC_TOUCH[$label]:-?}" "${PC_UNTOUCH[$label]:-?}"
done

echo ""
echo "Raw per-run output under $TMP_DIR/"
