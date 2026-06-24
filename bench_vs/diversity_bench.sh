#!/usr/bin/env bash
# TEMP (branch tmp/ethrex-diverse-fixtures): compare proving cost of ethrex blocks
# with different account-diversity modes (same / recipients / distinct) at a fixed
# tx count. Generates fixtures via the patched tooling/ethrex-fixtures, then proves
# each with the jemalloc-stats CLI and reports best-of-N time + peak heap + input size.
#
# Usage: bash bench_vs/diversity_bench.sh [N]   (N = tx count, default 20)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N=${1:-20}
WORK=/tmp/ethrex_div
ELF=executor/program_artifacts/rust/ethrex.elf
MODES=(same recipients distinct)
REPEATS=2

test -f "$ELF" || { echo "FATAL: $ELF missing — build the ethrex guest ELF first"; exit 1; }
mkdir -p "$WORK"

echo "=== building fixtures tool (with the new modes) ==="
( cd tooling/ethrex-fixtures && cargo build --release )
GEN=tooling/ethrex-fixtures/target/release/ethrex-fixtures

echo "=== generating N=$N fixtures: same / recipients / distinct ==="
for m in "${MODES[@]}"; do
  "$GEN" "$N" "$WORK/ethrex_${N}_${m}.bin" "$m"
done
ls -l "$WORK"/*.bin

echo "=== building CLI with jemalloc-stats ==="
cargo build --release -p cli --features jemalloc-stats
CLI=target/release/cli

declare -A BT BH SZ
for m in "${MODES[@]}"; do
  f="$WORK/ethrex_${N}_${m}.bin"
  SZ[$m]=$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")
  bt=""; bh=0
  for r in $(seq 1 "$REPEATS"); do
    out=$("$CLI" prove "$ELF" -o "$WORK/p.proof" --private-input "$f" --time 2>/dev/null)
    t=$(printf '%s\n' "$out" | sed -nE 's/.*Proving time: ([0-9.]+)s.*/\1/p' | head -1)
    h=$(printf '%s\n' "$out" | sed -nE 's/.*Peak heap: ([0-9]+) MB.*/\1/p' | head -1)
    rm -f "$WORK/p.proof"
    echo "  mode=$m run=$r -> time=${t:-?}s heap=${h:-?}MB"
    if [ -n "$t" ] && { [ -z "$bt" ] || awk -v a="$t" -v b="$bt" 'BEGIN{exit !(a<b)}'; }; then bt=$t; fi
    if [ -n "$h" ] && [ "$h" -gt "$bh" ]; then bh=$h; fi
  done
  BT[$m]=${bt:-NaN}; BH[$m]=$bh
done

echo
printf '%-12s %10s %10s %12s\n' "mode(N=$N)" "time(s)" "heap(MB)" "input(B)"
printf '%-12s %10s %10s %12s\n' "----------" "-------" "--------" "--------"
for m in "${MODES[@]}"; do
  printf '%-12s %10s %10s %12s\n' "$m" "${BT[$m]}" "${BH[$m]}" "${SZ[$m]}"
done
echo "(time=min of $REPEATS, heap=max of $REPEATS, default parallelism)"
echo "=== done ==="
