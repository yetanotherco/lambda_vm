#!/usr/bin/env bash
# TEMP (branch tmp/ethrex-diverse-fixtures): measure run-to-run stability of proving
# one ethrex `distinct` block. Proves the SAME fixture K times and reports the
# distribution (mean, sample-SD, CV%, min/max spread) of time, jemalloc peak heap,
# and OS peak RSS (via /usr/bin/time -v if present).
#
# Usage: bash bench_vs/stability_check.sh [N_transfers=20] [K_repeats=8]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

N=${1:-20}
K=${2:-8}
WORK=/tmp/ethrex_stab
ELF=executor/program_artifacts/rust/ethrex.elf
test -f "$ELF" || { echo "FATAL: $ELF missing — build the ethrex guest ELF first"; exit 1; }
mkdir -p "$WORK"

echo "=== build generator + jemalloc CLI ==="
( cd tooling/ethrex-fixtures && cargo build --release )
cargo build --release -p cli --features jemalloc-stats
GEN=tooling/ethrex-fixtures/target/release/ethrex-fixtures
CLI=target/release/cli

f="$WORK/ethrex_${N}_distinct.bin"
"$GEN" "$N" "$f" distinct

TIMEV=""
if [ -x /usr/bin/time ]; then TIMEV="/usr/bin/time -v"; else echo "(note: /usr/bin/time absent -> RSS will be n/a; 'sudo apt install time' to enable)"; fi

TIMES=""; HEAPS=""; RSSMB=""
for r in $(seq 1 "$K"); do
  terr="$WORK/run_${r}.err"
  if [ -n "$TIMEV" ]; then
    out=$($TIMEV "$CLI" prove "$ELF" -o "$WORK/p.proof" --private-input "$f" --time 2>"$terr")
    rss_kb=$(grep -i "Maximum resident set size" "$terr" | grep -oE '[0-9]+' | head -1)
  else
    out=$("$CLI" prove "$ELF" -o "$WORK/p.proof" --private-input "$f" --time 2>/dev/null)
    rss_kb=""
  fi
  rm -f "$WORK/p.proof"
  t=$(printf '%s\n' "$out" | sed -nE 's/.*Proving time: ([0-9.]+)s.*/\1/p' | head -1)
  h=$(printf '%s\n' "$out" | sed -nE 's/.*Peak heap: ([0-9]+) MB.*/\1/p' | head -1)
  rss_mb=""; [ -n "$rss_kb" ] && rss_mb=$(awk -v k="$rss_kb" 'BEGIN{printf "%.0f", k/1024}')
  echo "  run $r/$K -> time=${t:-?}s heap=${h:-?}MB rss=${rss_mb:-n/a}MB"
  TIMES="$TIMES $t"; HEAPS="$HEAPS $h"; [ -n "$rss_mb" ] && RSSMB="$RSSMB $rss_mb"
done

stats() {  # reads numbers on stdin; prints mean/sd/cv/min/max/spread
  awk '{for(i=1;i<=NF;i++){v=$i; n++; s+=v; ss+=v*v; if(min==""||v<min)min=v; if(v>max)max=v}}
       END{ if(n<1){print "n/a"; exit}
            m=s/n; var=(n>1)?(ss-n*m*m)/(n-1):0; if(var<0)var=0; sd=sqrt(var);
            printf "mean=%.2f  sd=%.2f  cv=%.3f%%  min=%.2f  max=%.2f  spread=%.3f%%",
                   m, sd, (m!=0?sd/m*100:0), min, max, (min>0?(max-min)/min*100:0) }'
}

echo
echo "=== stability over $K runs (N=$N transfers, distinct) ==="
echo -n "time(s)   : "; printf '%s\n' "$TIMES"  | stats; echo
echo -n "heap(MB)  : "; printf '%s\n' "$HEAPS"  | stats; echo
if [ -n "$RSSMB" ]; then echo -n "rss(MB)   : "; printf '%s\n' "$RSSMB" | stats; echo; fi
echo "=== done ==="
