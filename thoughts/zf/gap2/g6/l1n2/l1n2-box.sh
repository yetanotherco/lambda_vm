#!/usr/bin/env bash
# L1N2-NOISE job — twelve identical default arms of #1010's production tree at land/1010-whole-trees 7c8272701 (#1010's
# next head: keep-futile + whole trees, gated FAST2 166), A x12, wt<N>..wt<N+11>, through the lead's harness
# (zf-whir-arms.sh, md5 86951bc2), with three samplers running for the whole harness run:
#   g6_sampler.py (md5 56eff730, G6's) — the prover process's per-thread on-CPU and runqueue wait, cgroup CPU and
#     CFS throttling, host busy (0.05 s);
#   sys_sampler.py (md5 pinned) — CPU MHz, Tctl/Tccd, PSI cpu/memory/io, loadavg, meminfo, vmstat, and every
#     container process's CPU by short name (0.25 s; processes every 1 s);
#   nvidia-smi — SM clock, temperature, power (0.5 s).
# Then l1n2_readout.py (md5 pinned): which signal, if any, separates the arms whose level 1 runs slow. Pre-registered
# in G6-LEDGER.md §9 (fix2/1010-g6-census), before the run.
#
# usage: bash l1n2-box.sh <first-wt-number>     (lane tags: 1302 → wt1302..wt1313)
# Output: /root/zf/l1n2-wt<N>/{l1n2.log,readout.txt} and the raw sampler files next to them (they stay on the box).
# Exit: 0 done · 1 usage · 4 git/HEAD · 5 preflight · 9 card not idle · 11 harness red · 16 a sampler died at start.
set -euo pipefail

EXPECT_HEAD=7c8272701c1d664e379d24fb892669f1349bc3bb
BRANCH=land/1010-whole-trees
HARNESS=/root/zf/bin/zf-whir-arms.sh
HARNESS_MD5=86951bc260eafb8461055611b501665c
SUMMARY_MD5=a0228cf0dcf71b26cf77a658d830382b
G6_SAMPLER=/root/zf/g6/g6_sampler.py
G6_SAMPLER_MD5=56eff730175a96591c7546c62fd33caf
PREFIX=wt
[ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]] || { sed -n '2,15p' "${BASH_SOURCE[0]}"; exit 1; }
N0="$1"
SHA9=${EXPECT_HEAD:0:9}
SRC=/workspace/lambda_vm
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYS_SAMPLER="$HERE/sys_sampler.py"
SYS_SAMPLER_MD5=1a9c9163e0272ed47bde40003eef34a6
READOUT="$HERE/l1n2_readout.py"
READOUT_MD5=171e2f145e2f700d6cab0fde5add7274
MATCH=/workspace/lambda_vm-zf-whir/target/release/deps/lambda_vm_prover-
ARMS=(A:- A:- A:- A:- A:- A:- A:- A:- A:- A:- A:- A:-)
LAST=$((N0 + ${#ARMS[@]} - 1))
RUN="/root/zf/${PREFIX}${N0}-${PREFIX}${LAST}"
OUT="/root/zf/l1n2-wt${N0}"
[ ! -e "$OUT" ] || { echo "VERDICT: REFUSED — $OUT exists (tag used)"; exit 1; }
mkdir -p "$OUT"
LOG="$OUT/l1n2.log"
say() { echo "L1N2 $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
refuse() { local rc=$1; shift; say "REFUSE($rc): $*"; echo "VERDICT: REFUSED rc=$rc — $*" | tee -a "$LOG"; exit "$rc"; }
PIDS=()
stop_samplers() {
  local p
  for p in "${PIDS[@]}"; do kill -TERM "$p" 2>/dev/null || true; done
  for p in "${PIDS[@]}"; do wait "$p" 2>/dev/null || true; done
  PIDS=()
}
trap stop_samplers EXIT

say "L1N2-NOISE job · expect $EXPECT_HEAD ($BRANCH) · ${#ARMS[@]} identical arms · tags $PREFIX$N0..$PREFIX$LAST"
used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | { grep -c . || true; })"
say "card: ${used} MiB used, ${apps} compute app(s)"
[ "${used:-99999}" -lt 500 ] && [ "${apps:-1}" -eq 0 ] || refuse 9 "card not idle (${used} MiB, ${apps} apps)"

# ---- preflight
chk() { [ -r "$1" ] && [ "$(md5sum "$1" | cut -c1-32)" = "$2" ] || refuse 5 "$1 missing or not md5 $2"; }
chk "$HARNESS" "$HARNESS_MD5"
chk "$(dirname "$HARNESS")/zf_summary.py" "$SUMMARY_MD5"
chk "$G6_SAMPLER" "$G6_SAMPLER_MD5"
chk "$SYS_SAMPLER" "$SYS_SAMPLER_MD5"
chk "$READOUT" "$READOUT_MD5"
python3 "$SYS_SAMPLER" --selftest >> "$LOG" 2>&1 || refuse 5 "sys_sampler selftest failed"
python3 "$READOUT" --selftest > "$OUT/readout-selftest.txt" 2>&1 || refuse 5 "readout selftest failed"
git -C "$SRC" fetch -q origin "$BRANCH" || refuse 4 "git fetch origin $BRANCH failed"
full="$(git -C "$SRC" rev-parse --verify -q "${SHA9}^{commit}")" || refuse 4 "$SHA9 is not a commit after the fetch"
[ "$full" = "$EXPECT_HEAD" ] || refuse 4 "$SHA9 resolves to $full, expected $EXPECT_HEAD"
say "HEAD asserted: $full"

# ---- samplers, for the whole harness run
python3 "$G6_SAMPLER" --out "$OUT/g6.tsv" --match "$MATCH" --interval 0.05 > "$OUT/g6.err" 2>&1 < /dev/null &
PIDS+=($!)
python3 "$SYS_SAMPLER" --out "$OUT/sys.tsv" --interval 0.25 --procs-every 4 > "$OUT/sys.err" 2>&1 < /dev/null &
PIDS+=($!)
TZ=UTC nvidia-smi --query-gpu=timestamp,clocks.sm,temperature.gpu,power.draw --format=csv,noheader -lms 500 \
  > "$OUT/gpu.csv" 2> "$OUT/gpu.err" < /dev/null &
PIDS+=($!)
sleep 1
for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null || refuse 16 "sampler pid $p died at start (see $OUT/*.err)"; done
say "samplers: pids ${PIDS[*]}"

# ---- the arms (the harness takes /root/zf/.zf.lock and checks the card before each arm)
say "harness: ZF_BRANCH=$BRANCH bash $HARNESS $SHA9 $N0 ${ARMS[*]}"
set +e
ZF_BRANCH="$BRANCH" bash "$HARNESS" "$SHA9" "$N0" "${ARMS[@]}" 2>&1 | tee -a "$LOG"
HRC=${PIPESTATUS[0]}
set -e
say "harness returned $HRC"
stop_samplers
say "samplers stopped: g6 $({ grep -c '^T' "$OUT/g6.tsv" || true; }) thread samples · sys $({ grep -c '^S' "$OUT/sys.tsv" || true; }) · gpu $({ grep -c . "$OUT/gpu.csv" || true; })"

READ_LINE="READOUT: not run (no manifest)"
if [ -r "$RUN/manifest.tsv" ]; then
  set +e
  python3 "$READOUT" --manifest "$RUN/manifest.tsv" --g6 "$OUT/g6.tsv" --sys "$OUT/sys.tsv" --gpu "$OUT/gpu.csv" \
    > "$OUT/readout.txt" 2>&1
  set -e
  tee -a "$LOG" < "$OUT/readout.txt"
  READ_LINE="$(grep '^READOUT: ' "$OUT/readout.txt" | tail -1 || true)"
  [ -n "$READ_LINE" ] || READ_LINE="READOUT: no verdict line (see $OUT/readout.txt)"
fi
if [ "$HRC" -ne 0 ]; then
  echo "VERDICT: HARNESS RED rc=$HRC · ${READ_LINE#READOUT: }" | tee -a "$LOG"
  exit 11
fi
echo "VERDICT: ${READ_LINE#READOUT: }" | tee -a "$LOG"
