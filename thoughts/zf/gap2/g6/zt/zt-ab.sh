#!/usr/bin/env bash
# ZERO-TAIL job — two parts, in order, at fix2/1010-zerotail d9aab005a (#1010's keep-futile head 73342bc66 + the
# zero-tail upload, default off, + the pinned-vs-pageable microbench). Pre-registered in G6-LEDGER.md §8
# (fix2/1010-g6-census), before the run.
#   1. D-TRACE's box request 1: crypto/math-cuda/tests/h2d_pinned_bench.rs (pure DMA from cuMemHostAlloc memory vs the
#      pageable memcpy_htod, at column sizes and over epoch 0's 393 columns), in its own worktree
#      /workspace/lambda_vm-zf-ztbench. Stop rule: best pinned >= 1.3x pageable over the epoch mix, or stage 1b is not
#      worth building. A failure here is recorded and the arms still run.
#   2. The A/B: A = default, B = LAMBDA_VM_TRACE_UPLOAD=zerotail, A B B A A B B A, through the lead's harness
#      (zf-whir-arms.sh, md5 86951bc2) with ZF_ALLOW_OTHER_KNOBS=1, then zt_readout.py (md5 pinned) over its manifest.
#
# usage: bash zt-ab.sh <first-wt-number>        (lane tags: 1294 → wt1294..wt1301)
# Output: /root/zf/zt-wt<N>/{zt.log,microbench.log,readout.txt}; the harness's run is /root/zf/wt<N>-wt<N+7>/.
# Exit: 0 green · 1 usage · 4 git/HEAD · 5 preflight · 9 card not idle · 11 harness red.
set -euo pipefail

EXPECT_HEAD=d9aab005ac8ef7df5cdb07d9152451c3bf5a851b
BRANCH=fix2/1010-zerotail
HARNESS=/root/zf/bin/zf-whir-arms.sh
HARNESS_MD5=86951bc260eafb8461055611b501665c
PREFIX=wt
[ $# -ge 1 ] && [[ "$1" =~ ^[0-9]+$ ]] || { sed -n '2,14p' "${BASH_SOURCE[0]}"; exit 1; }
N0="$1"
SHA9=${EXPECT_HEAD:0:9}
SRC=/workspace/lambda_vm
BENCH_WT=/workspace/lambda_vm-zf-ztbench
SUMMARY_MD5=a0228cf0dcf71b26cf77a658d830382b
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
READOUT="$HERE/zt_readout.py"
READOUT_MD5=0c4755093707a6da8dad27cde247c52a
ARMS=(A:- B:LAMBDA_VM_TRACE_UPLOAD=zerotail B:LAMBDA_VM_TRACE_UPLOAD=zerotail A:- A:- B:LAMBDA_VM_TRACE_UPLOAD=zerotail B:LAMBDA_VM_TRACE_UPLOAD=zerotail A:-)
LAST=$((N0 + ${#ARMS[@]} - 1))
RUN="/root/zf/${PREFIX}${N0}-${PREFIX}${LAST}"
OUT="/root/zf/zt-wt${N0}"
[ ! -e "$OUT" ] || { echo "VERDICT: REFUSED — $OUT exists (tag used)"; exit 1; }
mkdir -p "$OUT"
LOG="$OUT/zt.log"
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
say() { echo "ZT $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
refuse() { local rc=$1; shift; say "REFUSE($rc): $*"; echo "VERDICT: REFUSED rc=$rc — $*" | tee -a "$LOG"; exit "$rc"; }
card_idle() {
  local used apps
  used="$(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | { grep -c . || true; })"
  say "card: ${used} MiB used, ${apps} compute app(s)"
  [ "${used:-99999}" -lt 500 ] && [ "${apps:-1}" -eq 0 ]
}

say "ZERO-TAIL job · expect $EXPECT_HEAD ($BRANCH) · arms ${ARMS[*]} · tags $PREFIX$N0..$PREFIX$LAST"
card_idle || refuse 9 "card not idle"

# ---- preflight: the pinned harness, its summary, the readout, and the sha on the branch
[ -r "$HARNESS" ] && [ "$(md5sum "$HARNESS" | cut -c1-32)" = "$HARNESS_MD5" ] \
  || refuse 5 "$HARNESS missing or not md5 $HARNESS_MD5"
[ "$(md5sum "$(dirname "$HARNESS")/zf_summary.py" | cut -c1-32)" = "$SUMMARY_MD5" ] \
  || refuse 5 "zf_summary.py next to the harness is not md5 $SUMMARY_MD5"
[ -r "$READOUT" ] && [ "$(md5sum "$READOUT" | cut -c1-32)" = "$READOUT_MD5" ] \
  || refuse 5 "$READOUT missing or not md5 $READOUT_MD5"
python3 -c 'import sys; assert sys.version_info >= (3, 8)' || refuse 5 "python3 >= 3.8 needed"
git -C "$SRC" fetch -q origin "$BRANCH" || refuse 4 "git fetch origin $BRANCH failed"
full="$(git -C "$SRC" rev-parse --verify -q "${SHA9}^{commit}")" || refuse 4 "$SHA9 is not a commit after the fetch"
[ "$full" = "$EXPECT_HEAD" ] || refuse 4 "$SHA9 resolves to $full, expected $EXPECT_HEAD"
say "HEAD asserted: $full"

# ---- part 1: the microbench, in its own worktree
MB_LINE="MICROBENCH: not run"
if [ -d "$BENCH_WT" ]; then
  git -C "$BENCH_WT" checkout -q --detach "$EXPECT_HEAD" || refuse 4 "checkout in $BENCH_WT failed"
else
  git -C "$SRC" worktree add -q --detach "$BENCH_WT" "$EXPECT_HEAD" || refuse 4 "worktree add $BENCH_WT failed"
fi
[ "$(git -C "$BENCH_WT" rev-parse HEAD)" = "$EXPECT_HEAD" ] || refuse 4 "$BENCH_WT HEAD is not $EXPECT_HEAD"
card_idle || refuse 9 "card not idle before the microbench"
say "microbench: cargo test --release -p math-cuda --test h2d_pinned_bench -- --ignored --nocapture (in $BENCH_WT)"
set +e
( cd "$BENCH_WT" && timeout -k 30 2400 cargo test --release -p math-cuda --test h2d_pinned_bench -- --ignored --nocapture ) > "$OUT/microbench.log" 2>&1
MRC=$?
set -e
if [ "$MRC" -eq 0 ] && grep -q '^MICROBENCH: ' "$OUT/microbench.log"; then
  sed -n '/^pinned vs pageable/,/^MICROBENCH: /p' "$OUT/microbench.log" | tee -a "$LOG"
  MB_LINE="$(grep '^MICROBENCH: ' "$OUT/microbench.log" | tail -1)"
else
  MB_LINE="MICROBENCH: FAILED rc=$MRC (see $OUT/microbench.log)"
  tail -20 "$OUT/microbench.log" | tee -a "$LOG"
fi
say "$MB_LINE"

# ---- part 2: the arms, through the harness (it takes /root/zf/.zf.lock and checks the card before each arm)
card_idle || refuse 9 "card not idle before the arms"
unset LAMBDA_VM_TRACE_UPLOAD
say "harness: ZF_BRANCH=$BRANCH ZF_ALLOW_OTHER_KNOBS=1 bash $HARNESS $SHA9 $N0 ${ARMS[*]}"
set +e
ZF_BRANCH="$BRANCH" ZF_ALLOW_OTHER_KNOBS=1 bash "$HARNESS" "$SHA9" "$N0" "${ARMS[@]}" 2>&1 | tee -a "$LOG"
HRC=${PIPESTATUS[0]}
set -e
say "harness returned $HRC"

READ_LINE="READOUT: not run (no manifest)"
if [ -r "$RUN/manifest.tsv" ]; then
  set +e
  python3 "$READOUT" "$RUN/manifest.tsv" > "$OUT/readout.txt" 2>&1
  RRC=$?
  set -e
  tee -a "$LOG" < "$OUT/readout.txt"
  READ_LINE="$(grep '^READOUT: ' "$OUT/readout.txt" | tail -1 || true)"
  [ -n "$READ_LINE" ] || READ_LINE="READOUT: rc=$RRC, no verdict line (see $OUT/readout.txt)"
fi
if [ "$HRC" -ne 0 ]; then
  echo "VERDICT: HARNESS RED rc=$HRC · ${READ_LINE#READOUT: } · ${MB_LINE}" | tee -a "$LOG"
  exit 11
fi
echo "VERDICT: ${READ_LINE#READOUT: } · ${MB_LINE}" | tee -a "$LOG"
