#!/usr/bin/env bash
# parallel-walk.sh — can a block's windows be walked in parallel? The host-only measurement on one desktop machine
# (CPU only: the card stays idle, and the script checks it).
#
#   bash parallel-walk.sh                   preflight, clone, CPU build, the small block (a warm-up, then a run), the
#                                           median twice, the read-out, the send-back tarball (≈ 20 min)
#   bash parallel-walk.sh --quick           the small block once and the median once at 8 samples (≈ 12 min)
#   bash parallel-walk.sh --preflight-only  the checks alone (≈ 1 min)
#   options: --dir DIR (work directory, default ~/mauro/lambda-vm-parallel-walk) · --skip-build (reuse the build of an
#            earlier run in DIR) · --dry-run (print the runs' commands; run nothing)
#
# What runs: the ignored test tables::trace_builder::windowed::parallel_walk_tests::the_walk_in_parallel_from_window_states
# at PW_REV (the commit this script came from; it refuses to run if its own bytes differ from that commit's), built
# without CUDA (CUDA_HOME points at an empty directory, so no kernel is compiled), each run under env -i with the
# production allocator posture. Blocks: 25368371 (small) and 25475471 (median), the fixtures pinned by sha256; not the
# large block (≈ 85 GB at the box's sample count). Samples a median run keeps: 16 (8 with --quick).
# Before each run the card must be idle and ≥ 46 GiB available; during it the card's memory and compute processes
# are sampled once a second.
#
# What leaves the machine: only DIR/parallel-walk-send-<stamp>.tar.gz, plain text (each run's out.log, the card's and
# the process's memory readings, the read-out, the preflight), scrubbed of DIR and HOME and scanned for credential
# markers before it is written.
# Exit: 0 done · 1 usage · 2 preflight failed · 3 setup or build failed · 4 the bundle scan refused
set -euo pipefail

# ------------------------------------------------------------------------------------------------ pins
REPO_URL="${PW_REPO_URL:-https://github.com/yetanotherco/lambda_vm.git}"
BRANCH=prof/itrace-mauro
FIXTURE_BRANCH=prof/iprof4-5090
FIXTURE_REV=df03191fea9a6cc9236354817b63390777e65fb1
ELF_NAME=ethrex_8f826601.elf
ELF_SHA256=8f826601776d4085cbb6fbf0302fe8d8d5d1be7940ac1aaca24899c6244ec80a
SMALL_NAME=ethrex_mainnet_25368371_573004e6.bin
SMALL_SHA256=573004e62e3680a00d3cdbae19dc4897e2ec60d6ec0c1d05d9ef118cb8aef17f
MED_NAME=ethrex_mainnet_25475471_8140dd6d.bin
MED_SHA256=8140dd6dc606759310d1fc4b0655d426ef6eab9467654468d71e41f27704bbd7
RUST_TOOLCHAIN=1.94.0
TEST=tables::trace_builder::windowed::parallel_walk_tests::the_walk_in_parallel_from_window_states
SECRET_MARKERS='TOKEN=|_TOKEN|API_KEY|_KEY=|SECRET|PASSWORD|BEGIN .*PRIVATE|ssh-(rsa|ed25519|dss)|ghp_|github_pat_|LS0tLS1CRUdJT|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20}'

# ------------------------------------------------------------------------------------------------ args
MODE=full; PREFLIGHT_ONLY=0; SKIP_BUILD=0; DRY=0; W="${PW_DIR:-$HOME/mauro/lambda-vm-parallel-walk}"
usage() { sed -n '2,26p' "${BASH_SOURCE[0]}"; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --quick) MODE=quick ;;
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --dry-run) DRY=1 ;;
    --dir) W="${2:?}"; shift ;;
    -h|--help) usage ;;
    *) echo "parallel-walk: unknown argument $1"; usage ;;
  esac
  shift
done
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$W"
W="$(cd "$W" && pwd)"
RUN="$W/runs/$STAMP"; SEND="$RUN/send"
mkdir -p "$SEND/runs" "$W/home" "$W/tmp" "$W/cargo-home" "$W/fixtures" "$W/no-cuda"
LOG="$SEND/parallel-walk.log"
say() { echo "PW $(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }
die() { local rc=$1; shift; say "STOP ($rc): $*"; echo "parallel-walk: stopped: $*  (log: $LOG)"; exit "$rc"; }
step() { printf '%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "${2:-}" >> "$SEND/steps.tsv"; say "== $1 ${2:-}"; }

NVSMI="$(command -v nvidia-smi 2>/dev/null || true)"
CARGO_BIN="$(dirname "$(command -v cargo 2>/dev/null || echo "$HOME/.cargo/bin/cargo")")"
RUSTUP_HOME_REAL="${RUSTUP_HOME:-$HOME/.rustup}"
BASE_PATH="$CARGO_BIN:/usr/local/bin:/usr/bin:/bin"
BUILD_ENV=("PATH=$BASE_PATH" "HOME=$W/home" "RUSTUP_HOME=$RUSTUP_HOME_REAL" "RUSTUP_TOOLCHAIN=$RUST_TOOLCHAIN"
           "CARGO_HOME=$W/cargo-home" "CARGO_TARGET_DIR=$W/target" "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C"
           "CUDA_HOME=$W/no-cuda" "CUDA_PATH=$W/no-cuda")
RUN_ENV=("PATH=$BASE_PATH" "HOME=$W/home" "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C" "CUDA_VISIBLE_DEVICES=")

card_used() { [ -z "$NVSMI" ] || "$NVSMI" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s += $1} END {print s + 0}'; }
card_apps() { [ -z "$NVSMI" ] || "$NVSMI" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true; }
card_idle() { [ -z "$NVSMI" ] && return 0; [ "$(card_used)" -lt "${PW_IDLE_MIB:-1024}" ] && [ "$(card_apps)" -eq 0 ]; }
avail_gib() { awk '/^MemAvailable:/ {printf "%d", $2 / 1048576}' /proc/meminfo; }
wait_ready() {  # the card idle and ≥ 46 GiB available, up to 5 min
  local t=0
  while :; do
    if card_idle && [ "$(avail_gib)" -ge "${PW_MIN_AVAIL_GIB:-46}" ]; then return 0; fi
    [ "$t" -lt 300 ] || { say "$1: not ready after 5 min (card $(card_used) MiB, $(card_apps) compute processes; $(avail_gib) GiB available)"; return 1; }
    sleep 10; t=$((t + 10))
  done
}
bounded() {  # bounded SECS OUT cmd...: own process group; past SECS: INT, 30 s, KILL to the group; status or 124
  local secs="$1" out="$2" pid t=0 rc
  shift 2
  set -m
  "$@" < /dev/null > "$out" 2>&1 &
  pid=$!
  set +m
  [ -z "${ON_START:-}" ] || "$ON_START" "$pid"
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$t" -ge "$secs" ]; then
      say "timeout after ${secs} s: stopping process group $pid"
      kill -INT -- "-$pid" 2>/dev/null || true
      for _ in $(seq 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
      kill -KILL -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 1; t=$((t + 1))
  done
  rc=0; wait "$pid" || rc=$?
  return "$rc"
}

# ------------------------------------------------------------------------------------------------ preflight
NFAIL=0
chk() { printf '%s %s\n' "$1" "$2" | tee -a "$SEND/preflight.txt" >> "$LOG"; echo "  $1 $2"; [ "$1" != FAIL ] || NFAIL=$((NFAIL + 1)); }
preflight() {
  step preflight
  : > "$SEND/preflight.txt"
  [ "$(uname -s)" = Linux ] && chk PASS "host: Linux · $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")" || chk FAIL "host: Linux needed"
  chk INFO "cpu: $(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo) · $(nproc) threads"
  for t in git tar timeout awk sed python3 sha256sum; do command -v "$t" > /dev/null || chk FAIL "missing tool: $t"; done
  python3 -c 'import sys; assert sys.version_info >= (3, 8)' 2>/dev/null && chk PASS "python3 $(python3 -c 'import sys; print(sys.version.split()[0])')" \
    || chk FAIL "python3 >= 3.8 needed"
  if [ -n "$NVSMI" ]; then
    card_idle && chk PASS "card: idle ($(card_used) MiB used, no compute process); this script does not use it" \
      || chk FAIL "card: busy ($(card_used) MiB used, $(card_apps) compute processes): close it and re-run (the measurement needs a quiet host)"
  else chk INFO "card: no nvidia-smi (a CPU host): fine, the card is not used"; fi
  local mt av fr
  mt="$(awk '/^MemTotal:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"; av="$(awk '/^MemAvailable:/ {printf "%.1f", $2 / 1048576}' /proc/meminfo)"
  awk -v m="$mt" 'BEGIN { exit !(m >= 56) }' && chk PASS "ram: MemTotal $mt GiB, $av available" || chk FAIL "ram: MemTotal $mt GiB; the median needs >= 56 GiB"
  awk -v a="$av" 'BEGIN { exit !(a >= 46) }' || chk FAIL "ram: only $av GiB available; close other programs (>= 46 needed before each run)"
  fr="$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)"
  [ "${fr:-0}" -ge 25 ] && chk PASS "disk: $fr GiB free in the work directory" || chk FAIL "disk: $fr GiB free in $W; need >= 25 (the build ≈ 10)"
  if env RUSTUP_HOME="$RUSTUP_HOME_REAL" "$CARGO_BIN/rustup" toolchain list 2>/dev/null | grep "^$RUST_TOOLCHAIN" > /dev/null; then
    chk PASS "rust: toolchain $RUST_TOOLCHAIN installed"
  else chk FAIL "rust: toolchain $RUST_TOOLCHAIN missing: rustup toolchain install $RUST_TOOLCHAIN"; fi
}

# ------------------------------------------------------------------------------------------------ setup and build
setup() {
  step setup "clone / fetch, the reader and the fixtures"
  local R="$W/repo" rev f
  if [ ! -d "$R/.git" ]; then git clone -q --no-checkout --filter=blob:none "$REPO_URL" "$R" || die 3 "git clone $REPO_URL failed"; fi
  git -C "$R" fetch -q origin "$BRANCH" "$FIXTURE_BRANCH" || die 3 "git fetch of $BRANCH and $FIXTURE_BRANCH failed"
  git -C "$R" cat-file -e "$FIXTURE_REV^{commit}" 2>/dev/null || die 3 "the fixtures' commit $FIXTURE_REV is not on origin"
  rev="${PW_REV:-$(git -C "$R" rev-parse "origin/$BRANCH")}"
  git -C "$R" cat-file -e "$rev:scripts/profile/parallel-walk.sh" 2>/dev/null || die 3 "no scripts/profile/parallel-walk.sh at $rev"
  [ "$(git -C "$R" show "$rev:scripts/profile/parallel-walk.sh" | sha256sum | cut -c1-64)" = "$(sha256sum "$SCRIPT" | cut -c1-64)" ] \
    || die 3 "this parallel-walk.sh differs from the one at $rev: download the script again, or set PW_REV to the commit it came from"
  PW_REV="$rev"
  git -C "$R" show "$rev:scripts/profile/parallel_walk_read.py" > "$W/parallel_walk_read.py"
  for f in "$ELF_NAME:$ELF_SHA256" "$SMALL_NAME:$SMALL_SHA256" "$MED_NAME:$MED_SHA256"; do
    git -C "$R" show "$FIXTURE_REV:scripts/profile/fixtures/${f%%:*}" > "$W/fixtures/${f%%:*}"
    [ "$(sha256sum "$W/fixtures/${f%%:*}" | cut -c1-64)" = "${f##*:}" ] || die 3 "fixture ${f%%:*}: sha256 mismatch"
  done
  local wt="$W/wt"
  if [ -e "$wt/.git" ]; then git -C "$wt" checkout -q --detach "$PW_REV"; else git -C "$R" worktree add -q --detach "$wt" "$PW_REV"; fi
  [ "$(git -C "$wt" rev-parse HEAD)" = "$PW_REV" ] || die 3 "the worktree is not at $PW_REV"
  [ -z "$(git -C "$wt" status --porcelain --untracked-files=no)" ] || die 3 "the worktree has local changes"
  grep -q 'fn the_walk_in_parallel_from_window_states' "$wt/prover/src/tables/trace_builder/windowed/parallel_walk_tests.rs" \
    || die 3 "the harness is not in the tree at $PW_REV"
  say "PW_REV $PW_REV · fixtures from $FIXTURE_REV, sha256 checked"
}
build() {
  local rel=""
  step build "cargo test --release -p lambda-vm-prover --lib --no-run (no CUDA)"
  if [ "$SKIP_BUILD" = 0 ]; then
    (cd "$W/wt" && bounded 5400 "$SEND/build.log" env -i "${BUILD_ENV[@]}" cargo test --release -p lambda-vm-prover --lib --no-run) \
      || { tail -30 "$SEND/build.log"; die 3 "the build failed: $SEND/build.log"; }
    rel="$({ sed -n 's/.*Executable unittests src\/lib.rs (\(.*lambda_vm_prover-[0-9a-f]*\)).*/\1/p' "$SEND/build.log" || true; } | tail -1)"
  fi
  if [ -z "$rel" ]; then
    local f
    for f in "$W/target"/release/deps/lambda_vm_prover-*; do
      [ -f "$f" ] && [ -x "$f" ] || continue
      if [ -z "$rel" ] || [ "$f" -nt "$rel" ]; then rel="$f"; fi
    done
  fi
  case "$rel" in /*|"") ;; *) rel="$W/wt/$rel" ;; esac
  [ -n "$rel" ] && [ -x "$rel" ] || die 3 "no test binary in $W/target"
  [ -z "$(git -C "$W/wt" status --porcelain --untracked-files=no)" ] || die 3 "the worktree changed during the build"
  BIN="$rel"
  say "binary $(basename "$BIN") sha256 $(sha256sum "$BIN" | cut -c1-16)…"
}

# ------------------------------------------------------------------------------------------------ the runs
sample() {  # sample PID RD: the card once a second, the process's VmHWM at its end
  local pid=$1 rd=$2 base used apps most=0 seen=0 hwm=0 h
  base="$(card_used)"
  while kill -0 "$pid" 2>/dev/null; do
    if [ -n "$NVSMI" ]; then
      used="$(card_used)"; apps="$(card_apps)"
      echo "$(date +%s) $used $apps" >> "$rd/card.tsv"
      [ "${used:-0}" -le "$most" ] || most="$used"
      [ "${apps:-0}" -eq 0 ] || seen=$((seen + 1))
    fi
    h="$(awk '/^VmHWM:/ {print $2}' "/proc/$pid/status" 2>/dev/null || true)"
    [ -z "$h" ] || hwm="$h"
    sleep 1
  done
  if [ -n "$NVSMI" ]; then echo "card before ${base:-0} MiB · most $most MiB · rise $(( most - ${base:-0} )) MiB · compute processes seen $seen" > "$rd/card.txt"
  else echo "no nvidia-smi: the card is not read" > "$rd/card.txt"; fi
  awk -v k="$hwm" 'BEGIN { printf "peak VmHWM %.2f GiB\n", k / 1048576 }' > "$rd/host.txt"
}
run_one() {  # run_one NAME FIXTURE SAMPLES THREADS TIMEOUT
  local name=$1 fix=$2 samples=$3 threads=$4 to=$5 rd words
  rd="$RUN/$name"; mkdir -p "$rd"
  words=("_RJEM_MALLOC_CONF=dirty_decay_ms:-1,muzzy_decay_ms:-1" "BLOCK_WHIR_ELF=$W/fixtures/$ELF_NAME" "BLOCK_WHIR_INPUT=$W/fixtures/$fix"
         "PARALLEL_WALK_SAMPLES=$samples" "PARALLEL_WALK_THREADS=$threads" "PARALLEL_WALK_BESIDE=8" "PARALLEL_WALK_GENERATORS=6")
  printf '%s\n' "${words[*]}" | sed "s#$W#<W>#g" > "$rd/knobs.txt"
  if [ "$DRY" = 1 ]; then say "DRY $name: env -i ${RUN_ENV[*]} ${words[*]} <bin> --ignored --exact --nocapture --test-threads=1 $TEST" ; return 0; fi
  wait_ready "$name" || { echo 97 > "$rd/rc.txt"; return 0; }
  step "run $name" "$fix · samples $samples · threads $threads"
  ON_START=start_sampler
  RD_NOW="$rd"
  SAMPLER_PID=
  cd "$W/wt/prover"
  set +e
  bounded "$to" "$rd/out.log" env -i "${RUN_ENV[@]}" "${words[@]}" "$BIN" --ignored --exact --nocapture --test-threads=1 "$TEST"
  local rc=$?
  set -e
  cd "$W"
  ON_START=
  [ -z "$SAMPLER_PID" ] || wait "$SAMPLER_PID" 2>/dev/null || true
  echo "$rc" > "$rd/rc.txt"
  say "$name: rc $rc · $(grep -hoE 'PARALLEL WALK DONE: .{0,160}' "$rd/out.log" | head -1) · $(cat "$rd/card.txt" 2>/dev/null) · $(cat "$rd/host.txt" 2>/dev/null)"
}
start_sampler() { sample "$1" "$RD_NOW" & SAMPLER_PID=$!; }

# ------------------------------------------------------------------------------------------------ main
preflight
if [ "$NFAIL" -gt 0 ]; then die 2 "$NFAIL preflight check(s) failed (see above)"; fi
[ "$PREFLIGHT_ONLY" = 0 ] || { say "preflight passed"; exit 0; }
setup
build
if [ "$MODE" = quick ]; then
  PLAN=("S-I:$SMALL_NAME:16:1,2,4,8:900" "S-M1:$MED_NAME:08:1,2,4,8:1800")
else
  PLAN=("W-I:$SMALL_NAME:16:1,2,4,8,12,16:900" "S-I:$SMALL_NAME:16:1,2,4,8,12,16:900"
        "S-M1:$MED_NAME:16:1,2,4,8,12,16:1800" "S-M2:$MED_NAME:16:1,2,4,8,12,16:1800")
fi
EXPECTED=""
for p in "${PLAN[@]}"; do
  IFS=: read -r name fix samples threads to <<< "$p"
  run_one "$name" "$fix" "$samples" "$threads" "$to"
  case "$name" in W-*) ;; *) EXPECTED="${EXPECTED:+$EXPECTED,}$name" ;; esac
done
[ "$DRY" = 0 ] || { say "dry run done"; exit 0; }
step readout
python3 "$W/parallel_walk_read.py" "$RUN" "$EXPECTED" > "$SEND/readout.txt" 2>&1 || say "the read-out failed: see $SEND/readout.txt"
grep '^VERDICT' "$SEND/readout.txt" | tee -a "$LOG" || true
# The send-back: plain text only, scrubbed of the work directory and the home directory.
for d in "$RUN"/*/; do
  n="$(basename "$d")"; [ "$n" = send ] && continue
  mkdir -p "$SEND/runs/$n"
  for f in out.log knobs.txt card.txt card.tsv host.txt rc.txt; do
    [ ! -f "$d/$f" ] || sed "s#$W#<W>#g; s#$HOME#<HOME>#g" "$d/$f" > "$SEND/runs/$n/$f"
  done
done
sed -i "s#$W#<W>#g; s#$HOME#<HOME>#g" "$SEND"/*.txt "$SEND"/*.log "$SEND"/*.tsv 2>/dev/null || true
{
  echo "rev $PW_REV · mode $MODE · stamp $STAMP"
  echo "cpu $(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo) · $(nproc) threads · MemTotal $(awk '/^MemTotal:/ {printf "%.1f GiB", $2 / 1048576}' /proc/meminfo)"
  echo "kernel $(uname -r) · $(. /etc/os-release 2>/dev/null; echo "${PRETTY_NAME:-?}")"
  [ -z "$NVSMI" ] || "$NVSMI" --query-gpu=name,driver_version,power.limit --format=csv,noheader 2>/dev/null
} > "$SEND/facts.txt"
if grep -rEl "$SECRET_MARKERS" "$SEND" > "$RUN/scan.txt" 2>/dev/null; then
  die 4 "the bundle scan found credential markers in: $(tr '\n' ' ' < "$RUN/scan.txt") — nothing written"
fi
OUTTGZ="$W/parallel-walk-send-$STAMP.tar.gz"
tar -C "$RUN" -czf "$OUTTGZ" send
say "send back: $OUTTGZ ($(du -h "$OUTTGZ" | cut -f1))"
echo "parallel-walk: done. Send back $OUTTGZ"
