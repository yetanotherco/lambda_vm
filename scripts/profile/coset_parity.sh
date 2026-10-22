#!/usr/bin/env bash
# coset_parity.sh — the coset round-0 leaf order's card parity on one GPU machine: builds three test binaries at a
# pinned commit and runs them, then writes a send-back tarball of derived text.
#
#   bash coset_parity.sh                   preflight, clone, builds, the three tests, the summary, the send-back tarball
#   bash coset_parity.sh --preflight-only  the checks alone (≈ 1 min); nothing is cloned, built or run
#   options: --dir DIR (work directory, default ~/mauro/lambda-vm-coset) · --gpu N (default 0) · --skip-build (reuse
#            the builds of an earlier run in the same DIR)
#
# Pinned: the build is BUILD_SHA (the script refuses any other), fetched from the branch this script lives on. Tests:
#   multilinear tests/whir_coset_device.rs (cuda), every test, one at a time — registered:
#     a_coset_chain_on_the_card_proves_the_host_chains_bytes        a coset chain committed on the card proves the bytes
#                                                                   of the same chain over a host tree (whose successor
#                                                                   can only be folded): host fold = card encode
#     a_revived_coset_stack_proves_the_whole_trees_bytes_from_the_stack_or_a_reencode
#     an_unencoded_successor_refuses_the_coset_prove_by_cause       the knob off, the factors on the host, the card
#                                                                   declining the encode: each refuses by its cause
#     a_revived_natural_stack_is_untouched
#   multilinear tests/whir_p1_device.rs (cuda), every test — registered: the_device_p1_chain_proves_the_host_chains_bytes,
#     the_device_p1_ext3_tree_is_the_hosts (the natural order on the card is unchanged)
#   math-cuda tests/whir_coset.rs: coset_blocks_equal_the_reencoded_codeword (three `J5 PARITY` lines, each 0 differing)
# PASS: every registered test passes (read by name: its `test <name> ... ok` line, its binary's closing `test result:
# ok. N passed; 0 failed` with N the registered count, no closing failures list) and the three parity lines read 0.
#
# What leaves the machine: only DIR/coset-send-<stamp>.tar.gz — the summary, the tests' and the builds' text output,
# the machine facts (GPU, driver, nvcc, rustc, CPU, memory) — scrubbed of DIR and HOME and scanned for credential
# markers before it is written. The builds and the tests run under env -i, with HOME, CARGO_HOME and TMPDIR inside DIR.
# Exit: 0 done (the summary's VERDICT says what) · 1 usage · 2 preflight failed · 3 clone or build failed · 4 the
# tarball scan refused (nothing to send; see the message)
set -euo pipefail

REPO_URL="${COSET_REPO_URL:-https://github.com/yetanotherco/lambda_vm.git}"
LANE_BRANCH=prof/coset-mauro
BUILD_SHA=a7c2830ba41559c23ad09743c1c8d571259e6dca
RUST_TOOLCHAIN=1.94.0
SECRET_MARKERS='TOKEN=|_TOKEN|API_KEY|_KEY=|SECRET|PASSWORD|BEGIN .*PRIVATE|ssh-(rsa|ed25519|dss)|ghp_|github_pat_|LS0tLS1CRUdJT|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{20}'

PREFLIGHT_ONLY=0; SKIP_BUILD=0; GPU=0; W="${COSET_DIR:-$HOME/mauro/lambda-vm-coset}"
usage() { sed -n '2,31p' "${BASH_SOURCE[0]}"; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --preflight-only) PREFLIGHT_ONLY=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --gpu) GPU="${2:?}"; shift ;;
    --dir) W="${2:?}"; shift ;;
    -h|--help) usage ;;
    *) echo "coset_parity: unknown argument $1"; usage ;;
  esac
  shift
done
[[ "$GPU" =~ ^[0-9]+$ ]] || usage
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$W"
W="$(cd "$W" && pwd)"
RUN="$W/runs/$STAMP"; SEND="$RUN/send"
mkdir -p "$SEND" "$W/home" "$W/tmp" "$W/cargo-home"
LOG="$SEND/coset.log"
say() { echo "COSET $(date -u +%H:%M:%SZ) $*" | tee -a "$LOG"; }
die() { local rc=$1; shift; say "STOP ($rc): $*"; echo "coset_parity: stopped: $*  (log: $LOG)"; exit "$rc"; }
step() { printf '%s\t%s\n' "$(date -u +%FT%TZ)" "$1" >> "$SEND/steps.tsv"; say "== $1"; }

find_tool() {  # find_tool NAME → path or empty: PATH, then CUDA_HOME/bin, /usr/local/cuda/bin
  local n="$1" p
  p="$(command -v "$n" 2>/dev/null || true)"
  [ -n "$p" ] && { echo "$p"; return; }
  for d in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda; do
    [ -n "$d" ] && [ -x "$d/bin/$n" ] && { echo "$d/bin/$n"; return; }
  done
  return 0
}
NVSMI="$(find_tool nvidia-smi)"; NVCC="$(find_tool nvcc)"
CARGO_BIN="$(dirname "$(command -v cargo 2>/dev/null || echo "$HOME/.cargo/bin/cargo")")"
RUSTUP_HOME_REAL="${RUSTUP_HOME:-$HOME/.rustup}"
CUDA_DIR="$(dirname "$(dirname "${NVCC:-/usr/local/cuda/bin/nvcc}")")"
BASE_PATH="$CARGO_BIN:$CUDA_DIR/bin:/usr/local/bin:/usr/bin:/bin"
RUN_ENV=("PATH=$BASE_PATH" "HOME=$W/home" "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C" "CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=$GPU")
[ -z "${LD_LIBRARY_PATH:-}" ] || RUN_ENV+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")
BUILD_ENV=("PATH=$BASE_PATH" "HOME=$W/home" "RUSTUP_HOME=$RUSTUP_HOME_REAL" "RUSTUP_TOOLCHAIN=$RUST_TOOLCHAIN"
           "CARGO_HOME=$W/cargo-home" "CARGO_TARGET_DIR=$W/target" "TMPDIR=$W/tmp" "LC_ALL=C" "LANG=C"
           "CUDA_HOME=$CUDA_DIR" "CUDA_VISIBLE_DEVICES=$GPU")
[ -z "${LD_LIBRARY_PATH:-}" ] || BUILD_ENV+=("LD_LIBRARY_PATH=$LD_LIBRARY_PATH")

gpu_q() { "$NVSMI" -i "$GPU" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | sed 's/^ *//; s/ *$//'; }
gpu_idle() {  # the card's used memory under the floor (a desktop's display holds some) and no compute process on it
  local used apps
  used="$(gpu_q memory.used)"
  apps="$("$NVSMI" -i "$GPU" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)"
  [ "${used:-99999}" -lt "${COSET_IDLE_MIB:-2048}" ] && [ "${apps:-1}" -eq 0 ]
}
bounded() {  # bounded SECS OUT cmd...: its own process group; past SECS: INT, 30 s, KILL to the group; status or 124
  local secs="$1" out="$2" pid t=0 rc
  shift 2
  set -m
  "$@" < /dev/null > "$out" 2>&1 &
  pid=$!
  set +m
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$t" -ge "$secs" ]; then
      kill -INT -- "-$pid" 2>/dev/null || true
      sleep 30
      kill -KILL -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
      return 124
    fi
    sleep 1
    t=$((t + 1))
  done
  rc=0
  wait "$pid" || rc=$?
  return "$rc"
}

# ------------------------------------------------------------------------------------------------ preflight
step preflight
{
  echo "stamp $STAMP · build $BUILD_SHA · branch $LANE_BRANCH · gpu $GPU"
  echo "gpu: $(gpu_q name) · driver $(gpu_q driver_version) · power limit $(gpu_q power.limit) W · memory $(gpu_q memory.total) MiB · used $(gpu_q memory.used) MiB"
  echo "nvcc: $( { "${NVCC:-nvcc}" --version 2>/dev/null || echo none; } | tail -1)"
  echo "cargo: $( { env -i "${BUILD_ENV[@]}" cargo --version 2>/dev/null || echo none; } | head -1) · rustc: $( { env -i "${BUILD_ENV[@]}" rustc --version 2>/dev/null || echo none; } | head -1)"
  echo "cpu: $(sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo 2>/dev/null | head -1) · $(nproc 2>/dev/null) threads"
  echo "memory: $(awk '/^MemTotal:/ {printf "%.1f", $2/1048576}' /proc/meminfo) GiB total · $(awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo) GiB available"
  echo "disk: $(df -Pk "$W" | awk 'NR == 2 {printf "%.1f", $4/1048576}') GiB free under the work directory"
  echo "os: $(sed -n 's/^PRETTY_NAME=//p' /etc/os-release 2>/dev/null | tr -d '"')"
} > "$SEND/machine.txt"
cat "$SEND/machine.txt" | tee -a "$LOG"
[ -n "$NVSMI" ] || die 2 "no nvidia-smi"
[ -n "$NVCC" ] || die 2 "no nvcc (the kernels would be built as empty stubs and every card test would fail)"
[ -x "$CARGO_BIN/cargo" ] || die 2 "no cargo at $CARGO_BIN"
env -i "${BUILD_ENV[@]}" rustc --version > /dev/null 2>&1 || die 2 "rustc $RUST_TOOLCHAIN is not installed (rustup toolchain install $RUST_TOOLCHAIN)"
command -v git > /dev/null || die 2 "no git"
[ "$(df -Pk "$W" | awk 'NR == 2 {print int($4/1048576)}')" -ge 15 ] || die 2 "under 15 GiB free under $W"
[ "$(awk '/^MemAvailable:/ {print int($2/1048576)}' /proc/meminfo)" -ge 12 ] || die 2 "under 12 GiB of memory available"
gpu_idle || die 2 "the card is busy: $(gpu_q memory.used) MiB used and a compute process on it (close it and run again)"
say "preflight passed"
[ "$PREFLIGHT_ONLY" = 0 ] || { echo "coset_parity: preflight passed; nothing else run (--preflight-only)"; exit 0; }

# ------------------------------------------------------------------------------------------------ clone and build
SRC="$W/src"
if [ "$SKIP_BUILD" = 0 ]; then
  step clone
  if [ ! -d "$SRC/.git" ]; then
    git clone -q --no-checkout --filter=blob:none "$REPO_URL" "$SRC" || die 3 "git clone $REPO_URL failed"
  fi
  git -C "$SRC" fetch -q origin "$LANE_BRANCH" || die 3 "git fetch origin $LANE_BRANCH failed"
  git -C "$SRC" cat-file -e "${BUILD_SHA}^{commit}" 2>/dev/null || die 3 "$BUILD_SHA is not on origin's $LANE_BRANCH"
  git -C "$SRC" checkout -q --detach --force "$BUILD_SHA" || die 3 "checkout $BUILD_SHA failed"
  [ "$(git -C "$SRC" rev-parse HEAD)" = "$BUILD_SHA" ] || die 3 "the checkout is not at $BUILD_SHA"
  [ -z "$(git -C "$SRC" status --porcelain --untracked-files=no)" ] || die 3 "the checkout at $BUILD_SHA is dirty"
  say "checkout at $BUILD_SHA"
  step build
  bounded 3600 "$RUN/build-multilinear.log" env -i "${BUILD_ENV[@]}" bash -c \
    "cd '$SRC' && cargo test --release -p multilinear --features cuda --test whir_coset_device --test whir_p1_device --no-run" \
    || { tail -30 "$RUN/build-multilinear.log" > "$SEND/build-multilinear.tail"; die 3 "the multilinear build failed (its tail: send/build-multilinear.tail)"; }
  bounded 3600 "$RUN/build-math-cuda.log" env -i "${BUILD_ENV[@]}" bash -c \
    "cd '$SRC' && cargo test --release -p math-cuda --test whir_coset --no-run" \
    || { tail -30 "$RUN/build-math-cuda.log" > "$SEND/build-math-cuda.tail"; die 3 "the math-cuda build failed (its tail: send/build-math-cuda.tail)"; }
  cat "$RUN/build-multilinear.log" "$RUN/build-math-cuda.log" > "$W/last-build.log"
fi
tail -15 "$W/last-build.log" > "$SEND/build.tail" 2>/dev/null || true
binary() {  # binary TEST → the newest test binary the last build named for TEST
  sed -n "s|.*Executable tests/$1\.rs (\(.*/$1-[0-9a-f]*\)).*|\1|p" "$W/last-build.log" | tail -1
}
B_COSET="$(binary whir_coset_device)"; B_P1="$(binary whir_p1_device)"; B_EVAL="$(binary whir_coset)"
for b in "$B_COSET" "$B_P1" "$B_EVAL"; do
  [ -n "$b" ] || die 3 "a test binary is not named in the build log (run without --skip-build)"
  case "$b" in /*) ;; *) b="$SRC/$b" ;; esac
  [ -x "$b" ] || die 3 "no test binary at $b"
done
abs() { case "$1" in /*) echo "$1" ;; *) echo "$SRC/$1" ;; esac; }
B_COSET="$(abs "$B_COSET")"; B_P1="$(abs "$B_P1")"; B_EVAL="$(abs "$B_EVAL")"
{ for b in "$B_COSET" "$B_P1" "$B_EVAL"; do echo "$(md5sum "$b" | cut -c1-32)  $(basename "$b")"; done; } > "$SEND/binaries.txt"

# ------------------------------------------------------------------------------------------------ the tests
gpu_idle || die 2 "the card became busy before the tests: $(gpu_q memory.used) MiB used"
step tests
set +e
bounded 1800 "$SEND/whir_coset_device.log" env -i "${RUN_ENV[@]}" "$B_COSET" --test-threads=1
echo "$?" > "$SEND/whir_coset_device.rc"
bounded 1200 "$SEND/whir_p1_device.log" env -i "${RUN_ENV[@]}" "$B_P1" --test-threads=1
echo "$?" > "$SEND/whir_p1_device.rc"
bounded 900 "$SEND/whir_coset.log" env -i "${RUN_ENV[@]}" "$B_EVAL" coset_blocks_equal_the_reencoded_codeword --exact --nocapture --test-threads=1
echo "$?" > "$SEND/whir_coset.rc"
set -e
say "tests done: rc $(cat "$SEND/whir_coset_device.rc") / $(cat "$SEND/whir_p1_device.rc") / $(cat "$SEND/whir_coset.rc")"

# ------------------------------------------------------------------------------------------------ the summary
step summary
python3 - "$SEND" > "$SEND/summary.txt" <<'PY'
import re, sys, os
send = sys.argv[1]
REG = {
    "whir_coset_device": ["a_coset_chain_on_the_card_proves_the_host_chains_bytes",
                          "a_revived_coset_stack_proves_the_whole_trees_bytes_from_the_stack_or_a_reencode",
                          "an_unencoded_successor_refuses_the_coset_prove_by_cause",
                          "a_revived_natural_stack_is_untouched"],
    "whir_p1_device": ["the_device_p1_chain_proves_the_host_chains_bytes", "the_device_p1_ext3_tree_is_the_hosts"],
    "whir_coset": ["coset_blocks_equal_the_reencoded_codeword"],
}
RESULT = re.compile(r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed")
PAIR = re.compile(r"([a-z0-9_]+)=([^\s]+)")
def lines(name):
    try:
        return open(os.path.join(send, name), errors="replace").read().splitlines()
    except OSError:
        return []
def failures(ls):
    names, inside = [], False
    for l in ls:
        if l.strip() == "failures:":
            inside, names = True, []
            continue
        if inside:
            if not l.strip():
                if names:
                    inside = False
                continue
            if l.startswith("    ") and not l.startswith("     "):
                names.append(l.strip())
            elif names:
                inside = False
    return names
out_rows = []
for binary, names in REG.items():
    ls = lines(f"{binary}.log")
    rc = (lines(f"{binary}.rc") or ["?"])[0].strip()
    res = next((RESULT.search(l) for l in reversed(ls) if RESULT.search(l)), None)
    res = (res.group(1), int(res.group(2)), int(res.group(3))) if res else None
    if binary == "whir_coset":
        # --nocapture: the name's line is glued to the test's own first line, so only the closing result counts.
        each = {n: res is not None and res[0] == "ok" for n in names}
    else:
        each = {n: any(l.startswith(f"test {n} ... ok") for l in ls) for n in names}
    failed = failures(ls)
    ok = rc == "0" and res is not None and res[0] == "ok" and res[1] == len(names) and res[2] == 0 and all(each.values()) and not failed
    print(f"ROW {binary} {'IN' if ok else 'OUT'} rc {rc} · result {res} · registered {len(names)}"
          + (f" · failed {' '.join(failed)}" if failed else ""))
    for n, passed in each.items():
        print(f"TEST {binary}::{n} {'ok' if passed else 'NOT OK'}")
    out_rows.append((binary, ok))
par = [dict(PAIR.findall(m.group(1))) for m in (re.search(r"J5 PARITY (.*)$", l) for l in lines("whir_coset.log")) if m]
for d in par:
    print(f"PARITY n={d.get('n')} cosets={d.get('cosets')} values={d.get('values')} vs_codeword={d.get('vs_codeword')} "
          f"vs_host={d.get('vs_host')} vs_queued={d.get('vs_queued')}")
par_ok = len(par) == 3 and all(d.get(k) == "0" for d in par for k in ("vs_codeword", "vs_host", "vs_queued"))
print(f"ROW parity {'IN' if par_ok else 'OUT'} {len(par)} lines, every one 0 differing" if par_ok else
      f"ROW parity OUT {len(par)} lines")
out = [b for b, ok in out_rows if not ok] + ([] if par_ok else ["parity"])
print("VERDICT: COSET PARITY " + ("PASS — every registered test passed, the parity lines read 0" if not out
      else "FAIL — rows out: " + " ".join(out)))
PY
cat "$SEND/summary.txt" | tee -a "$LOG"

# ------------------------------------------------------------------------------------------------ send back
step send-back
for f in "$SEND"/*; do
  [ -f "$f" ] || continue
  sed -i -e "s|$W|<DIR>|g" -e "s|$HOME|<HOME>|g" "$f"
done
if grep -rEl -- "$SECRET_MARKERS" "$SEND" > "$RUN/scan-hits.txt" 2>/dev/null; then
  die 4 "a credential marker in $(tr '\n' ' ' < "$RUN/scan-hits.txt") — nothing written; send the lane that path instead"
fi
TAR="$W/coset-send-$STAMP.tar.gz"
tar -czf "$TAR" -C "$RUN" send
say "send-back: $TAR ($(du -h "$TAR" | cut -f1))"
echo
echo "coset_parity: done — $(grep '^VERDICT' "$SEND/summary.txt" | head -1)"
echo "coset_parity: please send back $TAR"
