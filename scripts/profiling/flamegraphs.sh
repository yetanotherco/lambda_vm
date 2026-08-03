#!/usr/bin/env bash
# On-CPU + off-CPU flamegraphs for one prove run (plan §7 / Layer 4).
#
# The pair matters: with a GPU in the loop, wall-clock = on-CPU work + time
# blocked waiting (cuStreamSynchronize, futexes, page faults). The on-CPU
# graph shows host work between kernel launches; the off-CPU graph shows who
# is waiting on what. Together they account for the whole run.
#
# Usage:
#   scripts/profiling/flamegraphs.sh [options] <elf> [--private-input <bin>]
# Options:
#   --out DIR         output dir (default reports/flame_<elf>_<sha>_<stamp>)
#   --no-build        reuse ./target/release/cli as-is
#   --continuations   prove with --continuations
#   --skip-offcpu     only the on-CPU graph (off-CPU needs sudo + bpfcc-tools)
#   --offcpu-secs N   override the off-CPU capture window (default: the
#                     measured on-CPU run duration + 15s, so offcputime exits
#                     on its own and flushes — signalling it through sudo is
#                     unreliable)
# Env:
#   PROFILE_FEATURES  cli features (default "nvtx,jemalloc-stats")
#   EXTRA_PROVE_ARGS  appended to every `cli prove`
#
# Requires (setup_machine.sh installs all of it): perf, inferno-collapse-perf,
# inferno-flamegraph, bpfcc-tools (offcputime-bpfcc), and the perf sysctls.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT="" BUILD=1 CONT=0 SKIP_OFFCPU=0 OFFCPU_SECS="" ELF="" INPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    --no-build) BUILD=0; shift ;;
    --continuations) CONT=1; shift ;;
    --skip-offcpu) SKIP_OFFCPU=1; shift ;;
    --offcpu-secs) OFFCPU_SECS="$2"; shift 2 ;;
    --private-input) INPUT="$2"; shift 2 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) ELF="$1"; shift ;;
  esac
done
[[ -n "$ELF" && -f "$ELF" ]] || { echo "usage: flamegraphs.sh [options] <elf> [--private-input <bin>]" >&2; exit 2; }

for tool in perf inferno-collapse-perf inferno-flamegraph; do
  command -v "$tool" >/dev/null || { echo "ERROR: $tool missing — run setup_machine.sh" >&2; exit 1; }
done

SHA="$(git rev-parse --short HEAD)"
STAMP="$(date +%Y%m%d_%H%M%S)"
NAME="$(basename "${ELF%.elf}")"
OUT="${OUT:-reports/flame_${NAME}_${SHA}_${STAMP}}"
mkdir -p "$OUT"

FEATURES="${PROFILE_FEATURES:-nvtx,jemalloc-stats}"
PROVE_ARGS=(prove "$ELF" -o "$OUT/proof.bin" --time)
[[ -n "$INPUT" ]] && PROVE_ARGS+=(--private-input "$INPUT")
[[ "$CONT" == 1 ]] && PROVE_ARGS+=(--continuations)
# shellcheck disable=SC2206  # intentional word-split of extra args
PROVE_ARGS+=(${EXTRA_PROVE_ARGS:-})

if [[ "$BUILD" == 1 ]]; then
  echo "==> Building cli (frame pointers on — required for clean fp stacks)"
  CARGO_PROFILE_RELEASE_DEBUG=1 RUSTFLAGS="${RUSTFLAGS:-} -Cforce-frame-pointers=yes" \
    cargo build --release -p cli --features "$FEATURES"
fi
CLI="$ROOT/target/release/cli"

"$ROOT/scripts/profiling/capture_env.sh" > "$OUT/env.json"

# --- on-CPU ------------------------------------------------------------------
echo "==> on-CPU: perf record (997 Hz, fp call graphs, all threads)"
T0=$SECONDS
perf record -F 997 -g -o "$OUT/perf.data" -- "$CLI" "${PROVE_ARGS[@]}" \
  > "$OUT/oncpu_run.log" 2>&1
ONCPU_SECS=$((SECONDS - T0))
rm -f "$OUT/proof.bin"
perf script -i "$OUT/perf.data" | inferno-collapse-perf > "$OUT/oncpu.folded"
inferno-flamegraph --title "on-CPU: $NAME @ $SHA" "$OUT/oncpu.folded" > "$OUT/oncpu.svg"
echo "    $OUT/oncpu.svg"

# --- off-CPU -----------------------------------------------------------------
# Debian installs bcc tools under /usr/sbin, which user PATHs often lack.
OFFCPU_BIN="$(command -v offcputime-bpfcc || true)"
[[ -z "$OFFCPU_BIN" && -x /usr/sbin/offcputime-bpfcc ]] && OFFCPU_BIN=/usr/sbin/offcputime-bpfcc
if [[ "$SKIP_OFFCPU" == 1 ]]; then
  echo "==> off-CPU skipped (--skip-offcpu)"
elif [[ -z "$OFFCPU_BIN" ]]; then
  echo "==> off-CPU skipped: offcputime-bpfcc missing (apt install bpfcc-tools)"
else
  # Fixed capture window sized from the on-CPU run: offcputime exits on its
  # own and flushes its report. (Interrupting it through sudo is unreliable —
  # a SIGINT-based whole-run capture hung and produced 0 bytes in practice.)
  DUR="${OFFCPU_SECS:-$((ONCPU_SECS + 15))}"
  echo "==> off-CPU: second prove run under offcputime-bpfcc for ${DUR}s (needs sudo)"
  "$CLI" "${PROVE_ARGS[@]}" > "$OUT/offcpu_run.log" 2>&1 &
  PID=$!
  sudo "$OFFCPU_BIN" -df -p "$PID" "$DUR" > "$OUT/offcpu.folded" &
  OCP=$!
  wait "$PID" || true
  wait "$OCP" || true
  rm -f "$OUT/proof.bin"
  if [[ -s "$OUT/offcpu.folded" ]]; then
    inferno-flamegraph --colors io --countname us --title "off-CPU: $NAME @ $SHA" \
      "$OUT/offcpu.folded" > "$OUT/offcpu.svg"
    echo "    $OUT/offcpu.svg"
  else
    echo "    WARNING: empty off-CPU capture — retry with --offcpu-secs 30"
  fi
fi

echo
echo "==> Done: $OUT"
echo "    Read the pair together: a phase with low GPU busy% (phase_busy.md) and"
echo "    little on-CPU time is *waiting* — find the culprit stack in offcpu.svg."
