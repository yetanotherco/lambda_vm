#!/usr/bin/env bash
#
# bench_elf_digest.sh — measure the full-ELF Keccak cost on the CONTINUATION
# verify path (native wall-clock A/B), plus an estimate of the guest-cycle
# equivalent. This is the bench for the 2E+1 -> 1 elf_digest fix
# (perf/verifier-eval-points-elf-digest); bench_verify.sh cannot see it
# because it only exercises the monolithic path.
#
# What it does:
#   1. Builds the ethrex guest ELF + a big transfer fixture (real program,
#      many epochs) — needs the sysroot toolchain (server).
#   2. Builds cli at REF_A (branch) and REF_B (baseline) in an isolated worktree.
#   3. Proves ONE continuation bundle with the baseline binary (the change is
#      proof-format neutral: both binaries verify the same bundle).
#   4. Interleaved AB/BA timing of `verify --continuations`.
#   5. Reports mean times, delta, and — using the epoch count and ELF size —
#      the implied per-full-ELF-hash cost, to compare against the theory:
#      delta = 2 * epochs * ceil(elf_bytes/136) * per-permutation-cost.
#
# Usage: scripts/bench_elf_digest.sh [REF_A] [REF_B] [N_PAIRS] [TRANSFERS] [EPOCH_LOG2]
#   REF_A default perf/verifier-eval-points-elf-digest, REF_B origin/main,
#   N_PAIRS 10 (even), TRANSFERS 100, EPOCH_LOG2 20.
#   Env: REBUILD=1 forces rebuild + re-prove.

set -euo pipefail

REF_A="${1:-perf/verifier-eval-points-elf-digest}"
REF_B="${2:-origin/main}"
N_PAIRS="${3:-10}"
TRANSFERS="${4:-100}"
EPOCH_LOG2="${5:-20}"

ELF_REL="executor/program_artifacts/rust/ethrex.elf"
INPUT_REL="executor/tests/ethrex_bench_${TRANSFERS}.bin"
WORK="/tmp/elf_digest_bench"
WT="/tmp/elf_digest_wt"

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

if [ $((N_PAIRS % 2)) -ne 0 ]; then
  echo "WARNING: N_PAIRS=$N_PAIRS is odd; use an even count so AB/BA orders balance."
fi

mkdir -p "$WORK"
git fetch origin --quiet || echo "WARNING: fetch failed; resolving against possibly-stale refs."
SHA_A="$(git rev-parse "$REF_A")"
SHA_B="$(git rev-parse "$REF_B")"
echo "==> A (PR) $REF_A -> ${SHA_A:0:10}"
echo "==> B (base) $REF_B -> ${SHA_B:0:10}"

# --- 1. Guest ELF + fixture (built once, shared) ---
if [ ! -f "$ELF_REL" ]; then
  echo "==> Building ethrex guest ELF (needs sysroot)"
  export SYSROOT_DIR="${SYSROOT_DIR:-$HOME/.lambda-vm-sysroot}"
  make "$ELF_REL"
fi
if [ ! -f "$INPUT_REL" ]; then
  echo "==> Generating ethrex ${TRANSFERS}-transfer fixture"
  ( cd tooling/ethrex-fixtures && cargo build --release )
  tooling/ethrex-fixtures/target/release/ethrex-fixtures "$TRANSFERS" "$INPUT_REL" distinct
fi
ELF="$(cd "$(dirname "$ELF_REL")" && pwd)/$(basename "$ELF_REL")"
INPUT="$(cd "$(dirname "$INPUT_REL")" && pwd)/$(basename "$INPUT_REL")"
ELF_BYTES="$(wc -c < "$ELF" | tr -d '[:space:]')"
PERMS=$(( (ELF_BYTES + 135) / 136 ))
echo "==> ELF size: $ELF_BYTES bytes -> $PERMS Keccak permutations per full-ELF hash"

# --- 2. Build both cli binaries (isolated worktree, shared target dir) ---
need_build=0
if [ "${REBUILD:-0}" = "1" ] || [ ! -x "$WORK/cli_A" ] || [ ! -x "$WORK/cli_B" ]; then
  need_build=1
elif [ "$(cat "$WORK/cli_A.sha" 2>/dev/null)" != "$SHA_A" ] || \
     [ "$(cat "$WORK/cli_B.sha" 2>/dev/null)" != "$SHA_B" ]; then
  need_build=1
fi
if [ "$need_build" = "1" ]; then
  cleanup() { git worktree remove --force "$WT" 2>/dev/null || true; }
  trap cleanup EXIT
  git worktree remove --force "$WT" 2>/dev/null || true
  echo "==> Building both cli binaries in isolated worktree $WT"
  git worktree add --detach "$WT" "$SHA_B" >/dev/null
  build_cli() {
    echo "==> Building cli @ ${1:0:10} -> $2"
    git -C "$WT" checkout --quiet -f "$1"
    ( cd "$WT" && cargo build --release -p cli >"$WORK/build_$2.log" 2>&1 ) \
      || { echo "ERROR: build failed for $2; tail of log:"; tail -30 "$WORK/build_$2.log"; exit 1; }
    cp "$WT/target/release/cli" "$WORK/$2"
    echo "$1" >"$WORK/$2.sha"
  }
  build_cli "$SHA_B" cli_B
  build_cli "$SHA_A" cli_A
  cleanup
  trap - EXIT
else
  echo "==> Reusing cached cli binaries (REBUILD=1 to force)"
fi

# --- 3. Prove ONE continuation bundle with the baseline binary (format-neutral change) ---
BUNDLE="$WORK/bundle.bin"
if [ "${REBUILD:-0}" = "1" ] || [ ! -f "$BUNDLE" ]; then
  echo "==> Proving continuation bundle (epochs of 2^$EPOCH_LOG2 cycles)"
  "$WORK/cli_B" prove "$ELF" --private-input "$INPUT" -o "$BUNDLE" \
    --continuations --epoch-size-log2 "$EPOCH_LOG2" >"$WORK/prove.log" 2>&1 \
    || { echo "ERROR: prove failed; tail:"; tail -20 "$WORK/prove.log"; exit 1; }
else
  echo "==> Reusing cached bundle"
fi

# Epoch count: run the program once for its cycle count, derive E = ceil(cycles/2^log).
CYCLES="$("$WORK/cli_B" run "$ELF" --private-input "$INPUT" --cycles 2>/dev/null | grep 'Cycles:' | grep -oE '[0-9]+' | tail -1 || true)"
if [ -n "${CYCLES:-}" ] && [ "$CYCLES" -gt 0 ]; then
  EPOCHS=$(( (CYCLES + (1 << EPOCH_LOG2) - 1) / (1 << EPOCH_LOG2) ))
  echo "==> Program cycles: $CYCLES -> ~$EPOCHS epochs of 2^$EPOCH_LOG2"
else
  EPOCHS=0
  echo "==> WARNING: could not count cycles; per-pass estimate will be skipped."
fi

# --- 4. Interleaved AB/BA timing of verify --continuations ---
verify_time() { # $1=binary -> seconds or empty
  local out
  out="$("$1" verify "$BUNDLE" "$ELF" --continuations --time 2>&1)" || true
  printf '%s\n' "$out" | grep -o 'Verification time: [0-9.]*' | awk '{print $3}' || true
}

echo "==> Running $N_PAIRS interleaved pairs (negative delta = A faster)"
printf 'pair,a_time,b_time\n' > "$WORK/pairs.csv"
for i in $(seq 1 "$N_PAIRS"); do
  a="$(verify_time "$WORK/cli_A")"; b="$(verify_time "$WORK/cli_B")"
  [ -z "$a" ] || [ -z "$b" ] && { echo "ERROR: unparseable verify output"; "$WORK/cli_A" verify "$BUNDLE" "$ELF" --continuations --time; exit 1; }
  printf '%d,%s,%s\n' "$i" "$a" "$b" >> "$WORK/pairs.csv"
  printf '   pair %2d/%d   A=%ss  B=%ss   %+0.2f%%\n' \
    "$i" "$N_PAIRS" "$a" "$b" "$(awk "BEGIN{print ($a-$b)/$b*100}")"
done

# --- 5. Report ---
python3 - "$WORK/pairs.csv" "$ELF_BYTES" "$PERMS" "$EPOCHS" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
A = [float(r['a_time']) for r in rows]
B = [float(r['b_time']) for r in rows]
mA, mB = sum(A)/len(A), sum(B)/len(B)
elf_bytes, perms, epochs = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
d = mA - mB
print(f"\n=== elf-digest continuation bench ===")
print(f"  mean A (PR):  {mA:.3f}s   mean B (base): {mB:.3f}s   delta: {d:+.3f}s ({(mA-mB)/mB*100:+.2f}%)")
if epochs > 0:
    saved_passes = 2 * epochs          # 2 seeded-transcript builds per epoch removed
    secs_per_pass = -d / saved_passes if saved_passes else 0.0
    # Native software Keccak ~= 9000 cycles/perm; report both observed and theory.
    theory = saved_passes * perms * 9000 / 3e9   # seconds at ~3GHz
    print(f"  removed passes: 2 x {epochs} epochs = {saved_passes} full-ELF hashes/bundle")
    print(f"  observed: {secs_per_pass*1000:.1f} ms per pass ({perms} perms)")
    print(f"  theory @9k cycles/perm, 3GHz: ~{theory:.3f}s total saved")
    print(f"  guest equivalent (pre-#847 sponge ~1.3k cyc/perm): {saved_passes*perms*1322/1e6:.1f}M cycles saved")
else:
    print(f"  (epoch count unknown; delta = 2*epochs*perms*per-perm-cost)")
PY

echo "Artifacts + pairs.csv kept in $WORK"
