#!/usr/bin/env bash
#
# bench_recursion_scaling.sh — in-VM recursion-verifier scaling ladder.
#
# Sweeps ethrex block sizes × verifier presets: proves each block's inner
# execution via CONTINUATIONS (memory-bounded 2^EPOCH_LOG2-cycle epochs, so any
# block size proves on a bounded-RAM box), then executes the continuation
# recursion guest (recursion-cont-<preset>.elf) on the bundle and records the
# EXACT deterministic guest cycle count — the in-VM cost of verifying that
# block's proof.
#
# Sweep order is PRESET-MAJOR on purpose: the full block-size curve for the
# first preset completes before the next preset starts, so the headline regime
# (blowup2 = the realistic base-layer options, 219 FRI queries) yields a usable
# scaling curve as early as possible instead of only when everything ends.
#
# Usage: scripts/bench_recursion_scaling.sh [RESULTS_FILE=/tmp/recursion_scaling.txt]
#   Env:
#     TXS="1 4 8 16"                 block sizes (transfers); fixtures are read from
#                                    executor/tests/ethrex_bench_<N>.bin (committed for
#                                    1/4/8/16) and generated via tooling/ethrex-fixtures
#                                    when missing.
#     PRESETS="blowup2 blowup4 min"  verifier presets, most important first.
#     EPOCH_LOG2=21                  inner continuation epoch size (log2 cycles).
#
# Prereqs (the script fails fast on each):
#   cargo build --release -p cli
#   make compile-recursion-elfs                       (recursion-cont-*.elf)
#   make executor/program_artifacts/rust/ethrex.elf   (the inner guest)
#
# Output: one key=value line per cell in RESULTS_FILE, e.g.
#   txs=4 preset=blowup2 epochs=2 blob=145080513 cycles=18984803380 keccak=3152604 exec_wall_s=164
# Cycle counts are deterministic (machine-independent); wall times are not.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

TXS="${TXS:-1 4 8 16}"
PRESETS="${PRESETS:-blowup2 blowup4 min}"
EPOCH_LOG2="${EPOCH_LOG2:-21}"
RESULTS="${1:-/tmp/recursion_scaling.txt}"
WORK="$(mktemp -d /tmp/recursion_scaling.XXXXXX)"

CLI=target/release/cli
ART=executor/program_artifacts/recursion
ETHREX=executor/program_artifacts/rust/ethrex.elf

[ -x "$CLI" ] || { echo "ERROR: $CLI missing — run: cargo build --release -p cli" >&2; exit 1; }
[ -f "$ETHREX" ] || { echo "ERROR: $ETHREX missing — run: make $ETHREX" >&2; exit 1; }
for P in $PRESETS; do
  [ -f "$ART/recursion-cont-${P}.elf" ] || {
    echo "ERROR: $ART/recursion-cont-${P}.elf missing — run: make compile-recursion-elfs" >&2
    exit 1
  }
done

echo "==> results -> $RESULTS   (work dir: $WORK)"
: > "$RESULTS"

for P in $PRESETS; do
  for N in $TXS; do
    FIX=executor/tests/ethrex_bench_${N}.bin
    if [ ! -f "$FIX" ]; then
      echo "==> [${P}/${N}tx] generating missing fixture $FIX" >&2
      ( cd tooling/ethrex-fixtures && cargo build --release ) >"$WORK/fixtures_build.log" 2>&1
      tooling/ethrex-fixtures/target/release/ethrex-fixtures "$N" "$FIX" distinct >&2
    fi

    # Inner block cost, once per block size (cheap; repeated per preset is fine —
    # the count is deterministic and the run takes well under a second).
    ic="$("$CLI" execute "$ETHREX" --private-input "$FIX" --cycles | awk -F': ' '/^Cycles:/{print $2; exit}')"

    echo "==> [${P}/${N}tx] proving inner continuation (epoch=2^${EPOCH_LOG2}) ..." >&2
    rm -f /tmp/recursion_input.bin
    DLOG="$WORK/dump_${N}tx_${P}.log"
    if ! ( RECURSION_DUMP_PRESET="$P" RECURSION_DUMP_EPOCH_LOG2="$EPOCH_LOG2" \
           RECURSION_DUMP_INNER_ELF="$PWD/$ETHREX" RECURSION_DUMP_INNER_INPUT="$PWD/$FIX" \
           cargo test --release -p lambda-vm-prover --lib test_dump_recursion_input -- --ignored --nocapture ) \
           >"$DLOG" 2>&1 || [ ! -f /tmp/recursion_input.bin ]; then
      echo "txs=$N preset=$P inner_cycles=$ic DUMP_FAILED" >> "$RESULTS"
      echo "ERROR: [${P}/${N}tx] dump failed; tail of $DLOG:" >&2
      tail -20 "$DLOG" >&2
      continue
    fi
    epochs="$(grep -o 'continuation epochs: [0-9]*' "$DLOG" | awk '{print $3}')"
    BLOB="$WORK/blob_${N}tx_${P}.bin"
    mv /tmp/recursion_input.bin "$BLOB"
    sz="$(wc -c < "$BLOB" | tr -d ' ')"

    echo "==> [${P}/${N}tx] executing recursion-cont-${P}.elf (${epochs} epochs, ${sz} bytes) ..." >&2
    t0=$(date +%s)
    if out="$("$CLI" execute "$ART/recursion-cont-${P}.elf" --private-input "$BLOB" --cycles 2>"$WORK/exec_${N}tx_${P}.err")"; then
      t1=$(date +%s)
      cyc="$(printf '%s\n' "$out" | awk -F': ' '/^Cycles:/{print $2; exit}')"
      kec="$(printf '%s\n' "$out" | awk -F': ' '/^Keccak calls:/{print $2; exit}')"
      line="txs=$N preset=$P inner_cycles=$ic epochs=$epochs blob=$sz cycles=$cyc keccak=$kec exec_wall_s=$((t1 - t0))"
      echo "$line" >> "$RESULTS"
      echo "    $line" >&2
    else
      echo "txs=$N preset=$P inner_cycles=$ic epochs=$epochs blob=$sz EXEC_FAILED" >> "$RESULTS"
      echo "ERROR: [${P}/${N}tx] guest execute failed; tail of stderr:" >&2
      tail -10 "$WORK/exec_${N}tx_${P}.err" >&2
    fi
    rm -f "$BLOB"
  done
done

echo "==> done. Results:"
cat "$RESULTS"
