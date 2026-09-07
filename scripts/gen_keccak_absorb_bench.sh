#!/usr/bin/env bash
#
# gen_keccak_absorb_bench.sh — generate + compile a keccak-sponge-absorb
# saturated guest.
#
# The guest seeds a 200-byte keccak state and a DATA_BLOCKS × 136-byte message
# region, then fires the absorb ecall (a7 = u64::MAX - 3, spec -4) CALLS times,
# absorbing ALL DATA_BLOCKS blocks through ONE ecall each time, commits the
# 200-byte state and halts. Total absorbed blocks (= KECCAK_SPONGE rows =
# permutations) is N = CALLS × DATA_BLOCKS.
#
# Why a loop of large calls: the accelerator's win is the per-block guest glue
# it deletes, so the interesting shape is few ecalls × many blocks — the loop
# body is ~7 cycles per CALL regardless of DATA_BLOCKS, keeping the trace
# sponge-saturated. The calls deliberately reuse the same data region (the
# prover's cost per block is identical either way — no layer dedupes rows, the
# (ts, seq) keys differ per call) and chain the state across calls, so the
# absorbed content differs every call at zero extra cycles.
#
# Count-gate the run via the CLI counter before benching:
#   cargo run -p cli --release -- execute BENCH.elf --cycles
#   -> "KeccakAbsorb calls: CALLS"
#
# KECCAK_SPONGE commits one row per absorbed block, so padding-flush sweep
# points are powers of two: pick CALLS × DATA_BLOCKS = 2^k.
#
# Usage: scripts/gen_keccak_absorb_bench.sh CALLS DATA_BLOCKS OUT.elf
# Honors CLANG / ASM_CFLAGS / ASM_LDFLAGS like the Makefile's asm rule.

set -euo pipefail

CALLS="${1:?usage: gen_keccak_absorb_bench.sh CALLS DATA_BLOCKS out.elf}"
DATA_BLOCKS="${2:?usage: gen_keccak_absorb_bench.sh CALLS DATA_BLOCKS out.elf}"
OUT="${3:?usage: gen_keccak_absorb_bench.sh CALLS DATA_BLOCKS out.elf}"

if ! [[ "$CALLS" =~ ^[0-9]+$ ]] || [ "$CALLS" -lt 1 ]; then
    echo "gen_keccak_absorb_bench.sh: CALLS must be a positive integer, got '$CALLS'" >&2
    exit 1
fi
if ! [[ "$DATA_BLOCKS" =~ ^[0-9]+$ ]] || [ "$DATA_BLOCKS" -lt 1 ]; then
    echo "gen_keccak_absorb_bench.sh: DATA_BLOCKS must be a positive integer, got '$DATA_BLOCKS'" >&2
    exit 1
fi

CLANG="${CLANG:-clang}"
ASM_CFLAGS="${ASM_CFLAGS:---target=riscv64 -march=rv64im -mabi=lp64}"
ASM_LDFLAGS="${ASM_LDFLAGS:--fuse-ld=lld -nostdlib -Wl,-e,main}"

if ! command -v "$CLANG" >/dev/null 2>&1; then
    echo "gen_keccak_absorb_bench.sh: '$CLANG' not found; run 'make deps' or set CLANG=..." >&2
    exit 1
fi

DATA_BYTES=$((DATA_BLOCKS * 136))
DATA_DWORDS=$((DATA_BLOCKS * 17))
# 200-byte state + data region, rounded up to 16 for stack hygiene.
FRAME=$(((200 + DATA_BYTES + 15) / 16 * 16))

SRC="$(mktemp "${TMPDIR:-/tmp}/keccak_absorb_bench.XXXXXX.s")"
trap 'rm -f "$SRC"' EXIT

cat > "$SRC" <<ASM
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Frame: 200-byte keccak state at sp, $DATA_BLOCKS x 136-byte blocks at sp+200.
	li	t3, $FRAME
	sub	sp, sp, t3

	# Deterministic non-zero state: lane[i] = i + 1.
	mv	t0, sp
	li	t1, 1
	li	t2, 26
.Lstate_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Lstate_loop

	# Deterministic message data: dword[k] = k + 100 ($DATA_DWORDS dwords).
	addi	t0, sp, 200
	li	t1, 100
	li	t2, $((DATA_DWORDS + 100))
.Ldata_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Ldata_loop

	# $CALLS absorb calls of $DATA_BLOCKS blocks each. The state chains
	# across calls (the ecall updates it in place).
	li	s0, $CALLS
.Labsorb_loop:
	mv	a0, sp
	addi	a1, sp, 200
	li	a2, $DATA_BLOCKS
	li	a7, -4
	ecall
	addi	s0, s0, -1
	bnez	s0, .Labsorb_loop

	# Commit the final 200-byte state so the work is load-bearing.
	li	a0, 1
	mv	a1, sp
	li	a2, 200
	li	a7, 64
	ecall

	# Restore stack and halt.
	li	t3, $FRAME
	add	sp, sp, t3
	li	a0, 0
	li	a7, 93
	ecall
ASM

# shellcheck disable=SC2086  # flag strings are intentionally word-split
"$CLANG" $ASM_CFLAGS $ASM_LDFLAGS "$SRC" -o "$OUT"
echo "gen_keccak_absorb_bench: built $OUT (CALLS=$CALLS x DATA_BLOCKS=$DATA_BLOCKS = $((CALLS * DATA_BLOCKS)) KECCAK_SPONGE rows)"
