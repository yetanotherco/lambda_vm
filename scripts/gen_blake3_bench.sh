#!/usr/bin/env bash
#
# gen_blake3_bench.sh — generate + compile a blake3-saturated guest.
#
# The guest seeds a 176-byte BLAKE3 state region (layout: h[4 dwords] | m[8] |
# t[1] | len,flags[1] | out[8]), fires the BLAKE3 6-round compression ecall N
# times IN PLACE on that region, commits the 64-byte output and halts.
#
# The calls are deliberately NOT chained (out is not copied over m): chaining
# costs an 8-dword copy loop = ~82 of ~87 cycles per compression, and those
# copy cycles become CPU/MEMW rows that dilute the very table being measured.
# The prover's cost per ecall is identical either way — the executor runs every
# call, and no layer dedupes rows (timestamps differ per call) — so dropping
# the chain buys a ~5-cycle loop body and a >90% blake3-saturated trace, the
# same shape as gen_keccak_bench.sh. (The e2e correctness test, which is about
# values rather than cost, does chain: prover/src/tests test_prove_elfs_blake3.)
#
# The BLAKE3 table commits ONE row per compression (fully unrolled layout), so
# padding-flush sweep points are simply powers of two: N = 2^k. At ~5
# cycles/compression a 2^17-row table costs ~0.7M cycles — inside one 2^20
# epoch; use --epoch-size-log2 21 from N = 2^18 up.
#
# ABI (executor/src/vm/instruction/execution.rs BLAKE3_SYSCALL_NUMBER):
#   a7 = u64::MAX - 2, written as the sign-extended -3
#   a0 = 8-byte-aligned pointer to the 176-byte region
#
# Usage: scripts/gen_blake3_bench.sh N OUT.elf
# Honors CLANG / ASM_CFLAGS / ASM_LDFLAGS like the Makefile's asm rule.

set -euo pipefail

N="${1:?usage: gen_blake3_bench.sh N out.elf}"
OUT="${2:?usage: gen_blake3_bench.sh N out.elf}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "gen_blake3_bench.sh: N must be a positive integer, got '$N'" >&2
    exit 1
fi

CLANG="${CLANG:-clang}"
ASM_CFLAGS="${ASM_CFLAGS:---target=riscv64 -march=rv64im -mabi=lp64}"
ASM_LDFLAGS="${ASM_LDFLAGS:--fuse-ld=lld -nostdlib -Wl,-e,main}"

if ! command -v "$CLANG" >/dev/null 2>&1; then
    echo "gen_blake3_bench.sh: '$CLANG' not found; run 'make deps' or set CLANG=..." >&2
    exit 1
fi

SRC="$(mktemp "${TMPDIR:-/tmp}/blake3_bench.XXXXXX.s")"
trap 'rm -f "$SRC"' EXIT

cat > "$SRC" <<ASM
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 176-byte state region, 8-aligned.
	addi	sp, sp, -176
	andi	sp, sp, -8

	# Deterministic seed: input dword[k] = k + 1.
	mv	t0, sp
	li	t1, 1
	li	t2, 15
.Linit_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Linit_loop

	li	s0, $N
.Lperm_loop:
	# BLAKE3 6-round compression, in place. See the header for why the
	# calls are independent rather than chained.
	mv	a0, sp
	li	a7, -3
	ecall
	addi	s0, s0, -1
	bnez	s0, .Lperm_loop

	# Commit the final 64-byte output so the work is load-bearing.
	li	a0, 1
	addi	a1, sp, 112
	li	a2, 64
	li	a7, 64
	ecall

	li	a0, 0
	li	a7, 93
	ecall
ASM

# shellcheck disable=SC2086  # flag strings are intentionally word-split
"$CLANG" $ASM_CFLAGS $ASM_LDFLAGS "$SRC" -o "$OUT"
echo "gen_blake3_bench: built $OUT (N=$N compressions = $N BLAKE3 rows)"
