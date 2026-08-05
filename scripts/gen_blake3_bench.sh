#!/usr/bin/env bash
#
# gen_blake3_bench.sh — generate + compile a blake3-saturated guest.
#
# The guest seeds a 176-byte BLAKE3 state region (layout: h[4 dwords] | m[8] |
# t[1] | len,flags[1] | out[8]), then N times: fires the BLAKE3 6-round
# compression ecall and copies out over m, so every compression consumes the
# previous one's output — a strictly dependent chain that nothing can fold.
# Finally it commits the 64-byte output and halts.
#
# The BLAKE3 table commits ONE row per compression (fully unrolled layout), so
# padding-flush sweep points are simply powers of two: N = 2^k. Guest cost is
# ~21 cycles per compression (ecall + 8-dword copy + loop), so a 2^17-row table
# costs ~2.8M cycles — well inside a 2^22 epoch.
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
	# BLAKE3 6-round compression on the region.
	mv	a0, sp
	li	a7, -3
	ecall
	# Chain: m <- out (8 dwords), so the next compression depends on this one.
	li	t1, 0
.Lcopy_loop:
	slli	t2, t1, 3
	addi	t3, sp, 112
	add	t3, t3, t2
	ld	t4, 0(t3)
	addi	t3, sp, 32
	add	t3, t3, t2
	sd	t4, 0(t3)
	addi	t1, t1, 1
	li	t2, 8
	bne	t1, t2, .Lcopy_loop
	addi	s0, s0, -1
	bnez	s0, .Lperm_loop

	# Commit the final 64-byte output so the chain is load-bearing.
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
