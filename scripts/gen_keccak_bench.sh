#!/usr/bin/env bash
#
# gen_keccak_bench.sh — generate + compile a keccak-saturated guest.
#
# The guest allocates a 200-byte keccak state on the stack, seeds it
# deterministically (lane[i] = i+1), applies keccak-f[1600] N times IN PLACE via
# the accelerator ecall, commits the final state, and halts. The permutation is
# the only real work: the loop body is 5 instructions, so at N=5461 the guest
# retires ~27k cycles while emitting 5461 permutations = 131,064 KECCAK_RND rows.
# That makes the prover's cost essentially pure keccak (>85% of committed cells),
# which is the point — it turns prover wall time into a keccaks/sec reading.
#
# The state is fed back into itself each iteration (in-place permutation), so the
# work is a genuine dependent chain and cannot be constant-folded by anything.
#
# ABI (executor/src/vm/instruction/execution.rs):
#   a7 = KECCAK_SYSCALL_NUMBER = u64::MAX - 1, written as the sign-extended -2
#   a0 = state address, MUST be 8-byte aligned; 25 lanes x 8 bytes, permuted in place
#
# Usage: scripts/gen_keccak_bench.sh N OUT.elf
#
# Honors the same toolchain overrides as the Makefile's asm rule:
#   CLANG, ASM_CFLAGS, ASM_LDFLAGS

set -euo pipefail

N="${1:?usage: gen_keccak_bench.sh N out.elf}"
OUT="${2:?usage: gen_keccak_bench.sh N out.elf}"

if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
    echo "gen_keccak_bench.sh: N must be a positive integer, got '$N'" >&2
    exit 1
fi

CLANG="${CLANG:-clang}"
ASM_CFLAGS="${ASM_CFLAGS:---target=riscv64 -march=rv64im -mabi=lp64}"
ASM_LDFLAGS="${ASM_LDFLAGS:--fuse-ld=lld -nostdlib -Wl,-e,main}"

if ! command -v "$CLANG" >/dev/null 2>&1; then
    echo "gen_keccak_bench.sh: '$CLANG' not found; run 'make deps' or set CLANG=..." >&2
    exit 1
fi

SRC="$(mktemp "${TMPDIR:-/tmp}/keccak_bench.XXXXXX.s")"
trap 'rm -f "$SRC"' EXIT

cat > "$SRC" <<ASM
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 200 bytes of stack for the keccak state (25 x u64), 8-byte aligned as
	# the accelerator requires (UnalignedKeccakStateAddress otherwise).
	addi	sp, sp, -200
	andi	sp, sp, -8

	# Deterministic non-zero seed: lane[i] = i + 1.
	mv	t0, sp
	li	t1, 1
	li	t2, 26
.Linit_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Linit_loop

	# N in-place keccak-f[1600] permutations. Each iteration consumes the
	# previous output, so the chain is strictly sequential.
	li	s0, $N
.Lperm_loop:
	mv	a0, sp
	li	a7, -2
	ecall
	addi	s0, s0, -1
	bnez	s0, .Lperm_loop

	# Commit the final state so the permutations are load-bearing for the
	# public output and cannot be treated as dead.
	li	a0, 1
	mv	a1, sp
	li	a2, 200
	li	a7, 64
	ecall

	li	a0, 0
	li	a7, 93
	ecall
ASM

# shellcheck disable=SC2086  # flag strings are intentionally word-split
"$CLANG" $ASM_CFLAGS $ASM_LDFLAGS "$SRC" -o "$OUT"
echo "gen_keccak_bench: built $OUT (N=$N permutations, $((N * 24)) KECCAK_RND rows)"
