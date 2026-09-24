#!/bin/bash
# Generate + compile a keccak-heavy guest that applies keccak-f[1600] N times
# in place on a 200-byte state, then commits it. N is arg 1; output ELF is arg 2.
set -euo pipefail
N="${1:?usage: gen_keccak_bench.sh N out.elf}"
OUT="${2:?usage: gen_keccak_bench.sh N out.elf}"
SRC="$(mktemp /tmp/keccak_bench.XXXXXX.s)"
trap 'rm -f "$SRC"' EXIT

cat > "$SRC" <<ASM
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Allocate 200 bytes on the stack for the Keccak state (25 x u64).
	addi	sp, sp, -200

	# Initialize a non-zero, deterministic state: lane[i] = i + 1.
	mv	t0, sp
	li	t1, 1
	li	t2, 26
.Linit_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Linit_loop

	# Apply keccak-f[1600] N times in place.
	li	s0, $N
.Lperm_loop:
	mv	a0, sp
	li	a7, -2
	ecall
	addi	s0, s0, -1
	bnez	s0, .Lperm_loop

	# Commit the final 200-byte state.
	li	a0, 1
	mv	a1, sp
	li	a2, 200
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 200
	li	a0, 0
	li	a7, 93
	ecall
ASM

clang --target=riscv64 -march=rv64im -mabi=lp64 -fuse-ld=lld -nostdlib -Wl,-e,main "$SRC" -o "$OUT"
echo "built $OUT (N=$N)"
