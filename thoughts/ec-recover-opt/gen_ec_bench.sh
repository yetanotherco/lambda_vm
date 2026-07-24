#!/bin/bash
# Generate + compile a pure-EC guest: N chained full-ladder ECSM ecalls
# (k = N_secp − 1, the largest valid scalar → ~255 doubles + ~254 adds per call,
# ~509 ECDAS rows), feeding each xR back as the next base point, then commits
# the final xR. N is arg 1; output ELF is arg 2.
#
# A/B usage (mirrors thoughts/keccak-hwsl-inline methodology): build the same
# ELF once, prove it with the baseline and the paired-AreBytes prover builds on
# the bench box, alternating A/B runs. ~2000 calls ≈ 1M ECDAS rows ≈ one 2^20
# table. Do NOT run benches locally — hand the command to the bench server.
#
#   ./gen_ec_bench.sh 2000 /tmp/ec_bench_2000.elf
#
set -euo pipefail
N="${1:?usage: gen_ec_bench.sh N out.elf}"
OUT="${2:?usage: gen_ec_bench.sh N out.elf}"
SRC="$(mktemp /tmp/ec_bench.XXXXXX.s)"
trap 'rm -f "$SRC"' EXIT

cat > "$SRC" <<ASM
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Stack layout (96 bytes): xG at sp+0, k at sp+32, xR at sp+64.
	addi	sp, sp, -96

	# xG = secp256k1 Gx, little-endian (4 doublewords).
	li	t0, 0x59F2815B16F81798
	sd	t0, 0(sp)
	li	t0, 0x029BFCDB2DCE28D9
	sd	t0, 8(sp)
	li	t0, 0x55A06295CE870B07
	sd	t0, 16(sp)
	li	t0, 0x79BE667EF9DCBBAC
	sd	t0, 24(sp)

	# k = N_secp - 1 (largest valid scalar), little-endian (4 doublewords).
	li	t0, 0xBFD25E8CD0364140
	sd	t0, 32(sp)
	li	t0, 0xBAAEDCE6AF48A03B
	sd	t0, 40(sp)
	li	t0, 0xFFFFFFFFFFFFFFFE
	sd	t0, 48(sp)
	li	t0, 0xFFFFFFFFFFFFFFFF
	sd	t0, 56(sp)

	# N chained ECSM ecalls: xR = x(k * (xG, ·)), then xG <- xR.
	# x(k*P) is a valid x-coordinate again ((N-1)*P = -P shares P's x, so the
	# first iteration is a fixed point in VALUE; the executor still performs a
	# full double-and-add ladder per call — identical per-iteration work.)
	li	s0, $N
.Lloop:
	addi	a0, sp, 64
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -11
	ecall
	# xG <- xR (4 doublewords).
	ld	t0, 64(sp)
	sd	t0, 0(sp)
	ld	t0, 72(sp)
	sd	t0, 8(sp)
	ld	t0, 80(sp)
	sd	t0, 16(sp)
	ld	t0, 88(sp)
	sd	t0, 24(sp)
	addi	s0, s0, -1
	bnez	s0, .Lloop

	# Commit the final 32-byte xR.
	li	a0, 1
	addi	a1, sp, 64
	li	a2, 32
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 96
	li	a0, 0
	li	a7, 93
	ecall
ASM

clang --target=riscv64 -march=rv64im -mabi=lp64 -fuse-ld=lld -nostdlib -Wl,-e,main "$SRC" -o "$OUT"
echo "built $OUT (N=$N chained full-ladder ECSM calls, ~$((N * 509)) ECDAS rows)"
