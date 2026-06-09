	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	# Stack layout (96 bytes): xG at sp+0, k at sp+32, xR at sp+64.
	addi	sp, sp, -96

	# xG = secp256k1 Gx, little-endian (written once; reused by all calls).
	li	t0, 0x59F2815B16F81798
	sd	t0, 0(sp)
	li	t0, 0x029BFCDB2DCE28D9
	sd	t0, 8(sp)
	li	t0, 0x55A06295CE870B07
	sd	t0, 16(sp)
	li	t0, 0x79BE667EF9DCBBAC
	sd	t0, 24(sp)

	# k's high doublewords stay zero for all calls; only k[0] changes.
	sd	zero, 40(sp)
	sd	zero, 48(sp)
	sd	zero, 56(sp)

	# --- call 1: k = 1 (no ECDAS steps; start/final tuples cancel directly) ---
	li	t0, 1
	sd	t0, 32(sp)
	addi	a0, sp, 64
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -3
	ecall
	li	a0, 1
	addi	a1, sp, 64
	li	a2, 32
	li	a7, 64
	ecall

	# --- call 2: k = 5 (double, double, add) ---
	li	t0, 5
	sd	t0, 32(sp)
	addi	a0, sp, 64
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -3
	ecall
	li	a0, 1
	addi	a1, sp, 64
	li	a2, 32
	li	a7, 64
	ecall

	# --- call 3: k = 0xABCDEF (24-bit; many doubles + several adds) ---
	li	t0, 0xABCDEF
	sd	t0, 32(sp)
	addi	a0, sp, 64
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -3
	ecall
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
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
