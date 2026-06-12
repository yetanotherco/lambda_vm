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

	# k = 5 (little-endian); exercises double, double, add.
	li	t0, 5
	sd	t0, 32(sp)
	sd	zero, 40(sp)
	sd	zero, 48(sp)
	sd	zero, 56(sp)

	# ECSM ecall: a0 = &xR, a1 = &xG, a2 = &k, a7 = -3.
	addi	a0, sp, 64
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -3
	ecall

	# Commit the 32-byte result xR so the test can check it equals x(5G).
	# Commit syscall: a0 = fd(1), a1 = buf_addr, a2 = count, a7 = 64.
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
