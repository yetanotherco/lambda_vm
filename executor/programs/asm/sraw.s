	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SRAW: 32-bit shift right arithmetic
	# 0x80000000 >> 1 = 0xC0000000 (arithmetic, sign bit preserved)
	# Then sign-extended to 64 bits: 0xFFFFFFFFC0000000
	li	a2, 0x80000000
	addi	a3, zero, 1
	sraw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
