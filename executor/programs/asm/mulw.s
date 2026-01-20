	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# MULW: 32-bit multiply, keep lower 32 bits, sign-extend
	# 100000 * 30000 = 3000000000 = 0xB2D05E00
	# Sign-extended: 0xFFFFFFFFB2D05E00 (-1294967296)
	li	a2, 100000
	li	a3, 30000
	mulw	a0, a2, a3
	jalr	zero, 0(ra)
