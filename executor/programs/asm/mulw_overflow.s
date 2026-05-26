	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# MULW 0x10000 * 0x10000: full product 0x1_00000000, low 32 bits = 0.
	li	a2, 0x10000
	li	a3, 0x10000
	mulw	a0, a2, a3
	li	a0, 0
	li	a7, 93
	ecall
