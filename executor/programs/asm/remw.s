	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# REMW: 32-bit signed remainder
	# -100 % 7 = -2
	addi	a2, zero, -100
	addi	a3, zero, 7
	remw	a0, a2, a3
	li	a0, 0
	li	a7, 93
	ecall
