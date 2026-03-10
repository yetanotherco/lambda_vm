	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# DIVW: 32-bit signed division
	# -100 / 7 = -14 (rounds toward zero)
	addi	a2, zero, -100
	addi	a3, zero, 7
	divw	a0, a2, a3
	li	a0, 0
	li	a7, 93
	ecall
