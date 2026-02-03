	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# MULW: negative * positive
	# -10 * 20 = -200
	addi	a2, zero, -10
	addi	a3, zero, 20
	mulw	a0, a2, a3
	li	a7, 5
	ecall
