	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# DIVUW: 32-bit unsigned division
	# 0xFFFFFFFF / 2 = 0x7FFFFFFF (2147483647)
	li	a2, 0xFFFFFFFF
	addi	a3, zero, 2
	divuw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
