	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SUB: underflow wrapping
	# 0 - 1 = -1 = 0xFFFFFFFFFFFFFFFF
	addi	a2, zero, 0
	addi	a3, zero, 1
	sub	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
