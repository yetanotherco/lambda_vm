	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# REMUW: 32-bit unsigned remainder
	# 0xFFFFFFFF % 7 = 3
	li	a2, 0xFFFFFFFF
	addi	a3, zero, 7
	remuw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
