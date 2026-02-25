	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SUBW: 32-bit subtract, sign-extend result to 64 bits
	# 10 - 20 = -10
	addi	a2, zero, 10
	addi	a3, zero, 20
	subw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
