	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# ADDW: simple positive case
	# 10 + 20 = 30
	addi	a2, zero, 10
	addi	a3, zero, 20
	addw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
