	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# DIVUW 0x80000000 / 1: quotient 0x80000000 (bit 31 set).
	li	a2, 0x80000000
	addi	a3, zero, 1
	divuw	a0, a2, a3
	li	a0, 0
	li	a7, 93
	ecall
