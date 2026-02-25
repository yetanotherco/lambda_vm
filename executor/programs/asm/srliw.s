	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SRLIW: 32-bit shift right logical immediate
	# 0x80000000 >> 1 = 0x40000000
	li	a2, 0x80000000
	srliw	a0, a2, 1
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
