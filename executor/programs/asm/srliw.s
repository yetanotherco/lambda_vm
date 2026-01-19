	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SRLIW: 32-bit shift right logical immediate
	# 0x80000000 >> 1 = 0x40000000
	li	a2, 0x80000000
	srliw	a0, a2, 1
	jalr	zero, 0(ra)
