	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SUB: 64-bit subtraction
	# 0x200000000 - 0x100000000 = 0x100000000
	li	a2, 0x200000000
	li	a3, 0x100000000
	sub	a0, a2, a3
	li	a7, 5
	ecall
