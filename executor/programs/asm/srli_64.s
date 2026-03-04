	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SRLI: 64-bit shift right by 32
	# 0x123456789ABCDEF0 >> 32 = 0x12345678
	li	a2, 0x123456789ABCDEF0
	srli	a0, a2, 32
	li	a0, 0
	li	a7, 93
	ecall
