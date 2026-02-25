	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SRAI: 64-bit arithmetic shift right by 32
	# 0x8000000000000000 >> 32 (arithmetic) = 0xFFFFFFFF80000000
	li	a2, 0x8000000000000000
	srai	a0, a2, 32
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
