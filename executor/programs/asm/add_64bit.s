	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# ADD: 64-bit addition with values > 32 bits
	# 0x100000000 + 0x100000000 = 0x200000000
	li	a2, 0x100000000
	li	a3, 0x100000000
	add	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
