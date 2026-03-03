	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# MUL: 64-bit multiply with large values
	# 0x100000000 * 2 = 0x200000000
	li	a2, 0x100000000
	addi	a3, zero, 2
	mul	a0, a2, a3
	li	a7, 5
	ecall
