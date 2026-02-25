	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# ADDW: 32-bit add, sign-extend result to 64 bits
	# 0x7FFFFFFF + 1 = 0x80000000 (32-bit), sign-extends to 0xFFFFFFFF80000000
	li	a2, 0x7FFFFFFF
	li	a3, 1
	addw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
