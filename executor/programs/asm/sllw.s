	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SLLW: 32-bit shift left logical
	# 1 << 31 = 0x80000000, sign-extends to 0xFFFFFFFF80000000 (-2147483648)
	addi	a2, zero, 1
	addi	a3, zero, 31
	sllw	a0, a2, a3
	li	a0, 0
	li	a7, 93
	ecall
