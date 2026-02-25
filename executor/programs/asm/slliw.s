	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SLLIW: 32-bit shift left logical immediate
	# 1 << 31 = 0x80000000, sign-extends to 0xFFFFFFFF80000000
	addi	a2, zero, 1
	slliw	a0, a2, 31
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
