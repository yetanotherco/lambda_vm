	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# ADDIW: 32-bit add immediate, sign-extend result
	# 0x7FFFFFFF + 1 = 0x80000000, sign-extends to 0xFFFFFFFF80000000
	li	a2, 0x7FFFFFFF
	addiw	a0, a2, 1
	li	a0, 0
	li	a7, 93
	ecall
