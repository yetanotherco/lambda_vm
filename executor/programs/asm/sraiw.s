	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SRAIW: 32-bit shift right arithmetic immediate
	# 0x80000000 >> 1 = 0xC0000000 (sign bit preserved)
	# Sign-extended to 64 bits: 0xFFFFFFFFC0000000 (-1073741824)
	li	a2, 0x80000000
	sraiw	a0, a2, 1
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
