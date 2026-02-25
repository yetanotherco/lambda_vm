	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SLLI: 64-bit shift left by 63 (maximum shift)
	# 1 << 63 = 0x8000000000000000 = i64::MIN
	addi	a2, zero, 1
	slli	a0, a2, 63
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
