	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SLLW: shift amount uses only lower 5 bits
	# 1 << 33 should be 1 << 1 = 2 (since 33 & 0x1F = 1)
	addi	a2, zero, 1
	addi	a3, zero, 33
	sllw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
