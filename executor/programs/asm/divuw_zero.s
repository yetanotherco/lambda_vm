	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# DIVUW by zero: returns 0xFFFFFFFF (all ones in 32-bit)
	# Sign-extended to 64-bit: 0xFFFFFFFFFFFFFFFF = -1
	addi	a2, zero, 100
	addi	a3, zero, 0
	divuw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
