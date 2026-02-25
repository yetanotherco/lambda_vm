	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# REMW by zero: returns dividend
	addi	a2, zero, 42
	addi	a3, zero, 0
	remw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
