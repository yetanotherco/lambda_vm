	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SUB: basic subtraction
	# 30 - 10 = 20
	addi	a2, zero, 30
	addi	a3, zero, 10
	sub	a0, a2, a3
	jalr	zero, 0(ra)
