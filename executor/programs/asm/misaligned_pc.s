	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Jump to PC = 2 (not 4-aligned).
	jalr	zero, zero, 2
	li	a0, 0
	li	a7, 93
	ecall
