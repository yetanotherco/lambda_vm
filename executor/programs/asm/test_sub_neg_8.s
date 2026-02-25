	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: 64-bit SUB with negative values
	addi	t0, zero, -1		# 1. t0 = -1 (all ones)
	addi	t1, zero, 1		# 2. t1 = 1
	sub	a0, t0, t1		# 3. -1 - 1 = -2
	sub	a1, t1, t0		# 4. 1 - (-1) = 2
	sub	a2, t0, t0		# 5. -1 - (-1) = 0
	sub	a3, zero, t0		# 6. 0 - (-1) = 1
	sub	a4, t0, zero		# 7. -1 - 0 = -1
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall		# 8. Return
