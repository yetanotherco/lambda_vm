	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: 64-bit ADD with negative values
	addi	t0, zero, -1		# 1. t0 = -1 (all ones)
	addi	t1, zero, 1		# 2. t1 = 1
	add	a0, t0, t1		# 3. -1 + 1 = 0
	add	a1, t0, t0		# 4. -1 + -1 = -2
	add	a2, t1, t0		# 5. 1 + -1 = 0
	add	a3, t0, zero		# 6. -1 + 0 = -1
	add	a4, zero, t0		# 7. 0 + -1 = -1
	li	a0, 0
	li	a7, 93
	ecall		# 8. Return
