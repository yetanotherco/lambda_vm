	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: MUL only
	addi	t0, zero, 10		# 1. t0 = 10
	addi	t1, zero, 20		# 2. t1 = 20
	mul	a0, t0, t1		# 3. 10 * 20 = 200
	mul	a1, t0, t0		# 4. 10 * 10 = 100
	mulw	a2, t0, t1		# 5. MULW: 200
	mul	a3, t1, t1		# 6. 20 * 20 = 400
	mul	a4, zero, t0		# 7. 0 * 10 = 0
	li	a0, 0
	li	a7, 93
	ecall		# 8. Return
