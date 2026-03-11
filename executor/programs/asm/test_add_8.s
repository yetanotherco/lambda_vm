	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: 64-bit ADD only
	addi	t0, zero, 10		# 1. t0 = 10
	addi	t1, zero, 20		# 2. t1 = 20
	add	a0, t0, t1		# 3. 10 + 20 = 30
	add	a1, t0, zero		# 4. 10 + 0 = 10
	add	a2, zero, t1		# 5. 0 + 20 = 20
	add	a3, t0, t0		# 6. 10 + 10 = 20
	add	a4, t1, t1		# 7. 20 + 20 = 40
	li	a0, 0
	li	a7, 93
	ecall		# 8. Return
