	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: basic ADD, SUB, ADDW, SUBW
	addi	t0, zero, 10		# 1. t0 = 10
	addi	t1, zero, 20		# 2. t1 = 20
	add	a0, t0, t1		# 3. 10 + 20 = 30
	sub	a1, t0, t1		# 4. 10 - 20 = -10
	addw	a2, t0, t1		# 5. ADDW: 10 + 20 = 30
	subw	a3, t0, t1		# 6. SUBW: 10 - 20 = -10
	sub	a4, zero, t0		# 7. 0 - 10 = -10
	li	a7, 5
	ecall		# 8. Return
