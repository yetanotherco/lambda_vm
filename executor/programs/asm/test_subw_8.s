	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: 32-bit SUBW only
	addi	t0, zero, 10		# 1. t0 = 10
	addi	t1, zero, 20		# 2. t1 = 20
	subw	a0, t1, t0		# 3. 20 - 10 = 10
	subw	a1, t0, t1		# 4. 10 - 20 = -10
	subw	a2, t0, zero		# 5. 10 - 0 = 10
	subw	a3, zero, t0		# 6. 0 - 10 = -10
	subw	a4, t0, t0		# 7. 10 - 10 = 0
	li	a0, 0
	li	a7, 93
	ecall		# 8. Return
