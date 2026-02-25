	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: 32-bit ADDW only
	addi	t0, zero, 10		# 1. t0 = 10
	addi	t1, zero, 20		# 2. t1 = 20
	addw	a0, t0, t1		# 3. 10 + 20 = 30
	addw	a1, t0, zero		# 4. 10 + 0 = 10
	addw	a2, zero, t1		# 5. 0 + 20 = 20
	addw	a3, t0, t0		# 6. 10 + 10 = 20
	addw	a4, t1, t1		# 7. 20 + 20 = 40
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall		# 8. Return
