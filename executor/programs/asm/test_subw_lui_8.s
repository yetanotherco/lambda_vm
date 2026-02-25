	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: SUBW with LUI (sign extension edge case)
	addi	t0, zero, 1		# 1. t0 = 1
	lui	t1, 0x80000		# 2. t1 = 0x80000000 (negative as 32-bit)
	subw	a0, t1, t0		# 3. 0x80000000 - 1 = 0x7FFFFFFF (positive!)
	subw	a1, t1, zero		# 4. 0x80000000 - 0 = 0x80000000
	subw	a2, t0, t1		# 5. 1 - 0x80000000 = 0x80000001
	subw	a3, zero, t1		# 6. 0 - 0x80000000 = 0x80000000
	subw	a4, t1, t1		# 7. 0x80000000 - 0x80000000 = 0
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall		# 8. Return
