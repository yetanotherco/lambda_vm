	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: ADDW with LUI (sign extension edge case)
	addi	t0, zero, 1		# 1. t0 = 1
	lui	t1, 0x80000		# 2. t1 = 0x80000000 (negative as 32-bit)
	addw	a0, t1, t0		# 3. 0x80000000 + 1 = 0x80000001 (sign ext)
	addw	a1, t1, zero		# 4. 0x80000000 + 0 = 0x80000000
	addw	a2, t0, t0		# 5. 1 + 1 = 2
	addw	a3, zero, t1		# 6. 0 + 0x80000000 = 0x80000000
	addw	a4, t1, t1		# 7. overflow test
	li	a7, 5
	ecall		# 8. Return
