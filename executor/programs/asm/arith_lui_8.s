	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test with LUI
	addi	t0, zero, 10		# 1. t0 = 10
	addi	t1, zero, 20		# 2. t1 = 20
	lui	t2, 0x80000		# 3. t2 = 0x80000000 (LUI)
	add	a0, t0, t1		# 4. 10 + 20 = 30
	add	a1, t2, t0		# 5. 0x80000000 + 10 = 0x8000000A
	sub	a2, t0, t1		# 6. 10 - 20 = -10
	addw	a3, t0, t1		# 7. ADDW: 30
	li	a7, 5
	ecall		# 8. Return
