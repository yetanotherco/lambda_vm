	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: DIV/REM only
	addi	t0, zero, 10		# 1. t0 = 10
	addi	t1, zero, 100		# 2. t1 = 100
	div	a0, t1, t0		# 3. 100 / 10 = 10
	rem	a1, t1, t0		# 4. 100 % 10 = 0
	div	a2, t0, t0		# 5. 10 / 10 = 1
	rem	a3, t0, t0		# 6. 10 % 10 = 0
	addi	a4, zero, 0		# 7. nop
	li	a0, 0
	li	a7, 93
	ecall		# 8. Return
