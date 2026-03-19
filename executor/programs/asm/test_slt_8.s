	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: SLT operations only
	addi	t0, zero, 10		# 1. t0 = 10
	addi	t1, zero, 20		# 2. t1 = 20
	slti	a0, t0, 20		# 3. 10 < 20 = 1
	slti	a1, t1, 10		# 4. 20 < 10 = 0
	sltiu	a2, t0, 20		# 5. unsigned: 10 < 20 = 1
	slt	a3, t0, t1		# 6. 10 < 20 = 1
	sltu	a4, t1, t0		# 7. unsigned: 20 < 10 = 0
	li	a0, 0
	li	a7, 93
	ecall		# 8. Return
