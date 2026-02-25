	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: Shift operations only
	addi	t0, zero, 1		# 1. t0 = 1
	addi	t1, zero, 4		# 2. t1 = 4 (shift amount)
	sll	a0, t0, t1		# 3. 1 << 4 = 16
	slli	a1, t0, 8		# 4. 1 << 8 = 256
	srli	a2, a1, 4		# 5. 256 >> 4 = 16
	sra	a3, a0, t1		# 6. SRA: 16 >> 4 = 1
	srl	a4, a1, t1		# 7. SRL: 256 >> 4 = 16
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall		# 8. Return
