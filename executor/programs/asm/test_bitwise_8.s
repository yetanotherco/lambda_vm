	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 8-instruction test: Bitwise operations only
	addi	t0, zero, 0xFF		# 1. t0 = 255
	addi	t1, zero, 0x0F		# 2. t1 = 15
	andi	a0, t0, 0x0F		# 3. 255 & 15 = 15
	ori	a1, t1, 0xF0		# 4. 15 | 240 = 255
	xori	a2, t0, 0x0F		# 5. 255 ^ 15 = 240
	and	a3, t0, t1		# 6. 255 & 15 = 15
	or	a4, t0, t1		# 7. 255 | 15 = 255
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall		# 8. Return
