	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Test XOR register-register (8 instructions)
	addi	t0, zero, 0xFF		# t0 = 255
	addi	t1, zero, 0x0F		# t1 = 15
	xor	a0, t0, t1		# a0 = 0xFF ^ 0x0F = 0xF0 (240)
	addi	t2, zero, -1		# t2 = -1 (all bits set)
	xor	a1, t2, t0		# a1 = ~0xFF = 0xFFFFFFFFFFFFFF00
	xor	a2, t0, t0		# a2 = 0 (self XOR)
	addi	t3, zero, 0x55		# t3 = 0x55
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
