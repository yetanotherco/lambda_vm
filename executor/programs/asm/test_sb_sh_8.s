	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Test SB and SH (byte/half stores) - 8 instructions
	addi	t0, zero, 0x42		# t0 = 0x42 (byte value)
	addi	t1, zero, 0x789		# t1 = 0x789 (half value, fits in 12 bits)
	addi	sp, sp, -16		# Allocate stack space
	sb	t0, 0(sp)		# Store byte
	sh	t1, 2(sp)		# Store half
	lb	a0, 0(sp)		# Load byte back (a0 = 0x42)
	lh	a1, 2(sp)		# Load half back (a1 = 0x789)
	li	a0, 0
	li	a7, 93
	ecall
