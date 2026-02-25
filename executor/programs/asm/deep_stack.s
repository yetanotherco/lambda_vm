	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Deep stack usage: allocates 8192 bytes of stack space.
	# SP starts at 0xFFFF_FFFF_FFFF_FFF0.
	# After allocation: SP = 0x...DFF0 (falls in page 0x...D000).
	# With default stack_size=4096, only pages E000 and F000 are
	# initialized, so stores into page D000 leave the memory bus
	# unbalanced. Increasing stack_size to 8192 adds page D000.
	lui	t1, 2			# t1 = 8192
	sub	sp, sp, t1		# sp -= 8192 → 0x...DFF0
	addi	t0, zero, 0x42
	sb	t0, 0(sp)		# Store byte in page D000
	lb	a0, 0(sp)		# Load it back
	add	sp, sp, t1		# Restore stack
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
