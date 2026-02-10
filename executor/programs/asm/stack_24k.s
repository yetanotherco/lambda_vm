	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Test program that uses ~24 KiB of stack memory (6 pages).
	# Writes a pattern at each page boundary, reads back to verify.
	# RISC-V addi immediate range is [-2048, 2047], so we use
	# a loop to decrement SP by 24576 bytes.

	# Save original SP
	add	s0, sp, zero		# s0 = original sp

	# Allocate 24 KiB (24576 bytes) using loop: 12 iterations of -2048
	addi	t0, zero, 12		# loop counter
.alloc_loop:
	addi	sp, sp, -2048
	addi	t0, t0, -1
	bne	t0, zero, .alloc_loop

	# Now sp = original_sp - 24576
	# Write pattern bytes at ~4096-byte intervals to touch each page

	# Page 0: write at sp+0
	addi	t1, zero, 0x11
	sb	t1, 0(sp)

	# Compute sp+4096: use lui for 4096 = 0x1000
	lui	t2, 1			# t2 = 4096
	add	t3, sp, t2		# t3 = sp + 4096
	addi	t1, zero, 0x22
	sb	t1, 0(t3)		# Page 1

	add	t3, t3, t2		# t3 = sp + 8192
	addi	t1, zero, 0x33
	sb	t1, 0(t3)		# Page 2

	add	t3, t3, t2		# t3 = sp + 12288
	addi	t1, zero, 0x44
	sb	t1, 0(t3)		# Page 3

	add	t3, t3, t2		# t3 = sp + 16384
	addi	t1, zero, 0x55
	sb	t1, 0(t3)		# Page 4

	add	t3, t3, t2		# t3 = sp + 20480
	addi	t1, zero, 0x66
	sb	t1, 0(t3)		# Page 5

	# Read back pattern from page 0
	lb	a0, 0(sp)		# a0 = 0x11

	# Restore SP
	add	sp, s0, zero		# sp = original sp

	li	a7, 5
	ecall
