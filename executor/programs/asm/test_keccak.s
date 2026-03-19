	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	# Allocate 200 bytes on the stack for the Keccak state (25 × u64)
	# Stack grows downward; sp is already initialized
	addi	sp, sp, -200

	# Zero out the state (200 bytes = 25 doublewords)
	# Using a loop to store zeros
	mv	t0, sp		# t0 = pointer to state
	li	t1, 25		# t1 = counter (25 doublewords)
.Lzero_loop:
	sd	zero, 0(t0)	# Store zero doubleword
	addi	t0, t0, 8	# Next doubleword
	addi	t1, t1, -1	# Decrement counter
	bnez	t1, .Lzero_loop

	# Call keccak-f[1600] permutation
	# a0 = pointer to 200-byte state
	# a7 = syscall number (0xFFFFFFFE = u64::MAX - 1)
	mv	a0, sp
	li	a7, -2		# 0xFFFFFFFFFFFFFFFE in 64-bit
	ecall

	# Restore stack and halt
	addi	sp, sp, 200
	li	a0, 0		# exit code = 0
	li	a7, 93		# sys_exit
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
