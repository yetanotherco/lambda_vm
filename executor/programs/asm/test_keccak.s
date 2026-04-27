	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	# Allocate 200 bytes on the stack for the Keccak state (25 × u64)
	addi	sp, sp, -200

	# Zero out the state (200 bytes = 25 doublewords)
	mv	t0, sp
	li	t1, 25
.Lzero_loop:
	sd	zero, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, -1
	bnez	t1, .Lzero_loop

	# Call keccak-f[1600] permutation
	# a0 = pointer to 200-byte state
	# a7 = syscall number (0xFFFFFFFFFFFFFFFE = u64::MAX - 1)
	mv	a0, sp
	li	a7, -2
	ecall

	# Commit the post-permutation state so the test can verify the KAT.
	# Commit syscall: a0=fd(1), a1=buf_addr, a2=count, a7=64
	li	a0, 1
	mv	a1, sp
	li	a2, 200
	li	a7, 64
	ecall

	# Restore stack and halt
	addi	sp, sp, 200
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
