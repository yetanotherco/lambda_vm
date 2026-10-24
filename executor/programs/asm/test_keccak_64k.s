	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	# Place the 200-byte Keccak state (25 × u64) below sp so that it straddles
	# a 64 KiB boundary: lanes 0..11 sit below it and lanes 12..24 above, so
	# the 25 state pointers carry two different (Word, Half) keys.
	li	t2, -65536
	and	s0, sp, t2
	addi	s0, s0, -96

	# Zero out the state (200 bytes = 25 doublewords)
	mv	t0, s0
	li	t1, 25
.Lzero_loop:
	sd	zero, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, -1
	bnez	t1, .Lzero_loop

	# Call keccak-f[1600] permutation
	# a0 = pointer to 200-byte state
	# a7 = syscall number (0xFFFFFFFFFFFFFFFE = u64::MAX - 1)
	mv	a0, s0
	li	a7, -2
	ecall

	# Commit the post-permutation state so the test can verify the KAT.
	# Commit syscall: a0=fd(1), a1=buf_addr, a2=count, a7=64
	li	a0, 1
	mv	a1, s0
	li	a2, 200
	li	a7, 64
	ecall

	# Halt
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
