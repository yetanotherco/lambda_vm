	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Commit [0xAA,0xBB] early, do filler work, then commit [0xCC,0xDD] later —
	# so with a small epoch size the two commits fall in DIFFERENT epochs and the
	# second commit's epoch starts with x254 (commit index) already = 2.
	addi	sp, sp, -16		# allocate stack

	# --- first commit: bytes [0xAA, 0xBB] ---
	addi	t0, zero, 0xAA
	sb	t0, 0(sp)
	addi	t0, zero, 0xBB
	sb	t0, 1(sp)
	li	a0, 1			# fd = 1
	mv	a1, sp			# buf = sp
	li	a2, 2			# count = 2
	li	a7, 64			# syscall = Commit
	ecall

	# --- filler work (room for an epoch boundary between the two commits) ---
	addi	t1, zero, 0
	addi	t1, t1, 1
	addi	t1, t1, 1
	addi	t1, t1, 1
	addi	t1, t1, 1
	addi	t1, t1, 1
	addi	t1, t1, 1
	addi	t1, t1, 1
	addi	t1, t1, 1
	addi	t1, t1, 1

	# --- second commit: bytes [0xCC, 0xDD] ---
	addi	t0, zero, 0xCC
	sb	t0, 2(sp)
	addi	t0, zero, 0xDD
	sb	t0, 3(sp)
	li	a0, 1			# fd = 1
	addi	a1, sp, 2		# buf = sp+2
	li	a2, 2			# count = 2
	li	a7, 64			# syscall = Commit
	ecall

	# --- halt ---
	addi	sp, sp, 16
	li	a0, 0
	li	a7, 93			# syscall = Halt
	ecall
