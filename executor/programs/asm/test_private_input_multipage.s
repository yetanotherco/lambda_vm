	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Reads private input across TWO pages of the memory-mapped private-input
	# region and commits 8 bytes from the second page. Exercises multi-page
	# private input: two touched private pages => two non-preprocessed
	# GLOBAL_MEMORY tables in the continuation global proof.
	#
	# Layout: [len:u32 LE] at 0xFF000000, data follows. Page size = 1<<15 = 0x8000.
	# Page 0 = [0xFF000000, 0xFF008000); page 1 = [0xFF008000, 0xFF010000).

	li	t0, 0xFF000000		# page 0 base
	lw	t3, 0(t0)		# touch page 0 (read length)

	li	t2, 0xFF008000		# page 1 base (0xFF000000 + 0x8000)
	ld	t4, 0(t2)		# touch page 1 (read 8 bytes)

	# Commit 8 bytes from page 1 (0xFF008000), so the output depends on page 1.
	mv	a1, t2			# buf_addr = 0xFF008000
	li	a0, 1			# fd = 1
	li	a2, 8			# count = 8
	li	a7, 64			# syscall = Commit
	ecall

	# Halt
	li	a0, 0			# exit_code = 0
	li	a7, 93			# syscall = Halt
	ecall
