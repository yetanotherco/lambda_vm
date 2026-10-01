	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Commit 3 bytes [0xAA, 0xBB, 0xCC]: a public output that does not fill its
	# last 4-byte half, so the statement's last half has one pad byte.
	# Commit syscall: x17=64, x10=1(fd), x11=buf_addr, x12=count

	addi	sp, sp, -16		# allocate stack
	addi	t0, zero, 0xAA
	sb	t0, 0(sp)
	addi	t0, zero, 0xBB
	sb	t0, 1(sp)
	addi	t0, zero, 0xCC
	sb	t0, 2(sp)

	li	a0, 1			# fd = 1 (stdout)
	mv	a1, sp			# buf_addr = sp
	li	a2, 3			# count = 3
	li	a7, 64			# syscall = Commit
	ecall

	# Halt
	addi	sp, sp, 16		# deallocate stack
	li	a0, 0			# exit_code = 0
	li	a7, 93			# syscall = Halt (sys_exit)
	ecall
