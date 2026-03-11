	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Store 4 bytes [0xAA, 0xBB, 0xCC, 0xDD] on the stack, then commit them.
	# Commit syscall: x17=3, x10=1(fd), x11=buf_addr, x12=count

	addi	sp, sp, -16		# 1: allocate stack
	addi	t0, zero, 0xAA		# 2: t0 = 0xAA (170)
	sb	t0, 0(sp)		# 3: store byte 0
	addi	t0, zero, 0xBB		# 4: t0 = 0xBB (187)
	sb	t0, 1(sp)		# 5: store byte 1
	addi	t0, zero, 0xCC		# 6: t0 = 0xCC (204)
	sb	t0, 2(sp)		# 7: store byte 2
	addi	t0, zero, 0xDD		# 8: t0 = 0xDD (221)
	sb	t0, 3(sp)		# 9: store byte 3

	# Commit ecall
	li	a0, 1			# 10: fd = 1 (stdout)
	mv	a1, sp			# 11: buf_addr = sp
	li	a2, 4			# 12: count = 4
	li	a7, 3			# 13: syscall = Commit
	ecall				# 14: commit

	# Halt
	addi	sp, sp, 16		# 15: deallocate stack
	li	a0, 0			# 16: exit_code = 0
	li	a7, 93			# 17: syscall = Halt (sys_exit)
	ecall				# 18: halt
