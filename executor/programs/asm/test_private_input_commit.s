	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Read private inputs, then commit the first 4 bytes of data (skipping length prefix).
	# GetPrivateInputs syscall: x17=4, x10=dest_pointer
	# Commit syscall: x17=64, x10=1(fd), x11=buf_addr, x12=count

	addi	sp, sp, -256		# 1: allocate stack buffer

	# GetPrivateInputs: copies length-prefixed data to sp
	mv	a0, sp			# 2: dest pointer = sp
	li	a7, 4			# 3: syscall = GetPrivateInputs
	ecall				# 4: get_private_input

	# The data at sp is: [len_u32_le, data...]
	# Skip the 4-byte length prefix, commit the next 4 bytes
	addi	a1, sp, 4		# 5: buf_addr = sp + 4 (skip length prefix)
	li	a0, 1			# 6: fd = 1 (stdout)
	li	a2, 4			# 7: count = 4 bytes
	li	a7, 64			# 8: syscall = Commit
	ecall				# 9: commit

	# Halt
	addi	sp, sp, 256		# 10: deallocate stack
	li	a0, 0			# 11: exit_code = 0
	li	a7, 93			# 12: syscall = Halt
	ecall				# 13: halt
