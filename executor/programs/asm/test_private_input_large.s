	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Read private inputs, then commit 8 bytes (spans a doubleword load/store)
	# to test that multi-byte reads from GPI data work correctly.

	addi	sp, sp, -256		# 1: allocate stack buffer

	# GetPrivateInputs: copies length-prefixed data to sp
	mv	a0, sp			# 2: dest pointer = sp
	li	a7, 4			# 3: syscall = GetPrivateInputs
	ecall				# 4: get_private_input

	# Load doubleword at sp+8 (8-byte aligned, contains some data bytes)
	ld	t0, 8(sp)		# 5: load 8 bytes

	# Store it back to a different location
	sd	t0, 32(sp)		# 6: store 8 bytes

	# Load byte at sp+4, store byte at sp+100
	lb	t1, 4(sp)		# 7: load byte
	sb	t1, 100(sp)		# 8: store byte

	# Commit 8 bytes from sp+8
	addi	a1, sp, 8		# 9: buf_addr = sp + 8
	li	a0, 1			# 10: fd = 1
	li	a2, 8			# 11: count = 8 bytes
	li	a7, 64			# 12: syscall = Commit
	ecall				# 13: commit

	# Halt
	addi	sp, sp, 256		# 14: deallocate stack
	li	a0, 0			# 15: exit_code = 0
	li	a7, 93			# 16: syscall = Halt
	ecall				# 17: halt
