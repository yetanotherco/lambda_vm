	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Loop: read N bytes of private input, write each back to memory, commit 8 bytes.
	# This mimics what a real program would do: read private input, do memory ops on it.

	li	t0, -4096		# 1: sp offset (fits in imm)
	add	sp, sp, t0		# 2: allocate 4KB stack
	li	t0, -4096
	add	sp, sp, t0		# 3: another 4KB (total 8KB, spans 2 pages)

	# GetPrivateInputs: copy to sp
	mv	a0, sp			# 4: dest = sp
	li	a7, 4			# 5: syscall = GetPrivateInputs
	ecall				# 6: get_private_input

	# Read 8 bytes at sp+8 (aligned), write to sp+2000
	ld	t1, 8(sp)		# 7
	sd	t1, 2000(sp)		# 8

	# Read 8 bytes at sp+2000 (should match what we just wrote)
	ld	t2, 2000(sp)		# 9

	# Commit 8 bytes from sp+8
	addi	a1, sp, 8		# 10
	li	a0, 1			# 11
	li	a2, 8			# 12
	li	a7, 64			# 13
	ecall				# 14

	# Halt
	li	t0, 4096		# 15
	add	sp, sp, t0		# 16
	li	t0, 4096
	add	sp, sp, t0		# 17
	li	a0, 0			# 18
	li	a7, 93			# 19
	ecall				# 20
