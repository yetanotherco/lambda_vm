	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Three keccak256 calls (ECALL -5) of the empty input, one permutation
	# each, placed for windows of 8 cycles cut at cycle 16: call 1 (cycle 5)
	# before the cut, calls 2 and 3 (cycles 20 and 25) after it, and the halt
	# (cycle 37) in a later window than call 3. The cycle of each instruction
	# is in its comment; keep them there if this file changes.
	addi	sp, sp, -256		# 0
	addi	a0, sp, 128		# 1: input (zeros, not read at length 0)
	li	a1, 0			# 2: length 0
	mv	a2, sp			# 3: digest 1 at sp
	li	a7, -5			# 4: keccak256
	ecall				# 5: call 1
	nop				# 6
	nop				# 7
	nop				# 8
	nop				# 9
	nop				# 10
	nop				# 11
	nop				# 12
	nop				# 13
	nop				# 14
	nop				# 15
	addi	a0, sp, 128		# 16
	li	a1, 0			# 17
	addi	a2, sp, 32		# 18: digest 2
	li	a7, -5			# 19
	ecall				# 20: call 2
	addi	a0, sp, 128		# 21
	li	a1, 0			# 22
	addi	a2, sp, 64		# 23: digest 3
	li	a7, -5			# 24
	ecall				# 25: call 3
	nop				# 26
	nop				# 27
	nop				# 28
	nop				# 29
	nop				# 30
	nop				# 31
	nop				# 32
	nop				# 33
	addi	sp, sp, 256		# 34
	li	a0, 0			# 35: exit_code = 0
	li	a7, 93			# 36: syscall = Halt
	ecall				# 37: halt
