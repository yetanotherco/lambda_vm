	.text
	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Compare 5 < 3 with a real SLT, spill the result to the stack and commit
	# it. The committed public output is the LT chip's result, so a proof
	# reporting anything but 0 certifies a false comparison. The spill comes
	# right after the SLT so it is the first instruction to read the result.
	addi	sp, sp, -16
	addi	a3, zero, 5
	addi	a4, zero, 3
	slt	t1, a3, a4		# t1 = (5 < 3) = 0
	sd	t1, 0(sp)		# spill to stack
	li	a0, 1			# fd = 1
	mv	a1, sp			# buf = sp
	li	a2, 8			# count = 8
	li	a7, 64			# syscall = Commit
	ecall

	addi	sp, sp, 16
	li	a0, 0
	li	a7, 93			# syscall = Halt
	ecall
