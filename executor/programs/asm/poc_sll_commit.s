	.text
	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Compute 1 << 1 with a real SLL, spill the result to the stack and
	# commit it. The committed public output is the SHIFT chip's result,
	# so a proof reporting anything but 2 certifies a false shift.
	addi	a3, zero, 1
	addi	a4, zero, 1
	sll	t1, a3, a4		# t1 = 1 << 1
	addi	sp, sp, -16
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
