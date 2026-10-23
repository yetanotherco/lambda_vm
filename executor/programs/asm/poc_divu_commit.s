	.text
	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Compute 7 / 2 with a real DIVU, spill the quotient to the stack and
	# commit it. The committed public output is the DVRM chip's quotient,
	# so a proof reporting anything but 3 certifies a false division.
	addi	a3, zero, 7
	addi	a4, zero, 2
	divu	t1, a3, a4		# t1 = 7 / 2
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
