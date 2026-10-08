	.text
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Compute 3 * 5 with a real MUL, spill the product to the stack and
	# commit it. The committed public output is the MUL chip's lo result,
	# so a proof reporting anything but 15 certifies a false product.
	addi	a3, zero, 3
	addi	a4, zero, 5
	mul	t1, a3, a4		# t1 = 3 * 5
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
