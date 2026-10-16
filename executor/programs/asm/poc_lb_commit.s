	.text
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Store the byte 0x05 (sign bit clear), read it back with a real
	# sign-extending LB, spill the result and commit it. The committed public
	# output is the LOAD chip's res, so a proof reporting anything but 5
	# certifies a false sign extension.
	addi	sp, sp, -16
	addi	t0, zero, 5
	sb	t0, 8(sp)
	lb	t1, 8(sp)		# t1 = sext(0x05) = 5
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
