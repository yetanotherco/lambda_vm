	.text
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# The program's one byte store: a doubleword of zeros, then 0x34 into
	# its byte 0, then its byte 1 read back (0) and the destination written.
	addi	sp, sp, -16
	li	t0, 0x1234
	sd	zero, 0(sp)
	sb	t0, 0(sp)
	lbu	t1, 1(sp)
	li	t1, 0
	addi	sp, sp, 16
	li	a0, 0
	li	a7, 93			# syscall = Halt
	ecall
