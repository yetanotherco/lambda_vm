	.text
	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Loads and stores of every width, signed and unsigned, aligned and not,
	# and accesses whose bytes were last touched at different times. A load's
	# destination is never read again before it is written, so a test may
	# claim another loaded value without touching the instructions after it.
	addi	sp, sp, -64
	li	t0, 0x8172635445362718	# bytes 18 27 36 45 54 63 72 81

	# Aligned, every width.
	sd	t0, 0(sp)
	ld	t1, 0(sp)
	lw	t2, 0(sp)		# 0x45362718, sign clear
	lw	t2, 4(sp)		# 0x81726354, sign set
	lwu	t3, 4(sp)
	lh	t4, 6(sp)		# 0x8172, sign set
	lhu	t4, 6(sp)
	lh	t5, 0(sp)		# 0x2718, sign clear
	lb	t6, 7(sp)		# 0x81, sign set
	lbu	t6, 7(sp)
	lb	a1, 0(sp)		# 0x18, sign clear
	sw	t0, 8(sp)
	sh	t0, 12(sp)
	sb	t0, 14(sp)

	# Unaligned.
	lw	a2, 2(sp)
	lhu	a2, 3(sp)
	sh	t0, 9(sp)
	sd	t0, 17(sp)
	ld	a3, 1(sp)

	# Aligned, its bytes last touched at different times: byte 1 is newer.
	sd	zero, 32(sp)
	sb	t0, 33(sp)
	ld	a4, 32(sp)
	lw	a5, 32(sp)		# every byte read at the LD: one old timestamp

	# A signed byte load of 0x05 and of 0x80, each destination written next.
	li	t0, 0x8005
	sh	t0, 40(sp)
	lb	t1, 40(sp)		# 5
	li	t1, 0
	lb	t1, 41(sp)		# sign-extended 0x80
	li	t1, 0

	# A doubleword stored, read back and overwritten.
	li	t0, 0x0102030405060708
	sd	t0, 48(sp)
	ld	t1, 48(sp)
	li	t1, 0
	sd	zero, 48(sp)

	# A byte stored and read back, never touched again.
	li	t0, 7
	sb	t0, 56(sp)
	lbu	t1, 56(sp)
	li	t1, 0

	addi	sp, sp, 64
	li	a0, 0
	li	a7, 93			# syscall = Halt
	ecall
