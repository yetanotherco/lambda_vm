	# Two genesis pages dense enough for the cross-epoch genesis rule to stack
	# them: thousands of nonzero bytes each, 2^18 bytes (one page) apart. The
	# no-epoch WHIR block's prepared-opening negatives need two dense pages.
	.data
	.align 12
page_a:
	.fill	12288, 1, 0x5a
	.skip	262144
page_b:
	.fill	12288, 1, 0xa5

	.text
	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Read one byte of each page, commit their sum.
	la	t0, page_a
	lbu	t1, 0(t0)
	la	t0, page_b
	lbu	t2, 0(t0)
	add	t1, t1, t2
	addi	sp, sp, -16
	sb	t1, 0(sp)
	li	a0, 1			# fd = 1
	mv	a1, sp			# buf = sp
	li	a2, 1			# count = 1
	li	a7, 64			# syscall = Commit
	ecall

	addi	sp, sp, 16
	li	a0, 0
	li	a7, 93			# syscall = Halt
	ecall
