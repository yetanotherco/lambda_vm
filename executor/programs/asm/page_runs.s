	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Touches runs of contiguous zero pages and two private-input pages, for
	# the tables that hold a run of pages as one (page size 0x40000):
	# - six pages from 0x1000000 (page 64, aligned to 16): a block of four
	#   and a block of two;
	# - four pages from 0x1440000 (page 81, odd): blocks of one, two and one.
	# A word is written at each page's start and read back, two words across
	# the first run's first internal page boundary, and the private input is
	# read on both its pages. The reads' sum is committed.

	li	t1, 0x0123456789abcdef
	li	t3, 0

	# Run A: pages 64..69.
	li	t0, 0x1000000
	sd	t1, 0(t0)
	li	t0, 0x1040000
	sd	t1, 0(t0)
	li	t0, 0x1080000
	sd	t1, 0(t0)
	li	t0, 0x10c0000
	sd	t1, 0(t0)
	li	t0, 0x1100000
	sd	t1, 0(t0)
	li	t0, 0x1140000
	sd	t1, 0(t0)

	# Across the boundary of pages 64 and 65: the last word of 64 and the
	# word after it.
	li	t0, 0x103fff8
	sd	t1, 0(t0)
	addi	t2, t1, 1
	sd	t2, 8(t0)
	ld	t4, 0(t0)
	add	t3, t3, t4
	ld	t4, 8(t0)
	add	t3, t3, t4

	# Run B: pages 81..84.
	li	t0, 0x1440000
	sd	t1, 0(t0)
	li	t0, 0x1480000
	sd	t1, 0(t0)
	li	t0, 0x14c0000
	sd	t1, 0(t0)
	li	t0, 0x1500000
	sd	t1, 0(t0)

	# Read back each page's first word.
	li	t0, 0x1000000
	ld	t4, 0(t0)
	add	t3, t3, t4
	li	t0, 0x1140000
	ld	t4, 0(t0)
	add	t3, t3, t4
	li	t0, 0x1440000
	ld	t4, 0(t0)
	add	t3, t3, t4
	li	t0, 0x1500000
	ld	t4, 0(t0)
	add	t3, t3, t4

	# The private input: its length on page 0, a word on page 1.
	li	t0, 0xff000000
	lw	t4, 0(t0)
	add	t3, t3, t4
	li	t0, 0xff040000
	ld	t4, 0(t0)
	add	t3, t3, t4

	# Commit the sum's 8 bytes from a heap word.
	li	t0, 0x1000008
	sd	t3, 0(t0)
	mv	a1, t0			# buf_addr
	li	a0, 1			# fd = 1
	li	a2, 8			# count = 8
	li	a7, 64			# syscall = Commit
	ecall

	# Halt
	li	a0, 0			# exit_code = 0
	li	a7, 93			# syscall = Halt
	ecall
