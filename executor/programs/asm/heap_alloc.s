	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Simulates heap allocation by writing to memory well above ELF segments.
	# Writes across 4 pages (0x80000, 0x81000, 0x82000, 0x83000).
	# These addresses are far from both ELF segments (~0x10000-0x11000) and
	# stack (STACK_TOP ~0xFFFF..FFF0), exercising runtime page detection.

	addi	t1, zero, 0x42		# pattern byte

	# Page 0x80000
	lui	t0, 0x80		# t0 = 0x80000
	sb	t1, 0(t0)

	# Page 0x81000
	lui	t0, 0x81		# t0 = 0x81000
	sb	t1, 0(t0)

	# Page 0x82000
	lui	t0, 0x82		# t0 = 0x82000
	sb	t1, 0(t0)

	# Page 0x83000
	lui	t0, 0x83		# t0 = 0x83000
	sb	t1, 0(t0)

	# Also write to stack to ensure stack page is detected too
	addi	sp, sp, -16
	sd	t1, 0(sp)
	ld	a0, 0(sp)
	addi	sp, sp, 16

	li	a0, 0
	li	a7, 93
	ecall
