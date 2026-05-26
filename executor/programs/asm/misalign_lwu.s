	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Misaligned LWU at offset 1 from a 4-aligned base.
	# Access reads bytes [33, 34, 35, 36], crossing two 4-byte cells.
	li	t0, 32
	li	t1, 0x80FF1234
	sw	t1, 0(t0)
	li	t1, 0x000000AA
	sw	t1, 4(t0)
	lwu	a0, 1(t0)
	li	a0, 0
	li	a7, 93
	ecall
