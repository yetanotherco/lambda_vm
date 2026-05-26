	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Misaligned LD at offset 1 from a 4-aligned base.
	# Access reads bytes [33..40], crossing three 4-byte cells.
	li	t0, 32
	li	t1, 0x80FF1234
	sw	t1, 0(t0)
	li	t1, 0xDEADBEEF
	sw	t1, 4(t0)
	li	t1, 0x000000AA
	sw	t1, 8(t0)
	ld	a0, 1(t0)
	li	a0, 0
	li	a7, 93
	ecall
