	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Misaligned SW at offset 1 from a 4-aligned base.
	# Writes bytes to addresses [65..68], crossing two 4-byte cells.
	li	t0, 64
	li	t1, 0xDEADBEEF
	sw	t1, 1(t0)
	li	a0, 0
	li	a7, 93
	ecall
