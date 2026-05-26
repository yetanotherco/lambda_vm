	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Misaligned LHU at offset 3 from a 4-aligned base.
	# Access reads bytes [35, 36], crossing two 4-byte cells.
	li	t0, 32
	li	t1, 0x3412FF80
	sw	t1, 0(t0)
	li	t1, 0x000000AA
	sw	t1, 4(t0)
	lhu	a0, 3(t0)
	li	a0, 0
	li	a7, 93
	ecall
