	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Compare LWU vs LW: LWU zero-extends, LW sign-extends
	# Store 0x80000000
	# LWU should give 0x0000000080000000
	# LW should give 0xFFFFFFFF80000000
	# This test uses LWU
	li	a2, 0x80000000
	lui	a3, 0x80000         # Base address
	sw	a2, 0(a3)           # Store word
	lwu	a0, 0(a3)           # Load word unsigned (zero-extend)
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
