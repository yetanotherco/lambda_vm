	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# LWU: Load word unsigned (zero-extends to 64 bits)
	# Store 0xFFFFFFFF, load with LWU should give 0x00000000FFFFFFFF (not sign-extended)
	li	a2, 0xFFFFFFFF
	lui	a3, 0x80000         # Base address
	sw	a2, 0(a3)           # Store word
	lwu	a0, 0(a3)           # Load word unsigned
	li	a7, 5
	ecall
