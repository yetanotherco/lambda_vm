	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# LW: Load word (sign-extends to 64 bits)
	# Store 0x80000000, load with LW should give 0xFFFFFFFF80000000 (sign-extended)
	li	a2, 0x80000000
	lui	a3, 0x80000         # Base address
	sw	a2, 0(a3)           # Store word
	lw	a0, 0(a3)           # Load word (sign-extend)
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
