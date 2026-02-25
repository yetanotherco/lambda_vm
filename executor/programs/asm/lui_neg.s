	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# LUI: Load upper immediate with sign extension
	# lui x, 0x80000 loads 0x80000000 which sign-extends to 0xFFFFFFFF80000000
	lui	a0, 0x80000
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
