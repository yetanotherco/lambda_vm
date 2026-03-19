	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# LUI: Maximum positive value
	# lui x, 0x7FFFF loads 0x7FFFF000
	lui	a0, 0x7FFFF
	li	a0, 0
	li	a7, 93
	ecall
