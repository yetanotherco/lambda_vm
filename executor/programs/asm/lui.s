	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# LUI: Load upper immediate
	# lui x, 0x12345 loads 0x12345000 into x
	lui	a0, 0x12345
	li	a0, 0
	li	a7, 93
	ecall
