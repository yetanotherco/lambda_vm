	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# MULH: upper 64 bits of signed 128-bit product
	# 0x100000000 * 0x100000000 = 0x10000000000000000 (128-bit)
	# Upper 64 bits = 1
	li	a2, 0x100000000
	li	a3, 0x100000000
	mulh	a0, a2, a3
	jalr	zero, 0(ra)
