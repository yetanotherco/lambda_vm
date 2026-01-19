	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SRLW: 32-bit shift right logical
	# 0x80000000 >> 1 = 0x40000000 (logical, not arithmetic)
	li	a2, 0x80000000
	addi	a3, zero, 1
	srlw	a0, a2, a3
	jalr	zero, 0(ra)
