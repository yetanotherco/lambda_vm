	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SLLI: 64-bit shift left by 32 (only valid in RV64)
	# 1 << 32 = 0x100000000
	addi	a2, zero, 1
	slli	a0, a2, 32
	jalr	zero, 0(ra)
