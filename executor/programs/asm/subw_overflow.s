	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# SUBW: 0x80000000 - 1 = 0x7FFFFFFF (overflow from negative to positive)
	li	a2, 0x80000000
	addi	a3, zero, 1
	subw	a0, a2, a3
	jalr	zero, 0(ra)
