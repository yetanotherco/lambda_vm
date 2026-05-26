	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# REMUW 0x80000001 % 0xFFFFFFFE: remainder 0x80000001 (bit 31 set).
	li	a2, 0x80000001
	li	a3, 0xFFFFFFFE
	remuw	a0, a2, a3
	li	a0, 0
	li	a7, 93
	ecall
