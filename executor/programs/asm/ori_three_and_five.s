	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	ori	    a2, zero, 0x03   # 0011
	ori	    a0, a2, 0x05     # 0101
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
