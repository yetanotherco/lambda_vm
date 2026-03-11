	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	fence
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
