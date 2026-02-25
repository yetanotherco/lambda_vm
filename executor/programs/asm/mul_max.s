	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	li	a2, -1
	li	a3, -1
	mul    a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
