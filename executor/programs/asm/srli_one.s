	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	srli	a0, zero, 1
	li	a7, 5
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
