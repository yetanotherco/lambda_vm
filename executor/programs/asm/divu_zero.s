	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	addi	a2, zero, 10
	addi	a3, zero, 0
	divu    a0, a2, a3
	li	a7, 5
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
