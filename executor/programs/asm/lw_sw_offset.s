	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	addi a2,zero,1
	addi a3,zero,20
	sw a2,4(a3)
	lw a0,4(a3)
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
