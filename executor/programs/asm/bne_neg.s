	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
	li	a0, 0
	jalr	zero, 0(ra)
main:
	addi    a2,zero,1
	addi    a3,zero,2
	addi    a0,zero,2
	bne     a2, a3, -20
	li	a0, 0
	jalr	zero, 0(ra)
	addi    a0, zero, 1
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
