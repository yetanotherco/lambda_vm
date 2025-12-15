	.attribute	5, "rv32i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
	addi    a0,zero,3
	jalr	zero, 0(ra)
main:
	addi    a2,zero,1
	addi    a3,zero,2
	addi    a0,zero,2
	bne     a2, a3, -20
	jalr	zero, 0(ra)
	addi    a0, zero, 1
	jalr	zero, 0(ra)
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
