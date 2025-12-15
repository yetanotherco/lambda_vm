	.attribute	5, "rv32i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	addi    a3,zero,5
	addi    a2,zero,0
	bne     a2, a3, 12
	addi    a0, a2, 0
	jalr    zero, 0(ra)
	addi    a2,a2,1
	jal     zero,-16
	jalr	zero, 0(ra)
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
