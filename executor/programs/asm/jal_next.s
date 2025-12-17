	.attribute	5, "rv32i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	jal     zero, 4
	jalr	zero, 0(ra)
	addi    a0, zero, 1
	jalr	zero, 0(ra)
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
