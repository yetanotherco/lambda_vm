	.attribute	5, "rv32i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	addi	a2, zero, 0xFFFFFFFF # -1
	andi	a0, a2, 0xFFFFFFFF
	jalr	zero, 0(ra)
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
