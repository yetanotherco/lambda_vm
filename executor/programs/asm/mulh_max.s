	.attribute	5, "rv32i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
    addi	a0, zero, 1
    addi	a1, zero, 2
	addi	a2, zero, 200
	addi	a3, zero, 0xFFFFFFFF
	divu    a4, a3, a1
	sub     a5, a4, a0
	mulhu    a0, a2, a5
	jalr	zero, 0(ra)
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
