	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
    addi	a0, zero, 1
    addi	a1, zero, 2
	addi	a2, zero, -200
	li	a3, -1
	divu    a4, a3, a1
	sub     a5, a4, a0
	mulhsu    a0, a2, a5
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
