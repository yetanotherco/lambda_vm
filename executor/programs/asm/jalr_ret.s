	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	jal     a2, 4
	jalr    a0, 4(a2)
	jalr	zero, 0(ra)
	addi    a0, zero, 1
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
