	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	li	a2, -2            # load -2
	xori	a0, a2, -1        # XOR with -1 gives 1 (bit flip)
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
