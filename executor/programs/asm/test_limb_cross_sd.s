	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	li	t1, 1
	slli	t1, t1, 32
	addi	t2, t1, -4        # 0xFFFFFFFC: an 8-byte store here spans into hi=1
	li	t0, 0x0123456789ABCDEF
	sd	t0, 0(t2)
	ld	t3, 0(t2)
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
