	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	li	t0, 0
	li	t1, 1
	addiw	a0, zero, -1
	li	a0, 0
	li	a7, 93
	ecall
