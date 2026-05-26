	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Misaligned SH: 2-byte store at address 67.
	li	t0, 64
	li	t1, 0xAA12
	sh	t1, 3(t0)
	lwu	a1, 0(t0)
	li	t2, 0x12000000
	bne	a1, t2, .Lfail
	lwu	a1, 4(t0)
	li	t2, 0x000000AA
	bne	a1, t2, .Lfail
	li	a0, 0
	li	a7, 93
	ecall
.Lfail:
	li	a0, 1
	li	a7, 93
	ecall
