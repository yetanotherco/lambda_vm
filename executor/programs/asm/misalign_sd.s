	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Misaligned SD: 8-byte store at address 65.
	li	t0, 64
	li	t1, 0x123456789ABCDEF0
	sd	t1, 1(t0)
	lwu	a1, 0(t0)
	li	t2, 0xBCDEF000
	bne	a1, t2, .Lfail
	lwu	a1, 4(t0)
	li	t2, 0x3456789A
	bne	a1, t2, .Lfail
	lwu	a1, 8(t0)
	li	t2, 0x00000012
	bne	a1, t2, .Lfail
	li	a0, 0
	li	a7, 93
	ecall
.Lfail:
	li	a0, 1
	li	a7, 93
	ecall
