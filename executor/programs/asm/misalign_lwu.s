	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Misaligned LWU: 4-byte load at address 33.
	li	t0, 32
	li	t1, 0x80FF1234
	sw	t1, 0(t0)
	li	t1, 0x000000AA
	sw	t1, 4(t0)
	lwu	a0, 1(t0)
	li	t2, 0xAA80FF12
	bne	a0, t2, .Lfail
	li	a0, 0
	li	a7, 93
	ecall
.Lfail:
	li	a0, 1
	li	a7, 93
	ecall
