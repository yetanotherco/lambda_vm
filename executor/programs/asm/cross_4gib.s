	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Loads and stores across 2^32: t0 is the last aligned word of the low
	# 4 GiB range, 8(t0) the first word of high word 1. Every crossing width,
	# signed and unsigned, from that last word; a crossing load into it.
	li	t0, 0xFFFFFFF8
	li	t1, 0x807060504030201
	sd	t1, 0(t0)
	li	t1, 0x100f0e0d0c0b0a09
	sd	t1, 8(t0)
	ld	a0, 4(t0)
	li	t2, 0xc0b0a0908070605
	bne	a0, t2, .Lfail
	ld	a0, 1(t0)
	li	t2, 0x908070605040302
	bne	a0, t2, .Lfail
	lw	a0, 5(t0)
	li	t2, 0x9080706
	bne	a0, t2, .Lfail
	lwu	a0, 6(t0)
	li	t2, 0xa090807
	bne	a0, t2, .Lfail
	lh	a0, 7(t0)
	li	t2, 0x908
	bne	a0, t2, .Lfail
	lhu	a0, 7(t0)
	li	t2, 0x908
	bne	a0, t2, .Lfail
	li	t1, 0xa1a2a3a4a5a6a7a8
	sd	t1, 3(t0)
	ld	a0, 0(t0)
	li	t2, 0xa4a5a6a7a8030201
	bne	a0, t2, .Lfail
	ld	a0, 8(t0)
	li	t2, 0x100f0e0d0ca1a2a3
	bne	a0, t2, .Lfail
	li	t1, 0x8899aabb
	sw	t1, 6(t0)
	li	t1, 0x9988
	sh	t1, 7(t0)
	lh	a0, 7(t0)
	li	t2, 0xffffffffffff9988
	bne	a0, t2, .Lfail
	lw	a0, 5(t0)
	li	t2, 0xffffffff9988bba6
	bne	a0, t2, .Lfail
	ld	a0, 4(t0)
	li	t2, 0xca1889988bba6a7
	bne	a0, t2, .Lfail
	lb	a0, 9(t0)
	li	t2, 0xffffffffffffff88
	bne	a0, t2, .Lfail
	ld	a0, -4(t0)
	li	t2, 0xa803020100000000
	bne	a0, t2, .Lfail
	li	a0, 0
	li	a7, 93
	ecall
.Lfail:
	li	a0, 1
	li	a7, 93
	ecall
