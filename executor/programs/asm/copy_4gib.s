	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# The COPY chip across 2^32 (low word 0 of high word 1): memcpy and memmove
	# whose source or destination crosses it (and one whose source's first word,
	# unread, is the last below it), a memset across it and a commit from a
	# source across it; then loads check what memory holds.
	# memcpy is ecall -30, memset -31, write 64; x10 = dst, x11 = src, x12 = n.
	# A pattern on both sides of the line, and a source further below.
	li	t0, 0xffffffc0
	li	t1, 0x2ce32d2335df4552
	sd	t1, 0(t0)
	li	t0, 0xffffffc8
	li	t1, 0xca18dd5ae3c45eb9
	sd	t1, 0(t0)
	li	t0, 0xffffffd0
	li	t1, 0x860f53667996eed4
	sd	t1, 0(t0)
	li	t0, 0xffffffd8
	li	t1, 0xe21299d828cea893
	sd	t1, 0(t0)
	li	t0, 0xffffffe0
	li	t1, 0xc60c9ae76aeb1026
	sd	t1, 0(t0)
	li	t0, 0xffffffe8
	li	t1, 0x44096810dda31fd
	sd	t1, 0(t0)
	li	t0, 0xfffffff0
	li	t1, 0xf750ba5cd9cfcdc8
	sd	t1, 0(t0)
	li	t0, 0xfffffff8
	li	t1, 0xd85091117d6ce577
	sd	t1, 0(t0)
	li	t0, 0x100000000
	li	t1, 0xc11bf36eba4e03a
	sd	t1, 0(t0)
	li	t0, 0x100000008
	li	t1, 0xdf922c6e861b181
	sd	t1, 0(t0)
	li	t0, 0x100000010
	li	t1, 0xe3ffaef9e3b7b3fc
	sd	t1, 0(t0)
	li	t0, 0x100000018
	li	t1, 0xb8641c0ab2a8289b
	sd	t1, 0(t0)
	li	t0, 0x100000020
	li	t1, 0x76197f2e7152898e
	sd	t1, 0(t0)
	li	t0, 0x100000028
	li	t1, 0x38d0dcb70ccd2145
	sd	t1, 0(t0)
	li	t0, 0x100000030
	li	t1, 0x7f2c0d6bd4209570
	sd	t1, 0(t0)
	li	t0, 0x100000038
	li	t1, 0x3cc17db3a57455ff
	sd	t1, 0(t0)
	li	t0, 0xfffffe00
	li	t1, 0x132bb33884ef2022
	sd	t1, 0(t0)
	li	t0, 0xfffffe08
	li	t1, 0x5b647112c22b0549
	sd	t1, 0(t0)
	li	t0, 0xfffffe10
	li	t1, 0x4107f48f3013a624
	sd	t1, 0(t0)
	li	t0, 0xfffffe18
	li	t1, 0x6d8dd69a8d3b91a3
	sd	t1, 0(t0)
	li	t0, 0xfffffe20
	li	t1, 0x3427b9497ffef7f6
	sd	t1, 0(t0)
	li	t0, 0xfffffe28
	li	t1, 0x26b1c32b4e6a218d
	sd	t1, 0(t0)
	li	t0, 0xfffffe30
	li	t1, 0x8ad2b586c9c55a18
	sd	t1, 0(t0)
	li	t0, 0xfffffe38
	li	t1, 0x19eff1de6c23f87
	sd	t1, 0(t0)
	# memcpy(2^32 - 0x100 + 6, 2^32 - 13, 29): the source crosses
	li	a0, 0xffffff06
	li	a1, 0xfffffff3
	li	a2, 29
	li	a7, -30
	ecall
	# memcpy(2^32 - 11, 2^32 - 0x200 + 3, 27): the destination crosses
	li	a0, 0xfffffff5
	li	a1, 0xfffffe03
	li	a2, 27
	li	a7, -30
	ecall
	# memcpy(2^32 - 0x300 + 6, 2^32 + 3, 5): the first source word is 2^32 - 8
	li	a0, 0xfffffd06
	li	a1, 0x100000003
	li	a2, 5
	li	a7, -30
	ecall
	# memmove(2^32 - 13, 2^32 - 0x40 + 3, 64): the destination crosses
	li	a0, 0xfffffff3
	li	a1, 0xffffffc3
	li	a2, 64
	li	a7, -30
	ecall
	# memset from 2^32 - 13, 30 bytes: across the line
	li	a0, 0xfffffffb
	li	a1, 0xfffffff3
	li	a2, 30
	li	a7, -31
	ecall
	# write(1, 2^32 - 5, 12): a commit from a source across the line
	li	a0, 1
	li	a1, 0xfffffffb
	li	a2, 12
	li	a7, 64
	ecall
	# What memory holds, word by word and across the line.
	li	t0, 0xffffff00
	ld	a0, 0(t0)
	li	t2, 0x5cd9000000000000
	bne	a0, t2, .Lfail
	li	t0, 0xffffff08
	ld	a0, 0(t0)
	li	t2, 0x117d6ce577f750ba
	bne	a0, t2, .Lfail
	li	t0, 0xffffff10
	ld	a0, 0(t0)
	li	t2, 0x36eba4e03ad85091
	bne	a0, t2, .Lfail
	li	t0, 0xfffffd00
	ld	a0, 0(t0)
	li	t2, 0x13a6000000000000
	bne	a0, t2, .Lfail
	li	t0, 0xfffffd08
	ld	a0, 0(t0)
	li	t2, 0xf48f30
	bne	a0, t2, .Lfail
	li	t0, 0xffffffe0
	ld	a0, 0(t0)
	li	t2, 0xc60c9ae76aeb1026
	bne	a0, t2, .Lfail
	li	t0, 0xffffffe8
	ld	a0, 0(t0)
	li	t2, 0x44096810dda31fd
	bne	a0, t2, .Lfail
	li	t0, 0xfffffff0
	ld	a0, 0(t0)
	li	t2, 0x2ce32d2335cfcdc8
	bne	a0, t2, .Lfail
	li	t0, 0xfffffff8
	ld	a0, 0(t0)
	li	t2, 0x2ce32d2335c45eb9
	bne	a0, t2, .Lfail
	li	t0, 0x100000000
	ld	a0, 0(t0)
	li	t2, 0x2ce32d2335c45eb9
	bne	a0, t2, .Lfail
	li	t0, 0x100000008
	ld	a0, 0(t0)
	li	t2, 0x2ce32d2335c45eb9
	bne	a0, t2, .Lfail
	li	t0, 0x100000010
	ld	a0, 0(t0)
	li	t2, 0x2ce32d2335c45eb9
	bne	a0, t2, .Lfail
	li	t0, 0x100000018
	ld	a0, 0(t0)
	li	t2, 0x44096810dda31b9
	bne	a0, t2, .Lfail
	li	t0, 0x100000020
	ld	a0, 0(t0)
	li	t2, 0xb338845cd9cfcdc8
	bne	a0, t2, .Lfail
	li	t0, 0xfffffffc
	ld	a0, 0(t0)
	li	t2, 0x35c45eb92ce32d23
	bne	a0, t2, .Lfail
	li	t0, 0xfffffff5
	ld	a0, 0(t0)
	li	t2, 0x2335c45eb92ce32d
	bne	a0, t2, .Lfail
	li	t0, 0xffffffff
	ld	a0, 0(t0)
	li	t2, 0xe32d2335c45eb92c
	bne	a0, t2, .Lfail
	li	a0, 0
	li	a7, 93
	ecall
.Lfail:
	li	a0, 1
	li	a7, 93
	ecall
