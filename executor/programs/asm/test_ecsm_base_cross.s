	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# xR is written at low limb 0xFFFF_FFFC, so the DERIVED doubleword bases carry into
	# the high limb: addr[0] = 0x0_FFFF_FFFC, addr[1] = 0x1_0000_0004, addr[2] =
	# 0x1_0000_000C, addr[3] = 0x1_0000_0014. Only the per-access address columns can
	# express that — the old `ADDR_*_0 + 8i` derivation would put a non-canonical low
	# limb on the Memory bus for i = 1..3 and the trace would not balance. Byte carries
	# inside a doubleword are exercised too: bytes 4..7 of addr[0] land past 2^32.
	#
	# Stack layout (96 bytes): xG at sp+0, k at sp+32, read-back buffer at sp+64.
	addi	sp, sp, -96

	# xG = secp256k1 Gx, little-endian.
	li	t0, 0x59F2815B16F81798
	sd	t0, 0(sp)
	li	t0, 0x029BFCDB2DCE28D9
	sd	t0, 8(sp)
	li	t0, 0x55A06295CE870B07
	sd	t0, 16(sp)
	li	t0, 0x79BE667EF9DCBBAC
	sd	t0, 24(sp)

	# k = 5.
	li	t0, 5
	sd	t0, 32(sp)
	sd	zero, 40(sp)
	sd	zero, 48(sp)
	sd	zero, 56(sp)

	# t1 = 2^32 - 4 = 0xFFFF_FFFC.
	li	t1, 1
	slli	t1, t1, 32
	addi	t1, t1, -4

	# ECSM ecall: a0 = &xR (bases carry), a1 = &xG, a2 = &k, a7 = -11.
	addi	a0, t1, 0
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -11
	ecall

	# Read xR back and stage it where the commit syscall can reach it.
	ld	t2, 0(t1)
	sd	t2, 64(sp)
	ld	t2, 8(t1)
	sd	t2, 72(sp)
	ld	t2, 16(t1)
	sd	t2, 80(sp)
	ld	t2, 24(t1)
	sd	t2, 88(sp)

	li	a0, 1
	addi	a1, sp, 64
	li	a2, 32
	li	a7, 64
	ecall

	addi	sp, sp, 96
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
