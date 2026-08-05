	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# xR is written at low limb 0xFFFF_FFE1 (high limb 0), so its last doubleword
	# starts at +24 = 0xFFFF_FFF9 and its trailing bytes cross 2^32. Every base the
	# ECSM AIR derives stays inside the limb; MEMW's carry columns place the crossing
	# bytes at 0x1_0000_0000. Reading them back proves they landed there.
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

	# t1 = 2^32 - 31 = 0xFFFF_FFE1.
	li	t1, 1
	slli	t1, t1, 32
	addi	t1, t1, -31

	# ECSM ecall: a0 = &xR (crossing), a1 = &xG, a2 = &k, a7 = -11.
	addi	a0, t1, 0
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -11
	ecall

	# Read xR back from the crossing address into the stack buffer. The last load
	# spans 0xFFFF_FFF9..0x1_0000_0000.
	ld	t2, 0(t1)
	sd	t2, 64(sp)
	ld	t2, 8(t1)
	sd	t2, 72(sp)
	ld	t2, 16(t1)
	sd	t2, 80(sp)
	ld	t2, 24(t1)
	sd	t2, 88(sp)

	# Commit the read-back bytes so the test can compare them against x(5G).
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
