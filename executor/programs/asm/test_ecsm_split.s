	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Like test_ecsm.s, but the ECSM pointer registers (a0=&xR, a1=&xG, a2=&k)
	# are set at the very START and never rewritten before the ecall. With a small
	# continuation epoch size the ecall lands in a LATER epoch than the one that set
	# the pointers, so the per-epoch touched-cell pass must carry registers across
	# the boundary to compute the right addresses.
	addi	sp, sp, -96
	addi	a0, sp, 64
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -11

	# xG = secp256k1 Gx, little-endian (4 doublewords). The heavy 64-bit immediates
	# act as natural filler between the pointer setup and the ecall.
	li	t0, 0x59F2815B16F81798
	sd	t0, 0(sp)
	li	t0, 0x029BFCDB2DCE28D9
	sd	t0, 8(sp)
	li	t0, 0x55A06295CE870B07
	sd	t0, 16(sp)
	li	t0, 0x79BE667EF9DCBBAC
	sd	t0, 24(sp)

	# k = 5 (little-endian); exercises double, double, add.
	li	t0, 5
	sd	t0, 32(sp)
	sd	zero, 40(sp)
	sd	zero, 48(sp)
	sd	zero, 56(sp)

	# ECSM ecall: a0/a1/a2 were set far above (possibly in an earlier epoch).
	ecall

	# Commit the 32-byte result xR so the test can check it equals x(5G).
	li	a0, 1
	addi	a1, sp, 64
	li	a2, 32
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 96
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
