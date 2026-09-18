	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Same 5·G computation as `test_ecsm`, but the 96-byte output buffer is
	# pointed AT the inputs: out = sp+0 covers xG (sp+0..32) and k (sp+32..64).
	# The ecall reads both operands before it writes anything (reads at T and
	# T+1, xR at T+2, yR and yG at T+3), so the result must be the same as the
	# disjoint case even though every input byte is overwritten. That ordering
	# is what keeps each per-address chain monotone, and it is only proven if a
	# proof of this program verifies.
	addi	sp, sp, -160

	# xG = secp256k1 Gx, little-endian (4 doublewords).
	li	t0, 0x59F2815B16F81798
	sd	t0, 0(sp)
	li	t0, 0x029BFCDB2DCE28D9
	sd	t0, 8(sp)
	li	t0, 0x55A06295CE870B07
	sd	t0, 16(sp)
	li	t0, 0x79BE667EF9DCBBAC
	sd	t0, 24(sp)

	# k = 5 (little-endian).
	li	t0, 5
	sd	t0, 32(sp)
	sd	zero, 40(sp)
	sd	zero, 48(sp)
	sd	zero, 56(sp)

	# ECSM ecall: a0 = &out (96 bytes, aliasing both inputs), a1 = &xG,
	# a2 = &k, a7 = -11.
	addi	a0, sp, 0
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -11
	ecall

	# Commit xR, which now sits where xG used to be.
	# Commit syscall: a0 = fd(1), a1 = buf_addr, a2 = count, a7 = 64.
	li	a0, 1
	addi	a1, sp, 0
	li	a2, 32
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 160
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
