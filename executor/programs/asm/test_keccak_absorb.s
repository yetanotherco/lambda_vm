	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 608 bytes on the stack: 200-byte keccak state at sp, then
	# 3 x 136-byte rate blocks at sp+200 (regions disjoint, both 8-aligned).
	addi	sp, sp, -608

	# Deterministic non-zero state: lane[i] = i + 1 (25 lanes).
	# The host test replays the sponge over tiny-keccak from this seed.
	mv	t0, sp
	li	t1, 1
	li	t2, 26
.Lstate_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Lstate_loop

	# Deterministic message data: dword[k] = k + 100 (51 dwords = 3 blocks).
	addi	t0, sp, 200
	li	t1, 100
	li	t2, 151
.Ldata_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Ldata_loop

	# Absorb all 3 blocks in ONE ecall.
	# a0 = state, a1 = data, a2 = n_blocks, a7 = u64::MAX - 3 (spec -4).
	mv	a0, sp
	addi	a1, sp, 200
	li	a2, 3
	li	a7, -4
	ecall

	# Commit the final 200-byte state.
	li	a0, 1
	mv	a1, sp
	li	a2, 200
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 608
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
