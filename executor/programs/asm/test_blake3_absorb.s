	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 256 bytes on the stack, in two disjoint regions the absorb ABI requires:
	#   sp+0   .. sp+64   the 64-byte control region (cv_in | cv_out)
	#   sp+64  .. sp+256  three 64-byte message blocks, read in place
	# Both are 8-aligned because sp is.
	addi	sp, sp, -256

	# cv_in = dwords 1..4 at sp+0.
	mv	t0, sp
	li	t1, 1
	li	t2, 5
.Lcv_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Lcv_loop

	# Message = dwords 100..123 at sp+64 (24 dwords = 3 blocks).
	addi	t0, sp, 64
	li	t1, 100
	li	t2, 124
.Lmsg_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Lmsg_loop

	# Absorb all three blocks in ONE ecall — a group of 3 compression rows plus
	# its END row.
	# a0 = control region, a1 = message, a2 = num_blocks, a3 = first_flags,
	# a7 = syscall number (u64::MAX - 3 = -4).
	mv	a0, sp
	addi	a1, sp, 64
	li	a2, 3
	li	a3, 1
	li	a7, -4
	ecall

	# Chain into a SECOND absorb: copy cv_out (sp+32) over cv_in (sp+0). The
	# second ecall's END row then writes cv_out where the first one already
	# wrote, so its Memw `old` is non-zero rather than fresh memory.
	li	t1, 0
.Lchain_loop:
	slli	t2, t1, 3
	addi	t3, sp, 32
	add	t3, t3, t2
	ld	t4, 0(t3)
	mv	t3, sp
	add	t3, t3, t2
	sd	t4, 0(t3)
	addi	t1, t1, 1
	li	t2, 4
	bne	t1, t2, .Lchain_loop

	# The degenerate group: one block, interior flags. Re-reads the message's
	# second block, so the same addresses are touched at a second timestamp.
	mv	a0, sp
	addi	a1, sp, 128
	li	a2, 1
	li	a3, 0
	li	a7, -4
	ecall

	# Commit the 32-byte chaining value.
	li	a0, 1
	addi	a1, sp, 32
	li	a2, 32
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 256
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
