	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# 176 bytes on the stack for the BLAKE3 state region (22 x u64):
	# h[4 dwords] | m[8] | t[1] | block_len,flags[1] | out[8].
	addi	sp, sp, -176

	# Deterministic non-zero seed over the 14 input dwords: dword[k] = k + 1.
	# (t therefore = 13, block_len = 14, flags = 0 — arbitrary but fixed.)
	mv	t0, sp
	li	t1, 1
	li	t2, 15
.Linit_loop:
	sd	t1, 0(t0)
	addi	t0, t0, 8
	addi	t1, t1, 1
	bne	t1, t2, .Linit_loop

	# First compression.
	# a0 = pointer to the 176-byte region (8-aligned)
	# a7 = syscall number (u64::MAX - 2 = -3)
	mv	a0, sp
	li	a7, -3
	ecall

	# Chain: copy out (8 dwords at sp+112) over m (8 dwords at sp+32), so the
	# second call consumes the first call's output AND its out-region write has
	# non-zero previous content.
	li	t1, 0
.Lcopy_loop:
	slli	t2, t1, 3
	addi	t3, sp, 112
	add	t3, t3, t2
	ld	t4, 0(t3)
	addi	t3, sp, 32
	add	t3, t3, t2
	sd	t4, 0(t3)
	addi	t1, t1, 1
	li	t2, 8
	bne	t1, t2, .Lcopy_loop

	# Second compression.
	mv	a0, sp
	li	a7, -3
	ecall

	# Commit the final 64-byte output.
	li	a0, 1
	addi	a1, sp, 112
	li	a2, 64
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 176
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end0:
	.size	main, .Lfunc_end0-main
