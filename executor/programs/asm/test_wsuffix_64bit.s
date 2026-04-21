	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Exercises W-suffix instructions (ADDIW, SRLIW) on a register holding
	# a 64-bit value with non-zero upper 32 bits. The executor's Log must
	# store the full 64-bit register value in src1_val/src2_val so the
	# prover's MEMW_R Memory bus chain stays consistent.

	# Load a 64-bit value with non-zero hi32 into a0 (x10).
	# 0xDEADBEEF_12345678
	li	t0, 0xDEADBEEF		# t0 = 0xDEADBEEF (sign-extended)
	slli	t0, t0, 32		# t0 = 0xDEADBEEF_00000000
	li	t1, 0x12345678
	or	a0, t0, t1		# a0 = 0xDEADBEEF_12345678

	# Execute ADDIW on a0 — reads a0 (64-bit) but operates on lower 32.
	# If src1_val is truncated to 32 bits, the upper 0xDEADBEEF is lost and
	# the prover's MEMW_R chain for x10 word 1 won't balance.
	addiw	t2, a0, 1		# t2 = sign_extend32(0x12345678 + 1) = 0x12345679

	# Execute SRLIW — another W-suffix that reads a0.
	srliw	t3, a0, 4		# t3 = sign_extend32(0x12345678 >> 4) = 0x01234567

	# Commit 8 bytes of a0 (the original 64-bit value should be intact).
	# a0 was never written by ADDIW/SRLIW (they write t2/t3, not a0).
	addi	a1, sp, -8		# buf on stack
	sd	a0, 0(a1)		# store a0 to buf
	li	a0, 1			# fd = 1
	li	a2, 8			# count = 8
	li	a7, 64			# Commit
	ecall

	# Halt
	li	a0, 0
	li	a7, 93
	ecall
