	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Test ALL branch instructions (16 instructions)
	# Covers: BEQ, BNE, BLT, BGE, BLTU, BGEU
	# Each branch is set to NOT take (fall through) to maintain count

	# === 1-3: Setup ===
	addi	t0, zero, 10		# 1: t0 = 10
	addi	t1, zero, 20		# 2: t1 = 20
	addi	a0, zero, 0		# 3: a0 = 0 (counter)

	# === 4-15: Branch tests (all fall through) ===
	beq	t0, t1, skip1		# 4: 10 == 20? No, fall through
	addi	a0, a0, 1		# 5: a0 = 1 (executed)
skip1:
	bne	t0, t0, skip2		# 6: 10 != 10? No, fall through
	addi	a0, a0, 1		# 7: a0 = 2 (executed)
skip2:
	blt	t1, t0, skip3		# 8: 20 < 10? No, fall through
	addi	a0, a0, 1		# 9: a0 = 3 (executed)
skip3:
	bge	t0, t1, skip4		# 10: 10 >= 20? No, fall through
	addi	a0, a0, 1		# 11: a0 = 4 (executed)
skip4:
	bltu	t1, t0, skip5		# 12: 20 <u 10? No, fall through
	addi	a0, a0, 1		# 13: a0 = 5 (executed)
skip5:
	bgeu	t0, t1, done		# 14: 10 >=u 20? No, fall through
	addi	a0, a0, 1		# 15: a0 = 6 (executed)
done:
	li	a0, 0
	li	a7, 93
	ecall		# 16: Halt
