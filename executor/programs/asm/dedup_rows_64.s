	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# 64 loop iterations whose operands change every time, so each of the six
	# deduplicated tables gets about one unique row per iteration and instruction:
	#   MUL (mul, mulhu), DVRM (div, remu), LT (slt, sltu, blt, and DVRM's
	#   remainder checks), BYTEWISE (and, or, xor), EQ (beq, bne) and BRANCH
	#   (the taken bne and blt, whose rs1 value is part of the row).
	# The row-order tests need tables this wide: two HashMap orders of a table
	# with a handful of rows can coincide by chance, and 64 rows cannot.

	# === Setup ===
	addi	t0, zero, 1		# i = 1
	addi	t1, zero, 65		# the bound (i runs 1..=64)
	lui	t2, 0x12345		# K = 0x12345678
	addi	t2, t2, 0x678
	addi	s0, zero, 0		# accumulator

loop:
	mul	t3, t0, t2		# MUL: i * K
	mulhu	t4, t3, t2		# MUL: high word of (i * K) * K
	div	t5, t3, t0		# DVRM: (i * K) / i, signed
	remu	t6, t3, t1		# DVRM: (i * K) % 65, unsigned
	slt	a1, t5, t3		# LT: signed
	sltu	a2, t0, t6		# LT: unsigned
	and	a3, t3, t0		# BYTEWISE: AND
	or	a4, t3, t5		# BYTEWISE: OR
	xor	a5, t3, t4		# BYTEWISE: XOR
	beq	t3, t4, skip		# EQ: never equal, falls through
	add	s0, s0, a5
skip:
	bne	t0, t1, next		# EQ (inverted) + BRANCH: always taken, to the next instruction
next:
	addi	t0, t0, 1
	blt	t0, t1, loop		# LT + BRANCH: taken 63 times

	# === Halt ===
	li	a0, 0
	li	a7, 93
	ecall
