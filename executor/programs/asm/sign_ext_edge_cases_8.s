	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Edge cases for arg1/arg2 sign extension in word instructions
	# Tests the constraint: arg[4:] = (2^32-1) * sign_bit * signed
	#
	# Key test: arg2 sign extension for signed word ops (DIVW, REMW)
	# This wasn't triggered by all_instructions_64.s where arg2 values
	# were small positive numbers.

	# === 1-2: Setup values ===
	addi	t0, zero, 100		# 1: t0 = 100 (dividend)
	lui	t1, 0xFFFFF		# 2: t1 = 0xFFFFF000 (will become -4096 after addi)

	# === 3-4: Create negative divisor with bit 31 set ===
	addi	t1, t1, -16		# 3: t1 = 0xFFFFEFF0 = -4112 (bit 31 SET)
	addi	t2, zero, -5		# 4: t2 = -5 (another negative for testing)

	# === 5-6: DIVW with negative divisor - triggers arg2 sign extension ===
	# DIVW: word_instr=1, signed=1
	# arg1 = t0 = 100 (positive, no sign ext needed)
	# arg2 = t1 = 0xFFFFEFF0 (bit 31 SET, needs sign extension!)
	# Expected: 100 / -4112 = 0 (integer division)
	divw	a0, t0, t1		# 5: DIVW with arg2 bit 31 set

	# === 7: REMW with negative divisor - also triggers arg2 sign ext ===
	remw	a1, t0, t1		# 6: REMW: 100 % -4112 = 100

	# === 7-8: Finalize ===
	addi	a2, a0, 0		# 7: Copy result (NOP-like)
	jalr	zero, 0(zero)		# 8: Halt
