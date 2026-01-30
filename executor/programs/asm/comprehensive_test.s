	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Comprehensive test of VM operations without branching
	# Target: 32 instructions (power of 2)
	#
	# Uses registers:
	#   t0-t6: test values
	#   a0-a7: results
	#   s0-s5: more results

	# === Setup test values ===
	# 1. Load small positive
	addi	t0, zero, 10		# t0 = 10
	# 2. Load small positive 2
	addi	t1, zero, 20		# t1 = 20
	# 3. Load -1 (all bits set)
	addi	t2, zero, -1		# t2 = -1 = 0xFFFFFFFFFFFFFFFF
	# 4. Load 1
	addi	t3, zero, 1		# t3 = 1
	# 5. Load large value (upper bits)
	lui	t4, 0x80000		# t4 = 0x80000000 (bit 31 set)
	# 6. Load shift amount
	addi	t5, zero, 4		# t5 = 4 (for shifts)

	# === 64-bit ADD operations ===
	# 7. Basic add: 10 + 20 = 30
	add	a0, t0, t1		# a0 = 30
	# 8. Add with negative: 10 + (-1) = 9
	add	a1, t0, t2		# a1 = 9
	# 9. Add causing 64-bit wrap: -1 + 1 = 0
	add	a2, t2, t3		# a2 = 0

	# === 64-bit SUB operations ===
	# 10. Basic sub: 20 - 10 = 10
	sub	a3, t1, t0		# a3 = 10
	# 11. Sub negative result: 10 - 20 = -10
	sub	a4, t0, t1		# a4 = -10 = 0xFFFFFFFFFFFFFFF6
	# 12. Sub underflow: 0 - 1 = -1
	sub	a5, zero, t3		# a5 = -1 = 0xFFFFFFFFFFFFFFFF

	# === 32-bit word operations (sign extend) ===
	# 13. ADDW: 10 + 20 = 30 (32-bit, sign-extend)
	addw	a6, t0, t1		# a6 = 30
	# 14. SUBW with negative result
	subw	a7, t0, t1		# a7 = -10 (sign-extended to 64 bits)

	# === Multiplication ===
	# 15. Basic mul: 10 * 20 = 200
	mul	s0, t0, t1		# s0 = 200
	# 16. MULW (32-bit): 10 * 20 = 200
	mulw	s1, t0, t1		# s1 = 200

	# === Division and Remainder ===
	# 17. DIV: 20 / 10 = 2
	div	s2, t1, t0		# s2 = 2
	# 18. REM: 20 % 10 = 0
	rem	s3, t1, t0		# s3 = 0

	# === Shift operations ===
	# 19. SLL: 1 << 4 = 16
	sll	s4, t3, t5		# s4 = 16
	# 20. SLLI: 1 << 32 (tests 64-bit shift)
	slli	s5, t3, 32		# s5 = 0x100000000
	# 21. SRL: 16 >> 2 (logical right)
	srli	s6, s4, 2		# s6 = 4
	# 22. SRA: -1 >> 4 (arithmetic right, stays -1)
	sra	s7, t2, t5		# s7 = -1 (sign bit preserved)

	# === Bitwise operations ===
	# 23. ANDI: 0xFF & 0x0F = 0x0F
	addi	s8, zero, 0xFF		# s8 = 255
	# 24.
	andi	s9, s8, 0x0F		# s9 = 15
	# 25. ORI: 0x0F | 0xF0 = 0xFF
	ori	s10, s9, 0xF0		# s10 = 255
	# 26. XORI: 0xFF ^ 0x0F = 0xF0
	xori	s11, s10, 0x0F		# s11 = 240

	# === Comparisons (set less than) ===
	# 27. SLTI: 10 < 20 ? 1 : 0 = 1
	slti	t6, t0, 20		# t6 = 1
	# 28. SLTI: 20 < 10 ? 1 : 0 = 0
	slti	gp, t1, 10		# gp = 0
	# 29. SLTIU (unsigned): -1 < 10 ? (unsigned comparison, -1 is huge)
	sltiu	tp, t2, 10		# tp = 0 (0xFFFF... > 10 unsigned)

	# === Edge cases ===
	# 30. ADDI with max immediate
	addi	ra, zero, 2047		# ra = 2047 (max 12-bit signed imm)
	# 31. ADDI with min immediate
	addi	sp, zero, -2048		# sp = -2048 (min 12-bit signed imm)

	# === Return ===
	# 32. Return (a0 still has result from first add = 30)
	li	a7, 5
	ecall
