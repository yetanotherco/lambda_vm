	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Basic arithmetic test - 32 instructions (power of 2)
	# Tests ADD, SUB, ADDW, SUBW with various edge cases
	#
	# Uses registers:
	#   t0-t6: test values and intermediates
	#   a0-a7, s0-s11: results

	# === Setup: Load test values (8 instructions) ===
	addi	t0, zero, 10		# 1. t0 = 10
	addi	t1, zero, 20		# 2. t1 = 20
	addi	t2, zero, -1		# 3. t2 = -1 (0xFFFFFFFFFFFFFFFF)
	addi	t3, zero, 1		# 4. t3 = 1
	addi	t4, zero, 100		# 5. t4 = 100
	addi	t5, zero, -100		# 6. t5 = -100
	lui	t6, 0x80000		# 7. t6 = 0x80000000 (32-bit min signed)
	addi	s0, zero, 0		# 8. s0 = 0

	# === 64-bit ADD tests (8 instructions) ===
	add	a0, t0, t1		# 9.  10 + 20 = 30
	add	a1, t0, t2		# 10. 10 + (-1) = 9
	add	a2, t2, t3		# 11. -1 + 1 = 0 (wrap)
	add	a3, t4, t5		# 12. 100 + (-100) = 0
	add	a4, t2, t2		# 13. -1 + (-1) = -2
	add	a5, t3, t3		# 14. 1 + 1 = 2
	add	a6, s0, t0		# 15. 0 + 10 = 10
	add	a7, t1, t2		# 16. 20 + (-1) = 19

	# === 64-bit SUB tests (8 instructions) ===
	sub	s1, t1, t0		# 17. 20 - 10 = 10
	sub	s2, t0, t1		# 18. 10 - 20 = -10
	sub	s3, s0, t3		# 19. 0 - 1 = -1 (underflow)
	sub	s4, t2, t2		# 20. -1 - (-1) = 0
	sub	s5, t4, t5		# 21. 100 - (-100) = 200
	sub	s6, t5, t4		# 22. -100 - 100 = -200
	sub	s7, t3, t2		# 23. 1 - (-1) = 2
	sub	s8, t2, t3		# 24. -1 - 1 = -2

	# === 32-bit word operations with sign extension (7 instructions) ===
	addw	s9, t0, t1		# 25. ADDW: 10 + 20 = 30
	addw	s10, t6, t3		# 26. ADDW: 0x80000000 + 1 = 0x80000001 (sign ext to neg)
	addw	s11, t2, t3		# 27. ADDW: -1 + 1 = 0 (32-bit)
	subw	gp, t0, t1		# 28. SUBW: 10 - 20 = -10 (sign ext)
	subw	tp, t1, t0		# 29. SUBW: 20 - 10 = 10
	subw	ra, s0, t3		# 30. SUBW: 0 - 1 = -1 (sign ext)
	subw	sp, t6, t3		# 31. SUBW: 0x80000000 - 1 = 0x7FFFFFFF (pos)

	# === Return ===
	li	a7, 5
	ecall		# 32. Return
