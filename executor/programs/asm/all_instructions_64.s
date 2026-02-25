	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Comprehensive test of ALL RV64IM instructions (64 instructions, no branches)
	# This tests every ALU operation, load/store variant, and jump instruction
	#
	# Layout:
	#   1-8:   Setup values
	#   9-20:  64-bit arithmetic (ADD, SUB, SLT, SLTU)
	#   21-28: 32-bit word arithmetic (ADDW, SUBW)
	#   29-36: Shifts (SLL, SRL, SRA, SLLI, SRLI, SRAI)
	#   37-44: 32-bit word shifts (SLLW, SRLW, SRAW, SLLIW, SRLIW, SRAIW)
	#   45-52: Bitwise (AND, OR, XOR, ANDI, ORI, XORI)
	#   53-58: Multiplication (MUL, MULH, MULHSU, MULHU, MULW)
	#   59-62: Division/Remainder (DIV, DIVU, REM, REMU)
	#   63:    NOP (placeholder)
	#   64:    JALR (halt)

	# === 1-8: Setup test values ===
	addi	t0, zero, 10		# 1: t0 = 10
	addi	t1, zero, 20		# 2: t1 = 20
	addi	t2, zero, -1		# 3: t2 = -1 (all bits set)
	addi	t3, zero, 4		# 4: t3 = 4 (shift amount)
	lui	t4, 0x80000		# 5: t4 = 0x80000000 (bit 31 set)
	auipc	t5, 0			# 6: t5 = current PC
	addi	t6, zero, 0x55		# 7: t6 = 0x55 (pattern)
	addi	s0, zero, 0xFF		# 8: s0 = 0xFF

	# === 9-20: 64-bit arithmetic ===
	add	a0, t0, t1		# 9: a0 = 10 + 20 = 30
	add	a1, t0, t2		# 10: a1 = 10 + (-1) = 9
	sub	a2, t1, t0		# 11: a2 = 20 - 10 = 10
	sub	a3, t0, t1		# 12: a3 = 10 - 20 = -10
	slt	a4, t0, t1		# 13: a4 = (10 < 20) = 1
	slt	a5, t1, t0		# 14: a5 = (20 < 10) = 0
	sltu	a6, t0, t2		# 15: a6 = (10 <u -1) = 1 (unsigned)
	slti	a7, t0, 15		# 16: a7 = (10 < 15) = 1
	slti	s1, t0, 5		# 17: s1 = (10 < 5) = 0
	sltiu	s2, t0, 15		# 18: s2 = (10 <u 15) = 1
	addi	s3, t0, 100		# 19: s3 = 10 + 100 = 110
	addi	s4, t2, 1		# 20: s4 = -1 + 1 = 0

	# === 21-28: 32-bit word arithmetic ===
	addw	s5, t0, t1		# 21: s5 = 10 + 20 = 30 (32-bit)
	addw	s6, t4, t0		# 22: s6 = 0x80000000 + 10 (sign extends)
	subw	s7, t1, t0		# 23: s7 = 20 - 10 = 10 (32-bit)
	subw	s8, t0, t1		# 24: s8 = 10 - 20 = -10 (sign extends)
	addiw	s9, t0, 50		# 25: s9 = 10 + 50 = 60
	addiw	s10, t4, 1		# 26: s10 = 0x80000000 + 1 (sign extends)
	addw	s11, t2, t0		# 27: s11 = -1 + 10 = 9 (32-bit)
	subw	gp, zero, t0		# 28: gp = 0 - 10 = -10 (32-bit)

	# === 29-36: 64-bit shifts ===
	sll	a0, t0, t3		# 29: a0 = 10 << 4 = 160
	srl	a1, s0, t3		# 30: a1 = 0xFF >> 4 = 0x0F
	sra	a2, t2, t3		# 31: a2 = -1 >> 4 = -1 (arithmetic)
	slli	a3, t0, 8		# 32: a3 = 10 << 8 = 2560
	srli	a4, s0, 4		# 33: a4 = 0xFF >> 4 = 0x0F
	srai	a5, t2, 4		# 34: a5 = -1 >> 4 = -1 (arithmetic)
	slli	a6, t0, 32		# 35: a6 = 10 << 32 (64-bit shift)
	srli	a7, a6, 32		# 36: a7 = (10 << 32) >> 32 = 10

	# === 37-44: 32-bit word shifts ===
	sllw	s1, t0, t3		# 37: s1 = 10 << 4 = 160 (32-bit)
	srlw	s2, s0, t3		# 38: s2 = 0xFF >> 4 = 0x0F (32-bit)
	sraw	s3, t4, t3		# 39: s3 = 0x80000000 >> 4 (arith, sign ext)
	slliw	s4, t0, 8		# 40: s4 = 10 << 8 = 2560 (32-bit)
	srliw	s5, s0, 4		# 41: s5 = 0xFF >> 4 = 0x0F (32-bit)
	sraiw	s6, t4, 4		# 42: s6 = 0x80000000 >> 4 (arith, sign ext)
	slliw	s7, t0, 28		# 43: s7 = 10 << 28 (32-bit, may overflow)
	srliw	s8, t2, 16		# 44: s8 = 0xFFFFFFFF >> 16 = 0xFFFF (32-bit)

	# === 45-52: Bitwise operations ===
	and	s9, s0, t6		# 45: s9 = 0xFF & 0x55 = 0x55
	or	s10, t6, t0		# 46: s10 = 0x55 | 10 = 0x5F
	xor	s11, s0, t6		# 47: s11 = 0xFF ^ 0x55 = 0xAA
	andi	a0, s0, 0x0F		# 48: a0 = 0xFF & 0x0F = 0x0F
	ori	a1, t6, 0xF0		# 49: a1 = 0x55 | 0xF0 = 0xF5
	xori	a2, s0, 0x0F		# 50: a2 = 0xFF ^ 0x0F = 0xF0
	and	a3, t2, s0		# 51: a3 = -1 & 0xFF = 0xFF
	or	a4, zero, t6		# 52: a4 = 0 | 0x55 = 0x55

	# === 53-58: Multiplication ===
	mul	a5, t0, t1		# 53: a5 = 10 * 20 = 200
	mulw	a6, t0, t1		# 54: a6 = 10 * 20 = 200 (32-bit)
	mulh	a7, t0, t2		# 55: a7 = high64(10 * -1) = -1
	mulhu	s1, t0, t1		# 56: s1 = high64(10 * 20) = 0
	mulhsu	s2, t2, t0		# 57: s2 = high64(-1 * 10u)
	mul	s3, t2, t0		# 58: s3 = -1 * 10 = -10

	# === 59-62: Division and Remainder ===
	div	s4, t1, t0		# 59: s4 = 20 / 10 = 2
	divu	s5, t1, t0		# 60: s5 = 20 /u 10 = 2
	rem	s6, t1, t0		# 61: s6 = 20 % 10 = 0
	remu	s7, t1, t0		# 62: s7 = 20 %u 10 = 0

	# === 63-64: Finalize ===
	addi	zero, zero, 0		# 63: NOP (writes to zero register)
	mv	a1, a0
	li	a0, 0
	li	a7, 93			# 64: Setup ecall halt
	ecall				# 65: Halt
