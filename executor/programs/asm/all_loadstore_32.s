	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Test ALL load/store variants (32 instructions)
	# Covers: LB, LH, LW, LD, LBU, LHU, LWU, SB, SH, SW, SD

	# === 1-4: Setup ===
	addi	t0, zero, -1		# 1: t0 = -1 (0xFFFFFFFFFFFFFFFF)
	addi	t1, zero, 0x42		# 2: t1 = 0x42 (byte value)
	addi	t2, zero, 0x789		# 3: t2 = 0x789 (half value)
	addi	sp, sp, -64		# 4: Allocate stack space

	# === 5-11: Store operations ===
	sd	t0, 0(sp)		# 5: Store doubleword (-1)
	sw	t0, 8(sp)		# 6: Store word (0xFFFFFFFF)
	sh	t2, 12(sp)		# 7: Store half (0x789)
	sb	t1, 14(sp)		# 8: Store byte (0x42)
	sd	zero, 16(sp)		# 9: Store 0
	sw	t1, 24(sp)		# 10: Store word (0x42)
	sh	t1, 28(sp)		# 11: Store half (0x42)

	# === 12-18: Signed loads ===
	ld	a0, 0(sp)		# 12: Load doubleword: a0 = -1
	lw	a1, 8(sp)		# 13: Load word (sign ext): a1 = -1
	lh	a2, 12(sp)		# 14: Load half (sign ext): a2 = 0x789
	lb	a3, 14(sp)		# 15: Load byte (sign ext): a3 = 0x42
	lw	a4, 0(sp)		# 16: Load word from -1: a4 = -1 (sign ext)
	lh	a5, 0(sp)		# 17: Load half from -1: a5 = -1 (sign ext)
	lb	a6, 0(sp)		# 18: Load byte from -1: a6 = -1 (sign ext)

	# === 19-24: Unsigned loads ===
	lwu	a7, 8(sp)		# 19: Load word unsigned: a7 = 0xFFFFFFFF
	lhu	s0, 12(sp)		# 20: Load half unsigned: s0 = 0x789
	lbu	s1, 14(sp)		# 21: Load byte unsigned: s1 = 0x42
	lwu	s2, 0(sp)		# 22: Load word unsigned from -1: s2 = 0xFFFFFFFF
	lhu	s3, 0(sp)		# 23: Load half unsigned from -1: s3 = 0xFFFF
	lbu	s4, 0(sp)		# 24: Load byte unsigned from -1: s4 = 0xFF

	# === 25-30: More stores and loads with offsets ===
	addi	t3, zero, 123		# 25: t3 = 123
	sd	t3, 32(sp)		# 26: Store at offset 32
	ld	s5, 32(sp)		# 27: Load back: s5 = 123
	sw	t3, 40(sp)		# 28: Store word at offset 40
	lwu	s6, 40(sp)		# 29: Load unsigned: s6 = 123
	lw	s7, 40(sp)		# 30: Load signed: s7 = 123

	# === 31-32: Finalize ===
	addi	sp, sp, 64		# 31: Deallocate stack
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall		# 32: Halt
