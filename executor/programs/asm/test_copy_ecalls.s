	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Every row shape the COPY table has, 17 rows in all, so a cut at a few rows
	# leaves a chain across a table boundary and the whole run pads to 32:
	#   memcpy of 21 bytes        2 wide rows + 5 one-byte rows
	#   memset of 8 + 11 bytes    1 wide row + 3 one-byte rows (the seed is 8 sb)
	#   memcpy of 0 bytes         the zero-length row
	#   memcpy of 2 bytes         2 one-byte rows, onto its own source + 1
	#   write of 3 bytes          3 one-byte rows into the COMMIT domain
	# memcpy is ecall -30, memset -31, write 64; x10 = dst, x11 = src, x12 = n.
	addi	sp, sp, -128

	# A source pattern at sp+0..24.
	addi	t0, zero, 0x11
	sb	t0, 0(sp)
	addi	t0, zero, 0x22
	sb	t0, 1(sp)
	addi	t0, zero, 0x33
	sb	t0, 2(sp)
	addi	t0, zero, 0x44
	sb	t0, 9(sp)
	addi	t0, zero, 0x55
	sb	t0, 20(sp)

	# memcpy(sp+40, sp, 21)
	addi	a0, sp, 40
	mv	a1, sp
	addi	a2, zero, 21
	addi	a7, zero, -30
	ecall

	# memset(sp+64, 0x5A, 19): the seed, then the copy from seed to seed + 8
	addi	t0, zero, 0x5A
	sb	t0, 64(sp)
	sb	t0, 65(sp)
	sb	t0, 66(sp)
	sb	t0, 67(sp)
	sb	t0, 68(sp)
	sb	t0, 69(sp)
	sb	t0, 70(sp)
	sb	t0, 71(sp)
	addi	a0, sp, 72
	addi	a1, sp, 64
	addi	a2, zero, 11
	addi	a7, zero, -31
	ecall

	# memcpy(sp+100, sp+8, 0)
	addi	a0, sp, 100
	addi	a1, sp, 8
	addi	a2, zero, 0
	addi	a7, zero, -30
	ecall

	# memcpy(sp+2, sp+1, 2): overlapping, the snapshot reads before it writes
	addi	a0, sp, 2
	addi	a1, sp, 1
	addi	a2, zero, 2
	addi	a7, zero, -30
	ecall

	# write(1, sp+40, 3): commits 0x11 0x22 0x33
	addi	a0, zero, 1
	addi	a1, sp, 40
	addi	a2, zero, 3
	addi	a7, zero, 64
	ecall

	# Halt
	addi	sp, sp, 128
	addi	a0, zero, 0
	addi	a7, zero, 93
	ecall
