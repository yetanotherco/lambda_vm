	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# The COPY calls whose row split and overlap decide which bytes end at the
	# call's timestamp and which one after it:
	#   memcpy of 16 bytes onto its own source + 5    2 wide rows, forward overlap
	#   memcpy of 13 bytes onto its own source - 3    1 wide + 5 one-byte rows,
	#                                                 backward overlap (a memmove)
	#   memset of 8 + 13 bytes                        1 wide + 5 one-byte rows
	#   memset of 8 + 0 bytes                         the zero-length row
	#   memcpy of 7 bytes                             7 one-byte rows
	# memcpy is ecall -30, memset -31; x10 = dst, x11 = src, x12 = n.
	addi	sp, sp, -128

	# A source pattern at sp+0..20.
	addi	t0, zero, 0x11
	sb	t0, 0(sp)
	addi	t0, zero, 0x22
	sb	t0, 3(sp)
	addi	t0, zero, 0x33
	sb	t0, 7(sp)
	addi	t0, zero, 0x44
	sb	t0, 12(sp)
	addi	t0, zero, 0x55
	sb	t0, 15(sp)

	# memcpy(sp+5, sp, 16): the destination overlaps the source's tail
	addi	a0, sp, 5
	mv	a1, sp
	addi	a2, zero, 16
	addi	a7, zero, -30
	ecall

	# memcpy(sp+1, sp+4, 13): the destination overlaps the source's head
	addi	a0, sp, 1
	addi	a1, sp, 4
	addi	a2, zero, 13
	addi	a7, zero, -30
	ecall

	# memset(sp+64, 0xA5, 21): the seed, then the copy from seed to seed + 8
	addi	t0, zero, 0xA5
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
	addi	a2, zero, 13
	addi	a7, zero, -31
	ecall

	# memset(sp+96, 0xA5, 8): the seed alone, a zero-length copy after it
	addi	a0, sp, 104
	addi	a1, sp, 96
	addi	a2, zero, 0
	addi	a7, zero, -31
	ecall

	# memcpy(sp+110, sp+1, 7)
	addi	a0, sp, 110
	addi	a1, sp, 1
	addi	a2, zero, 7
	addi	a7, zero, -30
	ecall

	# Halt
	addi	sp, sp, 128
	addi	a0, zero, 0
	addi	a7, zero, 93
	ecall
