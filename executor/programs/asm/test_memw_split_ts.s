	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Exercise both MEMW (split old_timestamp) and MEMW_A (aligned fast path).
	#
	# MEMW path: sb+sb then lh — the two bytes at sp+0 and sp+1 are written
	# by separate instructions (different timestamps), so the lh read has
	# mismatched old_timestamps and routes to MEMW.
	#
	# MEMW_A path: sw then lw — all 4 bytes are written by one instruction
	# (same timestamp), so the lw read routes to MEMW_A.
	addi	sp, sp, -16		# allocate stack
	addi	t0, zero, 0x41		# t0 = 'A'
	addi	t1, zero, 0x42		# t1 = 'B'
	sb	t0, 0(sp)		# write byte 0 at sp+0 (timestamp T3)
	sb	t1, 1(sp)		# write byte 1 at sp+1 (timestamp T4 ≠ T3)
	lh	a0, 0(sp)		# read 2 bytes: old_ts[0]=T3, old_ts[1]=T4 → MEMW
	addi	t2, zero, 0x7FF		# t2 = word value
	sw	t2, 4(sp)		# write 4 bytes at sp+4, all timestamp T7 → MEMW_A
	lw	a1, 4(sp)		# read 4 bytes: all old_ts=T7 → MEMW_A
	addi	sp, sp, 16		# deallocate stack
	li	a0, 0
	li	a7, 93
	ecall
