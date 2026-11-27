	.attribute 5, "rv64i2p1"
	.globl	main
main:
	# Four shifts and aligned stores fill a streamed chunk each of SHIFT,
	# STORE and MEMW_A. Make more than one window at eight instructions.
	addi	sp, sp, -32
	li	t0, -123
	slli	t1, t0, 1
	sd	t1, 0(sp)
	srli	t1, t0, 2
	sd	t1, 8(sp)
	srai	t1, t0, 3
	sd	t1, 16(sp)
	slli	t1, t0, 4
	sd	t1, 24(sp)

	# Close the second window before HALT so the fourth rows are handed out.
	addi	t0, t0, 1
	addi	t0, t0, 1
	addi	t0, t0, 1
	addi	t0, t0, 1
	addi	sp, sp, 32
	li	a0, 0
	li	a7, 93
	ecall
