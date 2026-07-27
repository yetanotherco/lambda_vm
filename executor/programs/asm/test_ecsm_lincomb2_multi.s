	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# TWO lincomb2 ecalls from one program, reusing the same P1/P2/u buffers.
	#
	# The chip performs every MEMW access of a call at a SINGLE timestamp (the
	# four operand regions are pairwise disjoint and registers are a separate
	# address space), so repetition against the same addresses is exactly what
	# would break the access chain if that scheme were wrong. It is also the
	# real shape: a block does many ecrecovers.
	#
	# Stack layout (512 bytes):
	#   sp+0   P1 = xG‖yG        sp+64  P2 = x(2G)‖y(2G)     sp+128 u1‖u2
	#   sp+192 status1           sp+200 Q1 = xQ‖yQ
	#   sp+312 status2           sp+320 Q2 = xQ‖yQ
	# Every operand address is 8-byte aligned and pairwise ≥ 64 bytes from the
	# others, as the syscall's guards require.
	addi	sp, sp, -512

	# P1 = G  (xP1 at sp+0, yP1 at sp+32)
	li	t0, 0x59F2815B16F81798
	sd	t0, 0(sp)
	li	t0, 0x029BFCDB2DCE28D9
	sd	t0, 8(sp)
	li	t0, 0x55A06295CE870B07
	sd	t0, 16(sp)
	li	t0, 0x79BE667EF9DCBBAC
	sd	t0, 24(sp)
	li	t0, 0x9C47D08FFB10D4B8
	sd	t0, 32(sp)
	li	t0, 0xFD17B448A6855419
	sd	t0, 40(sp)
	li	t0, 0x5DA4FBFC0E1108A8
	sd	t0, 48(sp)
	li	t0, 0x483ADA7726A3C465
	sd	t0, 56(sp)

	# P2 = 2G (xP2 at sp+64, yP2 at sp+96)
	li	t0, 0xABAC09B95C709EE5
	sd	t0, 64(sp)
	li	t0, 0x5C778E4B8CEF3CA7
	sd	t0, 72(sp)
	li	t0, 0x3045406E95C07CD8
	sd	t0, 80(sp)
	li	t0, 0xC6047F9441ED7D6D
	sd	t0, 88(sp)
	li	t0, 0x236431A950CFE52A
	sd	t0, 96(sp)
	li	t0, 0xF7F632653266D0E1
	sd	t0, 104(sp)
	li	t0, 0xA3C58419466CEAEE
	sd	t0, 112(sp)
	li	t0, 0x1AE168FEA63DC339
	sd	t0, 120(sp)

	# u1 = 3, u2 = 5
	li	t0, 0x0000000000000003
	sd	t0, 128(sp)
	li	t0, 0x0000000000000000
	sd	t0, 136(sp)
	li	t0, 0x0000000000000000
	sd	t0, 144(sp)
	li	t0, 0x0000000000000000
	sd	t0, 152(sp)
	li	t0, 0x0000000000000005
	sd	t0, 160(sp)
	li	t0, 0x0000000000000000
	sd	t0, 168(sp)
	li	t0, 0x0000000000000000
	sd	t0, 176(sp)
	li	t0, 0x0000000000000000
	sd	t0, 184(sp)

	# lincomb2 ecall: a0 = &Q, a1 = &P1, a2 = &P2, a3 = &u, a7 = -12.
	addi	a0, sp, 200
	addi	a1, sp, 0
	addi	a2, sp, 64
	addi	a3, sp, 128
	li	a7, -12
	ecall
	# a0 holds the status; stash it just below its result block.
	sd	a0, 192(sp)

	# u1 = 7, u2 = 11
	li	t0, 0x0000000000000007
	sd	t0, 128(sp)
	li	t0, 0x0000000000000000
	sd	t0, 136(sp)
	li	t0, 0x0000000000000000
	sd	t0, 144(sp)
	li	t0, 0x0000000000000000
	sd	t0, 152(sp)
	li	t0, 0x000000000000000B
	sd	t0, 160(sp)
	li	t0, 0x0000000000000000
	sd	t0, 168(sp)
	li	t0, 0x0000000000000000
	sd	t0, 176(sp)
	li	t0, 0x0000000000000000
	sd	t0, 184(sp)

	# lincomb2 ecall: a0 = &Q, a1 = &P1, a2 = &P2, a3 = &u, a7 = -12.
	addi	a0, sp, 320
	addi	a1, sp, 0
	addi	a2, sp, 64
	addi	a3, sp, 128
	li	a7, -12
	ecall
	# a0 holds the status; stash it just below its result block.
	sd	a0, 312(sp)

	# Commit syscall: a0 = fd(1), a1 = buf_addr, a2 = count, a7 = 64.
	li	a0, 1
	addi	a1, sp, 192
	li	a2, 72
	li	a7, 64
	ecall

	# Commit syscall: a0 = fd(1), a1 = buf_addr, a2 = count, a7 = 64.
	li	a0, 1
	addi	a1, sp, 312
	li	a2, 72
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 512
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
