	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# ECSM lincomb2 round trip: Q = 3·G + 5·(2G) = 13·G.
	#
	# Stack layout (384 bytes). Every operand is 64 bytes — two 32-byte
	# little-endian values back to back — 8-byte aligned and pairwise disjoint,
	# as the syscall's address guards require:
	#   sp+0   P1 = xG‖yG           sp+64  P2 = x(2G)‖y(2G)
	#   sp+128 u1‖u2                sp+200 Q  = xQ‖yQ  (result)
	#   sp+192 status word (8 bytes, stored from a0 after the call)
	#
	# The frame is deliberately larger than the 264 bytes it uses: an operand's
	# last byte (+63) must not cross a 4 GiB boundary, and STACK_TOP
	# (0xFFFF_FFFF_FFFF_FFF0) sits right below one. 384 bytes keeps every
	# operand clear of it.
	addi	sp, sp, -384

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

	# u1 = 3 at sp+128, u2 = 5 at sp+160
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

	# a0 now holds the status word; stash it just below the result so the host
	# test can commit status‖xQ‖yQ as one contiguous 72-byte block.
	sd	a0, 192(sp)

	# Commit syscall: a0 = fd(1), a1 = buf_addr, a2 = count, a7 = 64.
	li	a0, 1
	addi	a1, sp, 192
	li	a2, 72
	li	a7, 64
	ecall

	# Restore stack and halt.
	addi	sp, sp, 384
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
