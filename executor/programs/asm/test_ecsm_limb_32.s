	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	addi	sp, sp, -64
	li	t0, 0x59F2815B16F81798
	sd	t0, 0(sp)
	li	t0, 0x029BFCDB2DCE28D9
	sd	t0, 8(sp)
	li	t0, 0x55A06295CE870B07
	sd	t0, 16(sp)
	li	t0, 0x79BE667EF9DCBBAC
	sd	t0, 24(sp)
	li	t0, 5
	sd	t0, 32(sp)
	sd	zero, 40(sp)
	sd	zero, 48(sp)
	sd	zero, 56(sp)
	li	t1, 1
	slli	t1, t1, 32
	addi	a0, t1, -32
	addi	a1, sp, 0
	addi	a2, sp, 32
	li	a7, -11
	ecall
	addi	sp, sp, 64
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
