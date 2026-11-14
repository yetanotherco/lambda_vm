	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Three ECREC calls (ECALL -12): two valid signatures and one with r = 0, which bails (status 1).
	# Per call c (base 176·c): sig r ‖ s at +0 (64 bytes, big-endian), hash at +64 (32, big-endian),
	# output at +96 (72: the key big-endian, then the status doubleword). The signatures are fixed test vectors.
	addi	sp, sp, -528

	# --- call 1 ---
	li	t0, 0xDF93B6A1E4BA8432
	sd	t0, 0(sp)
	li	t0, 0x1F89E40074C68ABE
	sd	t0, 8(sp)
	li	t0, 0x15A359511C68533B
	sd	t0, 16(sp)
	li	t0, 0x3971C5D2F3511056
	sd	t0, 24(sp)
	li	t0, 0x4EA528F7F2B2D497
	sd	t0, 32(sp)
	li	t0, 0x1DB1F63722A162D5
	sd	t0, 40(sp)
	li	t0, 0x344CC2FCE726D632
	sd	t0, 48(sp)
	li	t0, 0x57D05B099E72B070
	sd	t0, 56(sp)
	li	t0, 0x4433221100EEFFC0
	sd	t0, 64(sp)
	li	t0, 0xBBAA009988776655
	sd	t0, 72(sp)
	li	t0, 0x33221100FFEEDDCC
	sd	t0, 80(sp)
	li	t0, 0xBBAA998877665544
	sd	t0, 88(sp)
	addi	a0, sp, 96
	addi	a1, sp, 0
	addi	a2, sp, 64
	li	a3, 0
	li	a7, -12
	ecall

	# --- call 2 ---
	li	t0, 0xF8BCE8385BD46279
	sd	t0, 176(sp)
	li	t0, 0xF2012A43A8EFA82F
	sd	t0, 184(sp)
	li	t0, 0x113F7D4CE2539A0C
	sd	t0, 192(sp)
	li	t0, 0xDA2609E7B87C19DF
	sd	t0, 200(sp)
	li	t0, 0x79042F811168ECEC
	sd	t0, 208(sp)
	li	t0, 0x958F9472BE921F4C
	sd	t0, 216(sp)
	li	t0, 0x9DA3454487A7CDA6
	sd	t0, 224(sp)
	li	t0, 0xCDF235C831AB0AB4
	sd	t0, 232(sp)
	li	t0, 0x0000000000000000
	sd	t0, 240(sp)
	li	t0, 0x0000000000000000
	sd	t0, 248(sp)
	li	t0, 0x0000000000000000
	sd	t0, 256(sp)
	li	t0, 0x4200000000000000
	sd	t0, 264(sp)
	addi	a0, sp, 272
	addi	a1, sp, 176
	addi	a2, sp, 240
	li	a3, 1
	li	a7, -12
	ecall

	# --- call 3 ---
	li	t0, 0x0000000000000000
	sd	t0, 352(sp)
	li	t0, 0x0000000000000000
	sd	t0, 360(sp)
	li	t0, 0x0000000000000000
	sd	t0, 368(sp)
	li	t0, 0x0000000000000000
	sd	t0, 376(sp)
	li	t0, 0x4EA528F7F2B2D497
	sd	t0, 384(sp)
	li	t0, 0x1DB1F63722A162D5
	sd	t0, 392(sp)
	li	t0, 0x344CC2FCE726D632
	sd	t0, 400(sp)
	li	t0, 0x57D05B099E72B070
	sd	t0, 408(sp)
	li	t0, 0x4433221100EEFFC0
	sd	t0, 416(sp)
	li	t0, 0xBBAA009988776655
	sd	t0, 424(sp)
	li	t0, 0x33221100FFEEDDCC
	sd	t0, 432(sp)
	li	t0, 0xBBAA998877665544
	sd	t0, 440(sp)
	addi	a0, sp, 448
	addi	a1, sp, 352
	addi	a2, sp, 416
	li	a3, 0
	li	a7, -12
	ecall

	# Commit output 1 (72 bytes).
	li	a0, 1
	addi	a1, sp, 96
	li	a2, 72
	li	a7, 64
	ecall

	# Commit output 2 (72 bytes).
	li	a0, 1
	addi	a1, sp, 272
	li	a2, 72
	li	a7, 64
	ecall

	# Commit output 3 (72 bytes).
	li	a0, 1
	addi	a1, sp, 448
	li	a2, 72
	li	a7, 64
	ecall

	addi	sp, sp, 528
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
