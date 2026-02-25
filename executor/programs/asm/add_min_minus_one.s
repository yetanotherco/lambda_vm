	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	# Get i64::MIN (0x8000000000000000) then subtract 1 to overflow to i64::MAX
	li	a0, -1            # 0xFFFFFFFFFFFFFFFF
	srli	a0, a0, 1         # 0x7FFFFFFFFFFFFFFF = i64::MAX
	addi	a0, a0, 1         # 0x8000000000000000 = i64::MIN
	addi	a0, a0, -1        # overflow to i64::MAX
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
