	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	# Get i64::MAX (0x7FFFFFFFFFFFFFFF) then add 1 to overflow to i64::MIN
	li	a0, -1            # 0xFFFFFFFFFFFFFFFF
	srli	a0, a0, 1         # 0x7FFFFFFFFFFFFFFF = i64::MAX
	addi	a0, a0, 1         # overflow to i64::MIN
	jalr	zero, 0(ra)
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
