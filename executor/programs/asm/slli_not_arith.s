	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	# Test that slli is NOT arithmetic (top bit shifts out)
	# Construct 0x8000000000000001 (i64::MIN + 1)
	li	a2, -1            # 0xFFFFFFFFFFFFFFFF
	srli	a2, a2, 1         # 0x7FFFFFFFFFFFFFFF = i64::MAX
	addi	a2, a2, 2         # 0x8000000000000001 = i64::MIN + 1
	slli    a0, a2, 1         # 0x0000000000000002 = 2 (top bit shifted out)
	li	a7, 5
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
