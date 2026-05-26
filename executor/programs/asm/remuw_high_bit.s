	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# REMUW where the 32-bit unsigned remainder has bit 31 set.
	# 0x80000001 < 0xFFFFFFFE, so remainder = 0x80000001 (bit 31 set).
	# Zero-extended to u64 = 0x00000000_80000001. The CPU must send this
	# over the DVRM bus to match the chip's raw 64-bit unsigned rem output;
	# sending the sign-extended value (0xFFFFFFFF_80000001) would unbalance.
	li	a2, 0x80000001
	li	a3, 0xFFFFFFFE
	remuw	a0, a2, a3
	li	a0, 0
	li	a7, 93
	ecall
