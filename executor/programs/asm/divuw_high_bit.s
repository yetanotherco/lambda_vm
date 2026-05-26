	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# DIVUW where the 32-bit unsigned quotient has bit 31 set.
	# 0x80000000 / 1 = 0x80000000. Zero-extended to u64 = 0x00000000_80000000.
	# Sign-extending the same 32 bits would give 0xFFFFFFFF_80000000.
	# CPU must send the zero-extended value over the DVRM bus to match the
	# DVRM chip's raw 64-bit unsigned division output.
	li	a2, 0x80000000
	addi	a3, zero, 1
	divuw	a0, a2, a3
	li	a0, 0
	li	a7, 93
	ecall
