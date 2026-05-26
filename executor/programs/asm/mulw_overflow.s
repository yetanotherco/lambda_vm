	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# MULW where the 32-bit product overflows past bit 31.
	# 0x10000 * 0x10000 = 0x1_00000000. Low 32 bits = 0.
	# Raw i64 product of the (i32-sign-extended) operands = 0x1_00000000.
	# CPU must send the i32-wrapped value (0) over the MUL bus, not the raw
	# 64-bit product, otherwise the bus interaction does not balance.
	li	a2, 0x10000
	li	a3, 0x10000
	mulw	a0, a2, a3
	li	a0, 0
	li	a7, 93
	ecall
