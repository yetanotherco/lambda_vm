	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# DIV overflow case: i64::MIN / -1
	# This would overflow to i64::MAX+1, RISC-V returns i64::MIN
	li	a2, 0x8000000000000000  # i64::MIN
	addi	a3, zero, -1
	div	a0, a2, a3
	li	a7, 5
	ecall
