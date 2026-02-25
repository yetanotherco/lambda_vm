	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# DIVW overflow: i32::MIN / -1
	# In 2's complement, -2147483648 / -1 would be 2147483648 which overflows i32
	# RISC-V returns i32::MIN (-2147483648) sign-extended
	li	a2, -2147483648      # i32::MIN = 0x80000000
	addi	a3, zero, -1
	divw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
