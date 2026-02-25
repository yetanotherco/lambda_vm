	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# REMW overflow: i32::MIN % -1 = 0
	# The division would overflow, but remainder is 0
	li	a2, -2147483648      # i32::MIN
	addi	a3, zero, -1
	remw	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
