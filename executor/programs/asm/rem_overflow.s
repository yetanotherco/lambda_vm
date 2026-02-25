	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# REM overflow: i64::MIN % -1 = 0
	# The division would overflow, but remainder is 0
	li	a2, 0x8000000000000000  # i64::MIN
	addi	a3, zero, -1
	rem	a0, a2, a3
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall
