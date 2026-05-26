	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Misaligned SH: 2-byte store at address 67.
	li	t0, 64
	li	t1, 0xAA12
	sh	t1, 3(t0)
	li	a0, 0
	li	a7, 93
	ecall
