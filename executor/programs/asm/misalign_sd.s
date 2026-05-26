	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Misaligned SD: 8-byte store at address 65.
	li	t0, 64
	li	t1, 0x123456789ABCDEF0
	sd	t1, 1(t0)
	li	a0, 0
	li	a7, 93
	ecall
