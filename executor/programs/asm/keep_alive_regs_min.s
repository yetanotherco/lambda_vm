	.attribute 5, "rv64i2p1"
	.globl	main
main:
	# A little over 2^14 cycles (2^16 timestamps) leaves both registers
	# idle past a CPU slot's reach. The reads and write need keep-alives.
	li	t2, 5
	li	t3, 7
	li	t0, 8200
loop:
	addi	t0, t0, -1
	bnez	t0, loop

	add	t4, t2, t2
	addi	t3, t3, 1
	li	a0, 0
	li	a7, 93
	ecall
