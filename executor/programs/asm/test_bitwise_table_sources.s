	.attribute 5, "rv64i2p1"
	.globl	main
main:
	# Exercise all three producers whose lookup counts BITWISE can take
	# from their tables. Use nontrivial byte values in the comparisons.
	li	t0, 0x8001
	li	t1, 0x0204
	slt	t2, t1, t0
	xor	t2, t2, t0
	bltu	t1, t0, done
	addi	t2, t2, 1

done:
	li	a0, 0
	li	a7, 93
	ecall
