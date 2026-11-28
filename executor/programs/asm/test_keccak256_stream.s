	.attribute 5, "rv64i2p1"
	.section .rodata
input:
	.byte	0x00
	.fill	137, 1, 0xa5

	.text
	.globl	main
main:
	# One unaligned input crosses the 136-byte rate boundary. The single
	# syscall thus builds two permutations' KECCAK_RND rows and a padded tail.
	addi	sp, sp, -32
	la	a0, input
	addi	a0, a0, 1
	li	a1, 137
	mv	a2, sp
	li	a7, -5
	ecall

	# Commit the digest so the statement depends on the syscall's writes.
	li	a0, 1
	mv	a1, sp
	li	a2, 32
	li	a7, 64
	ecall

	addi	sp, sp, 32
	li	a0, 0
	li	a7, 93
	ecall
