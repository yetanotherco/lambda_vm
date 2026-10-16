	.data
	.align 3
secret:
	.dword 0x8877665544332211

	.text
	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Load 8 bytes out of the ELF's own .data section, spill them to the
	# stack, and commit them. The committed public output is therefore a
	# direct function of the ELF image bytes at `secret`, which the verifier
	# binds through the PAGE preprocessed commitment of that data page.
	la	t0, secret
	ld	t1, 0(t0)		# t1 = *secret
	addi	sp, sp, -16
	sd	t1, 0(sp)		# spill to stack
	li	a0, 1			# fd = 1
	mv	a1, sp			# buf = sp
	li	a2, 8			# count = 8
	li	a7, 64			# syscall = Commit
	ecall

	addi	sp, sp, 16
	li	a0, 0
	li	a7, 93			# syscall = Halt
	ecall
