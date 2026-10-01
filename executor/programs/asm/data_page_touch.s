	.data
	.align 3
counter:
	.dword 0x123456789ABCDEF0

	.text
	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Touch an ELF .data page: load, mutate, store back a static global so the
	# page is genuinely ELF-backed (init_values non-empty), not stack/zero-init.
	la	t0, counter		# 1: t0 = &counter
	ld	t1, 0(t0)		# 2: t1 = counter (0x123456789ABCDEF0)
	addi	t1, t1, 1		# 3: t1 += 1
	sd	t1, 0(t0)		# 4: counter = t1

	li	a0, 0
	li	a7, 93
	ecall		# 5: Halt
