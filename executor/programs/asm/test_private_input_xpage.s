	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Read private input directly from 0xFF000000 (memory-mapped).
	# Layout: [len:u32 LE] [12 reserved bytes] [data at +16]
	# Commits 8 bytes of data.
	#
	# Note: lui in RV64 sign-extends to 64 bits. lui with 0xFF000 would give
	# 0xFFFFFFFFFF000000. To get 0xFF000000 we need to construct it differently:
	# lui x, 0x100000 gives 0x100000000 (53 upper bits), too high.
	# Instead: load 0x0FF00000 and shift left by 4 bits, OR similar tricks.
	# Simplest: use li macro and let the assembler handle it.

	li	t0, 0xFF000000		# 1: t0 = 0xFF000000 (private input base)

	# Read length at 0xFF000000
	lw	t3, 0(t0)		# 2: t3 = length

	# Load 8 bytes of data at 0xFF000010 (aligned, start of data region)
	ld	t1, 16(t0)		# 3

	# Commit 8 bytes from 0xFF000010
	addi	a1, t0, 16		# 4: buf_addr = 0xFF000010
	li	a0, 1			# 5: fd = 1
	li	a2, 8			# 6: count = 8
	li	a7, 64			# 7: syscall = Commit
	ecall				# 8: commit

	# Halt
	li	a0, 0			# 9: exit_code = 0
	li	a7, 93			# 10: syscall = Halt
	ecall				# 11: halt
