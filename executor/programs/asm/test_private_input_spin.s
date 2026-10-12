	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Spins a loop whose count is the private input's first word, then
	# commits 8 bytes of the input: a guest whose cycle count, and so its epoch
	# count, a test chooses (13 cycles with no spins, 2 more per spin).
	#
	# Layout: [len:u32 LE] at 0xFF000000, then the input bytes. The spin count is
	# the input's first u32 (0xFF000004); the committed bytes are the next eight
	# (0xFF000008). The commit reads the private-input page in the last epoch, as
	# the reads at the start did in the first, so the page crosses every epoch
	# boundary the run has.

	li	t0, 0xFF000000		# t0 = 0xFF000000 (private input base)
	lw	t3, 0(t0)		# the length (touches the private-input page)
	lw	t4, 4(t0)		# the spin count
	beqz	t4, done
spin:
	addi	t4, t4, -1
	bnez	t4, spin
done:
	# Commit 8 bytes from 0xFF000008
	addi	a1, t0, 8		# buf_addr = 0xFF000008
	li	a0, 1			# fd = 1
	li	a2, 8			# count = 8
	li	a7, 64			# syscall = Commit
	ecall

	# Halt
	li	a0, 0			# exit_code = 0
	li	a7, 93			# syscall = Halt
	ecall
