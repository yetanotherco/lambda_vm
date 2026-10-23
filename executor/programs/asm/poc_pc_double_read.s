	.text
	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Exits with code 5. HALT enforces x10 == 0, so an honest prover cannot
	# prove this program.
	li	a0, 5
	# The PoC proves this row with pc_double_read = 1 one timestamp after the
	# previous row, so its read of a0 lands before the write above and returns
	# the stale initial a0 = 0.
	mv	a1, a0
	mv	a0, a1
	li	a7, 93			# syscall = Halt
	ecall
