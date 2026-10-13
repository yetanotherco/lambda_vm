	.text
	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Exit with a NON-ZERO code. HALT enforces x10 == 0, so an honest prover
	# cannot prove this program.
	li	a0, 1
	li	a7, 93			# syscall = Halt
	ecall

	# Never reached by an honest execution. The HALT PoC runs these rows AFTER
	# the exit ecall to zero x10 before the trace ends.
	li	a0, 0
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop
	nop

	# Lets the PoC's patched copy (first ecall -> nop) terminate in the executor.
	li	a7, 93
	ecall
