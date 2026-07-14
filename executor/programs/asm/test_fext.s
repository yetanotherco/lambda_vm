	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# FEXT_LOAD a = (1, 2, 3) into field-storage address 1.
	# a0 = field-storage address, a1/a2/a3 = coefficients, a7 = -20.
	li	a0, 1
	li	a1, 1
	li	a2, 2
	li	a3, 3
	li	a7, -20
	ecall

	# FEXT_LOAD b = (4, 5, 6) into field-storage address 2.
	li	a0, 2
	li	a1, 4
	li	a2, 5
	li	a3, 6
	li	a7, -20
	ecall

	# FEXT_LOAD c = (7, 8, 9) into field-storage address 3.
	li	a0, 3
	li	a1, 7
	li	a2, 8
	li	a3, 9
	li	a7, -20
	ecall

	# FEXT_FMA: out(addr 4) = a(addr 1) * b(addr 2) + c(addr 3).
	# a0/a1/a2 = a/b/c addresses, a3 = output address, a7 = -21.
	li	a0, 1
	li	a1, 2
	li	a2, 3
	li	a3, 4
	li	a7, -21
	ecall

	# FEXT_STORE: read result (addr 4) back into registers a1/a2/a3.
	# a0 = field-storage source address, a7 = -22.
	li	a0, 4
	li	a7, -22
	ecall

	# Halt.
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
