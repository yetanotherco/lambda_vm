	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# Load a = (1,2,3), b = (4,5,6), c = (7,8,9) into field-storage 1/2/3 once.
	li	a0, 1
	li	a1, 1
	li	a2, 2
	li	a3, 3
	li	a7, -20
	ecall
	li	a0, 2
	li	a1, 4
	li	a2, 5
	li	a3, 6
	li	a7, -20
	ecall
	li	a0, 3
	li	a1, 7
	li	a2, 8
	li	a3, 9
	li	a7, -20
	ecall

	# FEXT_FMA args (a=1, b=2, c=3, out=4) set once; a7 = -21.
	li	a0, 1
	li	a1, 2
	li	a2, 3
	li	a3, 4
	li	a7, -21

	# Loop: N = 4096 FEXT_FMA calls. Each writes out(4) = a*b + c at a fresh
	# timestamp (distinct addresses satisfy the accelerator's per-op guard).
	li	t0, 4096
.Lloop:
	ecall
	addi	t0, t0, -1
	bnez	t0, .Lloop

	# Halt.
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
