	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# BGE: Branch if greater or equal (signed) - not taken
	addi	a2, zero, -10      # -10 < 10
	addi	a3, zero, 10
	addi	a0, zero, 1        # Default result
	bge	a2, a3, greater_eq
	addi	a0, zero, 2        # Not taken path (should execute)
	j	end
greater_eq:
	addi	a0, zero, 3        # Taken path
end:
	li	a0, 0
	li	a7, 93
	ecall
