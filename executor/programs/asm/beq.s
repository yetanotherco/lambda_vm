	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# BEQ: Branch if equal - taken case
	addi	a2, zero, 10
	addi	a3, zero, 10
	addi	a0, zero, 1        # Default result
	beq	a2, a3, equal
	addi	a0, zero, 2        # Not taken path
	j	end
equal:
	addi	a0, zero, 3        # Taken path
end:
	mv	a1, a0
	li	a0, 0
	li	a7, 5
	ecall
