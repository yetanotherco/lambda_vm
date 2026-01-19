	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# BLT: Branch if less than (signed) - not taken case
	addi	a2, zero, 10       # 10 is not < 10
	addi	a3, zero, 10
	addi	a0, zero, 1        # Default result
	blt	a2, a3, less
	addi	a0, zero, 2        # Not taken path (should execute)
	j	end
less:
	addi	a0, zero, 3        # Taken path
end:
	jalr	zero, 0(ra)
