	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# BLTU: Branch if less than (unsigned) - taken case
	addi	a2, zero, 5        # 5 < 10 (unsigned)
	addi	a3, zero, 10
	addi	a0, zero, 1        # Default result
	bltu	a2, a3, less
	addi	a0, zero, 2        # Not taken path
	j	end
less:
	addi	a0, zero, 3        # Taken path
end:
	li	a0, 0
	li	a7, 93
	ecall
