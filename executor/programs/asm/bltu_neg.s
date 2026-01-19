	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# BLTU: -1 is 0xFFFFFFFFFFFFFFFF which is very large unsigned
	# So -1 is NOT less than 10 unsigned
	addi	a2, zero, -1       # 0xFFFFFFFFFFFFFFFF
	addi	a3, zero, 10
	addi	a0, zero, 1        # Default result
	bltu	a2, a3, less
	addi	a0, zero, 2        # Not taken path (should execute)
	j	end
less:
	addi	a0, zero, 3        # Taken path
end:
	jalr	zero, 0(ra)
