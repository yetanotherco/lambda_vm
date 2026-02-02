	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# BGEU: -1 is 0xFFFFFFFFFFFFFFFF which is very large unsigned
	# So -1 >= 10 is true (unsigned)
	addi	a2, zero, -1       # 0xFFFFFFFFFFFFFFFF
	addi	a3, zero, 10
	addi	a0, zero, 1        # Default result
	bgeu	a2, a3, greater_eq
	addi	a0, zero, 2        # Not taken path
	j	end
greater_eq:
	addi	a0, zero, 3        # Taken path
end:
	li	a7, 5
	ecall
