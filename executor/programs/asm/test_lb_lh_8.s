	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# Test LB, LH, LBU, LHU (byte/half loads) - 8 instructions
	addi	t0, zero, -1		# t0 = 0xFFFFFFFFFFFFFFFF
	addi	sp, sp, -16		# Allocate stack space
	sd	t0, 0(sp)		# Store -1 (all bits set)
	lb	a0, 0(sp)		# Signed byte load: a0 = -1 (sign extended)
	lbu	a1, 0(sp)		# Unsigned byte load: a1 = 255
	lh	a2, 0(sp)		# Signed half load: a2 = -1 (sign extended)
	lhu	a3, 0(sp)		# Unsigned half load: a3 = 65535
	jalr	zero, 0(zero)
