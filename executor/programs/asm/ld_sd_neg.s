	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# LD/SD: 64-bit store/load of negative value
	# -1 = 0xFFFFFFFFFFFFFFFF
	addi	a2, zero, -1
	lui	a3, 0x80000         # Base address
	sd	a2, 0(a3)           # Store doubleword
	ld	a0, 0(a3)           # Load doubleword
	jalr	zero, 0(ra)
