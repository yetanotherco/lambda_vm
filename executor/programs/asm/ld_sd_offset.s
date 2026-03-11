	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# LD/SD with offset
	li	a2, 0xDEADBEEFCAFEBABE
	lui	a3, 0x80000         # Base address
	sd	a2, 16(a3)          # Store at offset 16
	ld	a0, 16(a3)          # Load from offset 16
	li	a0, 0
	li	a7, 93
	ecall
