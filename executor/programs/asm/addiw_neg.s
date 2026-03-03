	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
	.globl	main
main:
	# ADDIW: add negative immediate
	# 100 + (-50) = 50
	addi	a2, zero, 100
	addiw	a0, a2, -50
	li	a7, 5
	ecall
