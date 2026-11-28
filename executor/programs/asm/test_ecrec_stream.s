	.attribute 5, "rv64i2p1"
	.section .rodata
	.balign	8
signature:
	# The first valid signature and hash from test_ecrec.s.
	.quad	0xDF93B6A1E4BA8432, 0x1F89E40074C68ABE
	.quad	0x15A359511C68533B, 0x3971C5D2F3511056
	.quad	0x4EA528F7F2B2D497, 0x1DB1F63722A162D5
	.quad	0x344CC2FCE726D632, 0x57D05B099E72B070
hash:
	.quad	0x4433221100EEFFC0, 0xBBAA009988776655
	.quad	0x33221100FFEEDDCC, 0xBBAA998877665544

	.text
	.globl	main
main:
	# One recovery, with an ECDAS witness. Commit the returned key and status.
	addi	sp, sp, -80
	mv	a0, sp
	la	a1, signature
	la	a2, hash
	li	a3, 0
	li	a7, -12
	ecall

	li	a0, 1
	mv	a1, sp
	li	a2, 72
	li	a7, 64
	ecall

	addi	sp, sp, 80
	li	a0, 0
	li	a7, 93
	ecall
