	.attribute	5, "rv64i2p1_m2p0_zmmul1p0"
.Lfunc_end0:
	.globl	main
main:
	# Keccak-256 of a buffer (ecall -5): x10 = input, x11 = length, x12 = digest.
	#
	# A 1024-byte input buffer at sp + 512 holds byte[i] = (13*i + 5) mod 256,
	# written a byte at a time. Eleven digests go to the table at sp (32 bytes
	# each) and are committed at the end; the host test recomputes each one.
	#
	#   call | input offset | length | case
	#   -----+--------------+--------+---------------------------------------
	#     0  |      0       |    0   | empty input: padding alone
	#     1  |      1       |    1   | one byte, odd address
	#     2  |      3       |   135  | len % 136 == 135: 0x01 and 0x80 meet
	#     3  |      0       |   136  | len % 136 == 0: a block of padding alone
	#     4  |      5       |   137  | two blocks, one tail byte
	#     5  |      2       |   271  | two blocks, 135-byte tail
	#     6  |      0       |   272  | three blocks, empty tail
	#     7  |      7       |   600  | five blocks, unaligned throughout
	#     8  |      8       |    20  | an address-sized input, aligned
	#     9  |      6       |    32  | a slot-sized input
	#    10  |      4       |   140  | digest written over its own input (sp+512+4+10)
	addi	sp, sp, -1536

	# Fill the input buffer.
	addi	t0, sp, 512
	li	t1, 0
	li	t2, 1024
	li	t3, 5
.Lfill:
	sb	t3, 0(t0)
	addi	t3, t3, 13
	andi	t3, t3, 255
	addi	t0, t0, 1
	addi	t1, t1, 1
	bne	t1, t2, .Lfill

	addi	s0, sp, 512	# input base
	mv	s1, sp		# digest table
	li	a7, -5

	addi	a0, s0, 0
	li	a1, 0
	addi	a2, s1, 0
	ecall

	addi	a0, s0, 1
	li	a1, 1
	addi	a2, s1, 32
	ecall

	addi	a0, s0, 3
	li	a1, 135
	addi	a2, s1, 64
	ecall

	addi	a0, s0, 0
	li	a1, 136
	addi	a2, s1, 96
	ecall

	addi	a0, s0, 5
	li	a1, 137
	addi	a2, s1, 128
	ecall

	addi	a0, s0, 2
	li	a1, 271
	addi	a2, s1, 160
	ecall

	addi	a0, s0, 0
	li	a1, 272
	addi	a2, s1, 192
	ecall

	addi	a0, s0, 7
	li	a1, 600
	addi	a2, s1, 224
	ecall

	addi	a0, s0, 8
	li	a1, 20
	addi	a2, s1, 256
	ecall

	addi	a0, s0, 6
	li	a1, 32
	addi	a2, s1, 288
	ecall

	# The digest overwrites bytes 10..42 of its own 140-byte input.
	addi	a0, s0, 4
	li	a1, 140
	addi	a2, s0, 14
	ecall

	# Copy that digest into the table (byte by byte), then commit the table.
	addi	t0, s0, 14
	addi	t1, s1, 320
	li	t2, 32
.Lcopy:
	lbu	t3, 0(t0)
	sb	t3, 0(t1)
	addi	t0, t0, 1
	addi	t1, t1, 1
	addi	t2, t2, -1
	bnez	t2, .Lcopy

	li	a0, 1
	mv	a1, s1
	li	a2, 352
	li	a7, 64
	ecall

	addi	sp, sp, 1536
	li	a0, 0
	li	a7, 93
	ecall
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
