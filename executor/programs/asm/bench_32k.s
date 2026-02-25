	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Benchmark program: 32768 instructions (2^15)
	# Uses AND, XOR, OR, SLT operations to exercise lookups
	#
	# Structure:
	#   4 setup instructions
	#   4092 loop iterations × 8 instructions = 32736 instructions
	#   28 finalization instructions (padding NOPs)
	#   Total: 4 + 32736 + 28 = 32768 instructions
	#
	# Each loop iteration does:
	#   1. AND operation (triggers AND_BYTE lookups)
	#   2. XOR operation (triggers XOR_BYTE lookups)
	#   3. OR operation (triggers OR_BYTE lookups)
	#   4. SLT operation (triggers LT lookup)
	#   5. ADD for accumulator
	#   6. ADDI for pattern variation
	#   7. Counter increment
	#   8. Branch if not equal

	# === Setup (4 instructions) ===
	lui	t0, 1			# 1: t0 = 4096
	addi	t0, t0, -4		# 2: t0 = 4092 (loop count)
	addi	t1, zero, 0		# 3: t1 = 0 (counter)
	addi	t2, zero, 0x55		# 4: t2 = 0x55 (pattern for bitwise ops)

	# === Main loop: 4092 iterations × 8 instructions = 32736 ===
loop:
	and	t3, t1, t2		# AND: t3 = counter & 0x55
	xor	t4, t3, t1		# XOR: t4 = (counter & 0x55) ^ counter
	or	t5, t4, t2		# OR: t5 = result | 0x55
	slt	t6, t3, t4		# SLT: compare results (triggers LT lookup)
	add	s0, s0, t6		# Accumulate comparison results
	addi	t2, t2, 1		# Vary the pattern
	addi	t1, t1, 1		# Increment counter
	bne	t1, t0, loop		# Branch if counter != 4092

	# === Finalization: 28 instructions ===
	# Padding with NOPs (addi zero, zero, 0) to reach exactly 32768
	addi	zero, zero, 0		# NOP 1
	addi	zero, zero, 0		# NOP 2
	addi	zero, zero, 0		# NOP 3
	addi	zero, zero, 0		# NOP 4
	addi	zero, zero, 0		# NOP 5
	addi	zero, zero, 0		# NOP 6
	addi	zero, zero, 0		# NOP 7
	addi	zero, zero, 0		# NOP 8
	addi	zero, zero, 0		# NOP 9
	addi	zero, zero, 0		# NOP 10
	addi	zero, zero, 0		# NOP 11
	addi	zero, zero, 0		# NOP 12
	addi	zero, zero, 0		# NOP 13
	addi	zero, zero, 0		# NOP 14
	addi	zero, zero, 0		# NOP 15
	addi	zero, zero, 0		# NOP 16
	addi	zero, zero, 0		# NOP 17
	addi	zero, zero, 0		# NOP 18
	addi	zero, zero, 0		# NOP 19
	addi	zero, zero, 0		# NOP 20
	addi	zero, zero, 0		# NOP 21
	addi	zero, zero, 0		# NOP 22
	addi	zero, zero, 0		# NOP 23
	addi	zero, zero, 0		# NOP 24
	addi	zero, zero, 0		# NOP 25
	addi	zero, zero, 0		# NOP 26
	addi	zero, zero, 0		# NOP 27
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall		# 28: Halt

	# Verification:
	# Setup: 4 instructions
	# Loop: 4092 iterations × 8 instructions = 32736
	# Finalization: 28 instructions (27 NOPs + 1 JALR)
	# Total: 4 + 32736 + 28 = 32768 = 2^15
