	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Program with exactly 128 instructions for testing segmentation
	# Total: 128 instructions
	# Setup: 4 instructions
	# Loop: 40 iterations * 3 instructions = 120 instructions
	# Teardown: 4 instructions
	# Total: 4 + 120 + 4 = 128

	# === Setup: 4 instructions ===
	addi	t0, zero, 40		# 1: counter = 40
	addi	t1, zero, 0		# 2: accumulator = 0
	addi	t2, zero, 1		# 3: increment = 1
	addi	zero, zero, 0		# 4: NOP (padding)

loop:
	# === Loop body: 3 instructions per iteration, 40 iterations = 120 instructions ===
	add	t1, t1, t2		# acc += 1
	addi	t0, t0, -1		# counter--
	bne	t0, zero, loop		# if counter != 0, continue

	# === Teardown: 4 instructions ===
	addi	a0, t1, 0		# 125: return accumulator
	addi	zero, zero, 0		# 126: NOP
	addi	zero, zero, 0		# 127: NOP
	jalr	zero, 0(zero)		# 128: halt
