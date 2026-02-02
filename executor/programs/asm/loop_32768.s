.attribute	5, "rv64i2p1_m2p0"
.globl	main
main:
	# Program with exactly 32768 instructions for profiling segmentation
	# Total: 32768 instructions (2^15)
	# Setup: 5 instructions
	# Loop: 10920 iterations * 3 instructions = 32760 instructions
	# Teardown: 3 instructions
	# Total: 5 + 32760 + 3 = 32768

	# === Setup: 5 instructions ===
	lui	t0, 3              # 1: t0 = 3 << 12 = 12288
	addi	t0, t0, -1368      # 2: t0 = 12288 - 1368 = 10920
	addi	t1, zero, 0        # 3: accumulator = 0
	addi	t2, zero, 1        # 4: increment = 1
	addi	zero, zero, 0      # 5: NOP (padding)

loop:
	# === Loop body: 3 instructions per iteration, 10920 iterations = 32760 instructions ===
	add	t1, t1, t2             # acc += 1
	addi	t0, t0, -1         # counter--
	bne	t0, zero, loop         # if counter != 0, continue

	# === Teardown: 3 instructions ===
	addi	a0, t1, 0          # return accumulator
	addi	zero, zero, 0      # NOP
	jalr	zero, 0(zero)      # halt
