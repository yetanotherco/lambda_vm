	.attribute	5, "rv64i2p1_m2p0"
	.globl	main
main:
	# Iterative Fibonacci - pure register arithmetic
	# ~1200k steps → CPU pads to 2^21
	#
	# Loop body: 5 instructions per iteration
	# 239999 iterations × 5 = 1199995 + 4 setup/teardown = 1199999

	li	t0, 0			# a = fib(0) = 0
	li	t1, 1			# b = fib(1) = 1
	li	a0, 239999		# iteration count

.loop:
	add	t2, t0, t1		# t2 = a + b
	mv	t0, t1			# a = b
	mv	t1, t2			# b = t2
	addi	a0, a0, -1		# n--
	bnez	a0, .loop		# loop if n != 0

	mv	a0, t1			# result = b
	mv	a1, a0
	li	a0, 0
	li	a7, 93
	ecall				# halt with result in a0
