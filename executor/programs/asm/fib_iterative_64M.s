	.attribute	5, "rv64i2p1_m2p0"
	.globl	_start
_start:
	# Iterative Fibonacci - pure register arithmetic
	# ~64M steps
	#
	# Loop body: 5 instructions per iteration
	# 12799999 iterations × 5 = 63999995 + 4 setup/teardown = 63999999

	li	t0, 0			# a = fib(0) = 0
	li	t1, 1			# b = fib(1) = 1
	li	a0, 12799999		# iteration count

.loop:
	add	t2, t0, t1		# t2 = a + b
	mv	t0, t1			# a = b
	mv	t1, t2			# b = t2
	addi	a0, a0, -1		# n--
	bnez	a0, .loop		# loop if n != 0

	mv	a0, t1			# result = b
	li	a0, 0
	li	a7, 93
	ecall				# halt with result in a0
