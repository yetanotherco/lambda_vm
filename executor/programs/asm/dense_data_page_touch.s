	.data
	.align 3
# `data_page_touch`, but on a page whose genesis is DENSE.
#
# That fixture's .data page holds one .dword, so its INIT column has 112 nonzero
# entries of 262,144. It passes the first half of the routing rule and the
# second half refuses it, so it exercises the SPARSE genesis route and leaves
# the prepared route with no table at fixture scale.
#
# THE RULE IT IS SIZED AGAINST, in `prover/src/continuation.rs`
# (`genesis_stack_plan`), read at 84a02b8c4. It has two parts and this fixture has
# to clear BOTH:
#
#   1. per page: `18 + 18*S` against `marginal_stacked_rows(18, n_fixed)`, the
#      rows one more carried page adds to the prepared leg. That marginal is a
#      FORM evaluated at the run's own bracket — 101 at one genesis page, 111 at
#      the block's thirty, 125 at four thousand — so the entry threshold is five
#      or six nonzero bytes and NOT a fixed number to quote here.
#   2. per candidate SET: its total savings against `PREPARED_LEG_ROWS`, the
#      175,066 rows the stacked chain costs once, however many pages ride it.
#      This is the part the 112-entry fixture fails: 1,923 saved rows do not buy
#      a 175,066-row chain.
#
# This one surrounds the touched cell with non-zero bytes so the page it lives
# on clears part 2 outright, whatever offset the linker puts it at.
#
# ⚠ WHY THE FILL IS ON BOTH SIDES. The counter has to be on a DENSE page, and
# where .data starts inside its page is the linker's business. With `n` bytes
# before and `n` after, the counter's own page holds at least `n` of them
# whatever its offset: a counter at the start of a page keeps the trailing fill,
# one at the end keeps the leading fill, and one in the middle keeps both. A
# single trailing fill would leave a counter near the end of a page with almost
# none of it.
#
# 32 KiB a side puts the floor at 32,768 nonzero entries. That is four orders
# past part 1 at any bracket, and its savings are `18 + 18*32,768 - marginal`,
# about 589,700 rows: 3.4x the chain part 2 weighs them against. The fixture is
# not sized to just barely qualify, which matters because the marginal is a form
# that moves with the run's page count and the chain's cost moves with the
# proof's posture.
	.fill 32768, 1, 0xA5
counter:
	.dword 0x123456789ABCDEF0
	.fill 32768, 1, 0x5A

	.text
	.attribute	5, "rv64i2p1"
	.globl	main
main:
	# The same five instructions `data_page_touch` runs: load, mutate, store
	# back a static global, so the page is genuinely ELF-backed and the cell
	# crosses an epoch boundary. Only the page's genesis differs.
	la	t0, counter		# 1: t0 = &counter
	ld	t1, 0(t0)		# 2: t1 = counter (0x123456789ABCDEF0)
	addi	t1, t1, 1		# 3: t1 += 1
	sd	t1, 0(t0)		# 4: counter = t1

	li	a0, 0
	li	a7, 93
	ecall		# 5: Halt
