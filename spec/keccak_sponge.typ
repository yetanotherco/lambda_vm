#import "/book.typ": book-page, aside
#import "/src.typ": load_config, load_chip
#import "/chip.typ": (
  compute_nr_interactions,
  render_chip_assumptions,
  render_chip_variable_table,
  total_nr_variables,
  total_nr_instantiated_columns,
  render_constraint_table,
  render_chip_padding_table,
)

#let config = load_config()
#let chip = load_chip("src/keccak_sponge.toml", config)

#show: book-page(chip.name)
#let sponge = raw(chip.name)
#let keccak = `KECCAK`
#let keccak_rnd = `KECCAK_RND`

The #keccak accelerator (@keccak) applies one permutation per `ECALL`.
Hashing a message of $ell$ bytes takes $ceil((ell + 1) / 136)$ permutations, and between them the guest must fold the next 136 bytes of message into the state itself --- seventeen loads, seventeen XORs and seventeen stores per block, all retired as ordinary VM cycles.
The #sponge accelerator absorbs that loop: one `ECALL` absorbs an entire run of whole rate blocks, at *one row per block*.

In this VM, we assign syscall number -4 to the #sponge accelerator. Its arguments are

/ `A0 = x10`: the address of the 200-byte state, which is updated in place,
/ `A1 = x11`: the address of $#`n` times 136$ bytes of message data,
/ `A2 = x12`: the number $#`n` > 0$ of rate blocks to absorb.

For each block $k = 0, ..., #`n` - 1$ the accelerator XORs the block into the first seventeen lanes of the state and then applies the keccak permutation $kappa$:
$ #`state`_j <- #`state`_j xor #`block`_(k, j) "for all" j < 17, quad "then" quad #`state` <- kappa(#`state`) . $
The eight capacity lanes are left out of the XOR.

#aside("Padding is the caller's job")[
  This accelerator only ever sees *whole* rate blocks: the `ECALL` takes a block count, not a byte length.
  A guest hashing $ell$ bytes absorbs $floor(ell / 136)$ whole blocks here and applies `pad10*1` to the remaining tail itself, pushing that one padded block through the #keccak accelerator (@keccak).
  Consequently the domain separator, the final padding bit and the rate are outside this chip's trusted boundary: no constraint here mentions them, and an incorrectly padded message is a caller bug that this chip will faithfully absorb.
]

= Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #sponge chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

#strong("Note on rows and calls.")
A row is a *block*, not a call.
The $#`n`$ rows of one call all carry the same `timestamp` --- an `ECALL` is a single CPU cycle --- and are distinguished by `seq`, which counts $0, 1, ..., #`n` - 1$.
`μ_first` marks the row that receives the `ECALL` and reads the state out of memory; `μ_last` marks the row that writes the final state back.
A single-block call is both at once.

#strong("Note on " + raw("state_in") + " and " + raw("block") + ".")
`state_in` is indexed as the specification's three-dimensional state, matching #keccak and #keccak_rnd, so lane $5#`y` + #`x`$ of memory is $#`state_in`_(#`x`, #`y`)$.
`block` is indexed by that same flat lane number, since the seventeen rate lanes do not form a rectangle in $(#`x`, #`y`)$.

= Assumptions
#render_chip_assumptions(chip, config)

= Constraints

The chip takes the `ECALL` off the bus on its first row only:
#render_constraint_table(chip, config, groups: "output")

That first row also reads the three argument registers.
Note that `A1` is read against `d`, this row's block address: `seq` is zero here (@sponge:c:first_seq_zero), so the two coincide.
#render_constraint_table(chip, config, groups: "registers")

#strong("Addressing.")
Rather than materializing a pointer per lane, as #keccak does, the chip forms each access address as a *linear* expression $#`base`_0 + #`offset`$ over the low address word, leaving the high word untouched.
This costs no columns at all, and is faithful exactly when the low word has room for the largest offset --- which is what the executor guarantees for both regions before the call is logged.
It is also fail-closed: a base whose low word has drifted produces `MEMW` tokens for addresses no memory cell matches, so the bus fails to balance.
`s_addr` is nevertheless kept as bytes, so that the alignment of the state pointer can be checked in-chip.
#render_constraint_table(chip, config, groups: "mem")

The reads all happen at the `ECALL`'s timestamp and the writeback one tick later.
This is what keeps every $(#`address`, #`timestamp`)$ pair unique: the writeback of the last row lands on the same 25 addresses as the state read of the first, and only the timestamp separates them.

Absorption itself is a lookup, one per byte of the rate.
The lookup range-checks both operands and pins the result, so `xored` needs no separate range check --- and `state_in` and `block` need none either.
#render_constraint_table(chip, config, groups: "absorb")

The permutation is delegated to the round chip #keccak_rnd (@keccak), which this chip drives exactly as the #keccak core chip does, except for the extra key discussed below:
#render_constraint_table(chip, config, groups: "round")

Consecutive blocks of a call are tied together by a self-referential bus, in the same shape as `CNB` in @commit: every row but the last sends its result forward, and every row but the first consumes the result of its predecessor.
Alongside the state, the chain carries the call's registers and the block index, so that all of them are pinned to the values the first row read out of the registers.
#render_constraint_table(chip, config, groups: "chain")

Finally, the flags and the block count:
#render_constraint_table(chip, config, groups: "bits")

= Why every permutation carries a key

All $#`n`$ permutations of one call share a single `timestamp`, and the round chip echoes whatever key it receives through its twenty-four rounds.
Suppose the `KECCAK` interaction carried only $(#`timestamp`, #`round`)$, as it did before this chip existed.
Take a call of three blocks, and let a dishonest prover assign the three permutation results to the three rows in the rotated order
$ #`state_out`_0 = kappa(#`absorbed`_2), quad #`state_out`_1 = kappa(#`absorbed`_0), quad #`state_out`_2 = kappa(#`absorbed`_1) . $
Every tuple sent on the `KECCAK` bus is still received exactly once, and every tuple received is still sent exactly once --- *the bus balances*.
The chain does not object either: it demands only that each row's `state_in` equal its predecessor's `state_out`, which this assignment satisfies.
What the guest gets back is the sponge of the same three blocks *in a different order*.
Nor does such a witness require inverting $kappa$: the prover computes $#`absorbed`_0$ from memory, then $kappa(#`absorbed`_0)$, and each remaining row's input follows from a value it already has.

The fix is to give every permutation of a call a key of its own.
The `KECCAK` interaction carries a third scalar `seq`; #keccak_rnd passes it along its round chain untouched, exactly as it passes `timestamp`; the #keccak core chip, which runs one permutation per `ECALL`, always sends zero.
This chip sends the block index, and the block index is pinned twice over: @sponge:c:first_seq_zero anchors the first row at zero, and the chain sends `seq + 1` where it receives `seq`, so the rows of one call carry $0, 1, ..., #`n` - 1$.
A chain long enough to wrap the field would need about $2^64$ rows.
The values being distinct, the two `KECCAK` legs of block $k$ are the only ones keyed $(#`timestamp`, k)$, and the rotation above no longer balances.

#strong("Why the rows of one call form a simple path.")
The keying argument assumes the rows of a call are what they look like: one first row, one last row, and a single chain between them.

- There is exactly one `μ_first` row per call, because the CPU sends one `ECALL` token per `ECALL` and a second first row would have to consume it twice.
- The chain neither forks nor merges: each token is emitted once (on $#`μ` - #`μ_last`$) and consumed once (on $#`μ` - #`μ_first`$). Duplicating a link would either duplicate the `ECALL` consumption upstream, or downstream duplicate a write to one $(#`address`, #`timestamp`)$ pair, which the memory argument rejects.
- The call has exactly $#`n`$ rows: the first row reads $#`n`$ out of `A2`, the chain carries it unchanged, and the last row must satisfy @sponge:c:last_seq_is_n. Pinning the low word alone would leave the alias $#`n` + p$, which is why @sponge:c:last_n_high_zero pins the high word to zero as well; this bounds a provable call to $#`n` < 2^32$, which is vacuous, as $#`n`$ rows must physically exist in this table.
- A call can neither stop early nor run forever: a non-last row's chain token must be consumed by a successor, and a last row must satisfy @sponge:c:last_seq_is_n, so $#`n` = 0$ admits no witness at all --- as does any row count other than $#`n`$.

= Padding
The #sponge table can be padded to the next power of two with the following value assignments:
#render_chip_padding_table(chip, config)

All-zero rows satisfy the arithmetic constraints and, having $#`μ` = #`μ_first` = #`μ_last` = 0$, contribute nothing to any bus.

= Notes/potential optimizations
- `state_out` is a full 200 columns, but the chain already carries it to the next row, where it reappears as `state_in`. A representation that shares one of the two would save 200 columns per row at the cost of a wider bus tuple.
- The eight capacity lanes are copied from `state_in` into the `KECCAK` tuple untouched. Only the seventeen rate lanes actually differ between `state_in` and `absorbed`.
- The chip could absorb the final padded block too, if the `ECALL` took a byte length rather than a block count and the padding bits were constrained in-chip. That would remove one `KECCAK` `ECALL` per hash, at the price of bringing `pad10*1` inside the trusted boundary.
