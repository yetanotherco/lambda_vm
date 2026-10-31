#import "/book.typ": book-page
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
#let chip = load_chip("src/keccak.toml", config)

#show: book-page(chip.name)
#let keccak = raw(chip.name)

The #keccak chip applies the keccak permutation $kappa$ to a given memory range;
other aspects of keccak hashing (such as repeated permutation invocation, 
input padding and state initialization) fall outside the scope of this accelerator.

This permutation $kappa: FF_2^1600 -> FF_2^1600$ operates on 1600 bits and is composed of 24 applications of round-permutation $Lambda: FF_2^1600 times NN -> FF_2^1600$, where the additional parameter is the round constant.
$Lambda$ is defined as the composition $iota compose chi compose pi compose rho compose theta$, where only $iota$ depends on the round constant.
#footnote("More details on the KECCAK permutation: FIPS 202, NIST, " + link("https://csrc.nist.gov/pubs/fips/202/final"))

The keccak accelerator comprises two chips: a core chip that interacts with the memory --- loading the input and writing the output, and a round chip that applies the round permutation.


= Core chip
== Columns
#let nr_variables = total_nr_variables(chip)
#let nr_columns = total_nr_instantiated_columns(chip, config)
#let nr_interactions = compute_nr_interactions(chip)

The #keccak chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(chip, config)

== Constraints
In this VM, we assign syscall number -2 to the #keccak accelerator.
The chip therefore contributes the following interaction to the lookup-argument:
#render_constraint_table(chip, config, groups: "output")

The address containing the state to be permuted is passed in as argument `A0 = x10`.
The following constraints describe that this address is read into `addr` (@keccak:c:read_addr), from which `state_ptr` --- the collection of pointers to all lanes of the state --- is derived (@keccak:c:state_ptr).
The state is then read into `input_state`, while the `output_state` is written back to the indicated address (@keccak:c:load_store_state).
#render_constraint_table(chip, config, groups: "mem")

Lastly, the input state is pushed to the Keccak-round function, while the output after 24 rounds is taken off the bus:
#render_constraint_table(chip, config, groups: "round")

The table has a second mode (the sponge mode below), and its rows are split by two mode bits: $mu_p$ marks a row of this permutation ECALL, $mu_s$ a row of a Keccak-256 call.
Every interaction above except the two round-chip ones is gated by $mu_p$.
The two round-chip interactions are the only ones both modes share, and they are weighted by $mu_p + mu_s$: a row of either mode asks the round chip because of its own mode bit, so its output is the permutation of its input.
The column $mu$ is the sum of the two bits (it marks the real rows for the table's cuts), so with all three bits a row is in at most one mode:
#render_constraint_table(chip, config, groups: "mode")

== Padding
The #keccak table can be padded to the next power of two with the following value assignments:
#render_chip_padding_table(chip, config)

= Sponge mode: Keccak-256 of a buffer
#let sponge = load_chip("src/keccak_sponge.toml", config)

ECALL number -5 (`KECCAK256`) hashes a buffer in guest memory: `A0 = x10` holds its address, `A1 = x11` its length in bytes and `A2 = x12` the address the 32-byte digest is written to.
Both addresses may have any alignment, and the digest may overlap the input: the input is read at the call's timestamp $T$ and the digest written at $T + 1$, so the hash is of the bytes as they were before the call.
The hash is Keccak-256 with the original Keccak padding `0x01 ... 0x80` over the 136-byte rate.
The call is refused unless every block's 136-byte window and the 32-byte digest stay below one $2^32$ boundary of the low address word: $(#`x10` mod 2^32) + 136 dot (floor(#`x11` / 136) + 1) <= 2^32$ and $(#`x12` mod 2^32) + 31 < 2^32$.

The call is proven by the #keccak chip itself, in a second mode.
Two mode bits split the rows: `MU_P` marks a row of the permutation ECALL above, `MU_S` a row of a Keccak-256 call, with $mu = #`MU_P` + #`MU_S`$ and all three bits, so a row is in at most one mode.
Every interaction and constraint of the permutation mode is gated by `MU_P`, and every one of the sponge mode by `MU_S`, `first`, `last` or a read-mask bit that is zero off a sponge row.
The two round-chip interactions are weighted by `MU_P + MU_S`, so a sponge row's mode bit itself puts it on the round chip (on a sponge row that sum is `MU_S`, as written below).
A call of $n$ bytes takes $floor(n / 136) + 1$ rows, one per permutation: row $j$ reads block $j$ of the input, XORs the padded block into the rate of the state the previous row's permutation ended in, and sends the result to the round chip; the last row writes the first four lanes of its own output.

== Columns
#let nr_variables = total_nr_variables(sponge)
#let nr_columns = total_nr_instantiated_columns(sponge, config)
#let nr_interactions = compute_nr_interactions(sponge)

A sponge row uses #nr_variables variables, expressed using #nr_columns columns, and leverages #nr_interactions interaction(s):
#render_chip_variable_table(sponge, config)

On a sponge row the chip's `timestamp` holds the key $(T, j)$: the call's timestamp and the block index.
The round chip links a request to its reply by that key alone, so it has to be unique per permutation: $j$ differs between the rows of one call, $T$ between calls, and a permutation row's key is $(T', 0)$ at its own ECALL's timestamp.
Every memory access of a sponge row uses $(T, 0)$ or $(T + 1, 0)$, never $j$.

== Constraints
The first row receives the ECALL and reads its three operands. The length is read with a zero high word, so it is below $2^32$. The first row is block 0 and starts from the zero state:
#render_constraint_table(sponge, config, groups: "call")

Consecutive rows of one call are linked by the `KECCAK_SPONGE` interaction, which carries the remaining byte count, the next block's address and the state; the receiving row takes the previous output's rate into `prev` and its capacity directly into the capacity lanes of `input_state`:
#render_constraint_table(sponge, config, groups: "chain")

A lane wholly inside the input is read as one 8-byte access; the bytes of the lane the input ends in are read one at a time. Byte 7 of a lane is never read alone:
#render_constraint_table(sponge, config, groups: "read")

The read mask $s_(L,b) = "full"_L + "part"_(L,b)$ ($b < 7$), $s_(L,7) = "full"_L$, is a prefix of the block. A row that is not last reads all 136 bytes; the last row reads exactly `count` bytes, at most 135. Since `count` starts at the length and falls by 136 per row, this pins the number of rows to $floor(n / 136) + 1$: the KECCAK rows of one proof are at most `SPONGE_MAX_ROWS` $= 2^40$ (the block format's caps are checked against it when the prover is built), so $n - 136 j$ lies in $(-2^48, 2^32)$ and equals a value in $[0, 135]$ in the field only when it does in the integers. (The argument needs only $136 dot N + 2^32 < p$, about $N < 2^56$ rows.)
Bytes past the input are never read and are zero:
#render_constraint_table(sponge, config, groups: "mask")

The padded block is a linear form of the columns: $m'_(L,b) = "msg"_(L,b) + s_"prev" - s_(L,b)$, where $s_"prev"$ is the mask bit of the byte before ($1$ before byte 0), plus `0x80` on byte 135 of the last row. The term $s_"prev" - s$ is the `0x01` right after the input; on byte 135 of a last block of 135 input bytes the two meet as `0x81`. The permutation input is the previous rate XOR the padded block:
#render_constraint_table(sponge, config, groups: "absorb")

The permutation itself is the round chip's, as in the permutation mode:
#render_constraint_table(sponge, config, groups: "round")

The last row writes the digest, the first four lanes of its own output, at $T + 1$:
#render_constraint_table(sponge, config, groups: "digest")

The block's last byte and the digest's last byte stay below $2^32$ in the low address word, so the offsets above add to the low word without a carry.
The equation on `src_end` (@keccak_sponge:c:src_end) is more than a range check: the link sends $"src_end" + 1$ as the next row's `src`, so it is also what places every block after the first right after the one before; without it a block could be read from any address.
#render_constraint_table(sponge, config, groups: "range")

= Round chip
#let round_chip = load_chip("src/keccak_round.toml", config)
#let keccak_rnd = raw(round_chip.name)

== Columns
#let nr_variables = total_nr_variables(round_chip)
#let nr_columns = total_nr_instantiated_columns(round_chip, config)
#let nr_interactions = compute_nr_interactions(round_chip)

The #keccak_rnd chip is comprised of #nr_variables variables that are expressed using #nr_columns columns and leverages #nr_interactions interaction(s):
#render_chip_variable_table(round_chip, config)

#strong("Note on " + raw("start") + ".")
`start` contains the state to which the permutation should be applied.
Its three-dimensional array mimics the specification's three-dimensional state
#footnote("FIPS 202, NIST, Section 3.1 (" + link("https://csrc.nist.gov/pubs/fips/202/final") + ")")
and orders the bits as prescribed.
#footnote("FIPS 202, NIST, Section B.1, Algorithm 10 (" + link("https://csrc.nist.gov/pubs/fips/202/final") + ")")

#strong("Note on " + raw("rnc") + " and " + raw("rbc") + ".")
Rho rotates every lane by a rotation offset in $[0, 64)$.
These offsets are identical for every round.
#footnote("FIPS 202, NIST, page 13, Table 2 (" + link("https://csrc.nist.gov/pubs/fips/202/final") + ")")
We decompose each offset in three components: the lower nibble (4 bits) are represented by `rnc`, while the upper two bits are represented by as `Bit`s in `rbc`.
That is, $#`rho_offset[x][y]` = #`rnc[x][y]` + 16 dot #`rbc[x][y][0]` + 32 dot #`rbc[x][y][1]`$.


== Constraints

The following constraints ensure that `theta` captures the state after applying the first subpermutation of the round-permutation: $theta$.
Note here that `Cxz_left` and `Cxz_right` do have to be range-checked; it cannot be assumed that this implicitly follows from @keccak:c:Dxz combined with `rotated_Cxz`'s definition.
#render_constraint_table(round_chip, config, groups: "theta")

Next, we constrain that `rho` captures the state after applying subpermutation $rho$.
Note here as well that `rot_left` and `rot_right` do have to be range-checked; it cannot be assumed that this implicitly follows from later constraints.
#render_constraint_table(round_chip, config, groups: "rho")

Observe that the lane-permutation performed by $pi$ is absorbed in `pi`'s definition.
The next permutation that is constrained in $chi$:
#render_constraint_table(round_chip, config, groups: "chi")

Lastly, the round constants are added to one of the lanes in the state.
`iota` contains the updated lane.
In the definition of `out`, the output of `chi` and `iota` is combined to construct the output of the permutation.
#render_constraint_table(round_chip, config, groups: "iota")

Lastly, the round chip contributes the following interactions to the lookup:
#render_constraint_table(round_chip, config, groups: "io")

== Notes/potential optimizations
- one does not have to repeat `addr` in `state_ptr`; this saves 4 columns and 4 `IS_HALF` checks.
- step $rho$ does not need to be applied to `state[0][0]`; its has a zero-shift. This saves 16 columns and 4 `HWSL` interactions.
- when the output of `HWSL` are `Byte`s mapped as `Half`s, we find that out of every four output bytes, at least one is zero. 
  Since `rnc` is constant, @keccak:c:rho_rotation makes those zero-bytes show up in `rot_left` and `rot_right` at constant locations.
  This means 96 columns can be removed from the chip at no cost.
  Likewise, 96 `IS_BYTE` interactions can be dropped from @keccak:c:range_rot_left and @keccak:c:range_rot_right.
  - the shift-constants are equivalent to $1 mod 16$ for $(#`x`, #`y`) = (1, 0)$ and $-1 mod 16$ for $(2, 3)$. This means that for those lanes it suffices to constrain `rot_left`/`rot_right` as `Bit`s rather than `Byte`s, saving an additional 8 `IS_BYTE` interactions.
- $#`rc[2]` = #`rc[4]` = #`rc[5]` = #`rc[6]` = 0$. As such, those elements need not be stored in `rc`, and need not be XORed into the state in the $iota$-step. This saves 8 columns and 4 `XOR_BYTE` interactions.
- when executed in large volumnes, `KECCAK_RND` could benefit from having a three-way XOR lookup table. With this in place, the 80 interactions in @keccak:c:theta_cxz_start and @keccak:c:theta_cxz could be dropped.
  Likewise, 80 columns could be removed from the chip (a \~5% savings).

= Round constant lookup
#let rc_chip = load_chip("src/keccak_rc.toml", config)
#let keccak_rc = raw(rc_chip.name)

== Columns
#let nr_variables = total_nr_variables(rc_chip)
#let nr_columns = total_nr_instantiated_columns(rc_chip, config)

We provide the round constants through a short precomputed lookup table: #keccak_rc.
#render_chip_variable_table(rc_chip, config)
#render_constraint_table(rc_chip, config)