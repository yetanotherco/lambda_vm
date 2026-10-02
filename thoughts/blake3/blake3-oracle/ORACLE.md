# BLAKE3 Compression-Function Oracle

**Purpose.** Trust anchor for a future BLAKE3 accelerator (precompile chip) in
the Lambda VM STARK prover. Phase 1 = this oracle (the reference `f` + external
validation + chip-contract reuse map). Phase 2 = chip constraint design, gated
against this oracle. The oracle is the reference the chip's trace generation and
constraints will be checked against; a wrong oracle silently poisons everything
downstream, so the validation section is the load-bearing part.

**Scope.** The reference is the BLAKE3 **compression function** `f`, NOT the full
tree hash. `blake3_ref.py` also contains a full tree hasher, but that exists
*only* so `f` can be validated against the official whole-hash test vectors. The
chip implements `f`; it does not implement the tree.

---

## 1. Validation status: **VALIDATED**

`test_oracle.py` passes all of the following (re-run: `./venv/bin/python test_oracle.py`):

| # | External anchor | Independent of our code? | What it covers | Result |
|---|---|---|---|---|
| 1 | Official **`test_vectors.json`** (BLAKE3 team, `test_vectors/test_vectors.json`, fetched from the BLAKE3 GitHub repo) | Yes — authored by the BLAKE3 authors | 35 input lengths (0 … 102400 B) × 3 modes (default hash, keyed hash, derive-key), extended (131-byte) output | **PASS 35/35 × 3** |
| 2 | Official **`blake3` PyPI package** v1.0.9 (the reference Rust implementation via FFI) | Yes — separate codebase | 23 randomised input lengths (0 … 100000 B) × {default, XOF, keyed, derive-key} = 92 differential checks | **PASS 92/92** |
| 3 | **Plonky3 `blake3-air`** compression, ported in `test_oracle.py` from `others/Plonky3/blake3-air/src/generation.rs` | Yes — Plonky3 team, different codebase | 20 000 random `(h, m, t, block_len)` compared at the **compression-function level** (flags = 0, 7 rounds) | **PASS 20000/20000** |

Anchors 1–2 validate `f` *indirectly but exhaustively*: the whole-hash path
drives `f` under every flag combination (`CHUNK_START`, `CHUNK_END`, `PARENT`,
`ROOT`, `KEYED_HASH`, `DERIVE_KEY_CONTEXT`, `DERIVE_KEY_MATERIAL` and their
compositions) and a wide range of counters (chunk indices 0…99 for the 102400 B
case, plus XOF output-block counters). Anchor 3 validates `f` **directly** at the
compression level against a second independent implementation (flags = 0 only,
since Plonky3's AIR hardcodes `v[15] = 0`).

The constants were independently cross-checked: `IV` and `MSG_PERMUTATION` in
`blake3_ref.py` match `others/Plonky3/blake3-air/src/constants.rs` (`IV` stored
there as `[lo16, hi16]` pairs; `MSG_PERMUTATION = [2,6,3,10,7,0,4,13,1,11,12,5,9,14,15,8]`).

> Note: the BLAKE3 repo's `reference_impl/reference_impl.py` returned HTTP 404 at
> fetch time (repo layout changed), so it is **not** used. `f` was written from
> the spec's G-function definition; the three anchors above stand on their own.

---

## 2. Precise definition of both variants

Everything is on 32-bit unsigned words, little-endian. `⊞` = add mod 2³²,
`⊕` = XOR, `x ⋙ n` = rotate-right by `n` bits.

### 2.1 Constants

```
IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]

MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
```

### 2.2 The G function (quarter round)

`G(v, a, b, c, d, mx, my)` mutates working-state words `v[a], v[b], v[c], v[d]`:

```
v[a] = v[a] ⊞ v[b] ⊞ mx
v[d] = (v[d] ⊕ v[a]) ⋙ 16
v[c] = v[c] ⊞ v[d]
v[b] = (v[b] ⊕ v[c]) ⋙ 12
v[a] = v[a] ⊞ v[b] ⊞ my
v[d] = (v[d] ⊕ v[a]) ⋙  8
v[c] = v[c] ⊞ v[d]
v[b] = (v[b] ⊕ v[c]) ⋙  7
```

### 2.3 The round

Given the (already permuted-for-this-round) 16-word schedule `m`:

```
# columns
G(v, 0, 4,  8, 12, m[0],  m[1])
G(v, 1, 5,  9, 13, m[2],  m[3])
G(v, 2, 6, 10, 14, m[4],  m[5])
G(v, 3, 7, 11, 15, m[6],  m[7])
# diagonals
G(v, 0, 5, 10, 15, m[8],  m[9])
G(v, 1, 6, 11, 12, m[10], m[11])
G(v, 2, 7,  8, 13, m[12], m[13])
G(v, 3, 4,  9, 14, m[14], m[15])
```

### 2.4 The compression function `f` (parameterised by `ROUNDS`)

Inputs: `h[0..8]` (chaining value, 8×u32), `m[0..16]` (message block, 16×u32),
`t` (u64 counter), `block_len` (u32, 0..64), `flags` (u32).

```
v[0..8]  = h[0..8]
v[8..12] = IV[0..4]
v[12]    = t mod 2³²          # counter low
v[13]    = t >> 32            # counter high
v[14]    = block_len
v[15]    = flags

schedule = m
for r in 0 .. ROUNDS-1:
    round(v, schedule)
    schedule = permute(schedule)     # trailing permute after last round is unused

# feed-forward (produces the FULL 16-word output)
for i in 0..8:
    out[i]   = v[i]   ⊕ v[i+8]
    out[i+8] = v[i+8] ⊕ h[i]
return out[0..16]
```

The truncated 8-word chaining value used inside the tree is `out[0:8]`. The XOF
root output uses **all 16** output words — this is why `f` returns 16 words.

### 2.5 Variant A — standard: `ROUNDS = 7`

The function above with `ROUNDS = 7`. This is standard BLAKE3, validated by
anchors 1–3.

### 2.6 Variant B — nonstandard: `ROUNDS = 6`

**Exactly** the function in §2.4 with `ROUNDS = 6`: rounds 0..5 are applied,
round `r` mixing `permute^r(m)`, followed by the identical feed-forward. The
ONLY difference from Variant A is the loop bound. This is a **NONSTANDARD**
function; **no external test vectors exist**. Its anchoring is derivative:

* **(a) Code-diff anchor.** In `blake3_ref.py`, `compress_6round(...)` is literally
  `compress(..., rounds=6)` — same IV, same initial-state layout, same G, same
  message permutation schedule, same feed-forward. `test_oracle.py`
  (`test_6round_derivation`) asserts `compress_6round == compress(rounds=6)` and
  that it differs from `ROUNDS=7` on 2000/2000 random inputs.
* **(b) Canonical vectors.** 10 deterministic vectors (fixed seeds 0..9) are
  generated and recorded below. These are Variant B's canonical reference going
  forward. Full inputs/outputs are in `canonical_6round_vectors.json`.

#### Canonical 6-round vectors (seeds 0..9)

Each row: 32-hex-digit words. `out` is the full 16-word output concatenated
(`out[0]` first). Inputs `h` (8 words), `m` (16 words), `t`, `block_len`,
`flags` are in `canonical_6round_vectors.json`; a summary fingerprint is shown
here (`out[0]` and `out[15]`) so the doc alone pins the vectors' identity.

| seed | t | block_len | flags | out[0] | out[15] |
|---|---|---|---|---|---|
| 0 | 0xb4e1357d4a84eb03 | 42 | 0x34 | 0xced9d1ff | 0xb75f3915 |
| 1 | 0xc74803e31ba16215 | 50 | 0x5e | 0xf2a972e9 | 0xdfb91125 |
| 2 | 0x7604e4b4e73695c3 | 58 | 0x7c | 0x5aa6b114 | 0x775f2f92 |
| 3 | 0x92d3043afcf249f3 | 36 | 0x1f | 0xeed92fab | 0xdc293166 |
| 4 | 0x49c7b59b995253fd | 57 | 0x29 | 0xca00bda3 | 0x7561eb37 |
| 5 | 0x6a3753915c76f18a | 18 | 0x43 | 0x14a9f66f | 0xbb7a485d |
| 6 | 0x390567c27bd6aa42 | 26 | 0x03 | 0x32a6ff70 | 0x2a7a62b2 |
| 7 | 0x12bd4acefaecbd38 | 53 | 0x2a | 0xa632ad45 | 0xf3f33689 |
| 8 | 0x329911da9fbd8735 | 19 | 0x5b | 0x913b2ae1 | 0x3c5a654b |
| 9 | 0xeaeb999b8a2e547e | 64 | 0x15 | 0xf5ee9114 | 0xd18a8b94 |

(To re-derive: `random.Random(seed)` then draw `h=8×u32, m=16×u32, t=u64,
block_len∈[0,65), flags∈[0,128)` in that order — see
`test_oracle.canonical_6round_vectors`.)

---

## 3. Chip-contract reuse map

Every primitive op of `f` mapped onto the existing precomputed-table contracts.
Citations are to `prover/src/tables/bitwise.rs` (the 2²⁰-row BITWISE table) and
the KECCAK chips, which are the architectural template for a byte-oriented
delegation chip.

The BITWISE table (`bitwise.rs:97`, `NUM_ROWS = 256·256·16 = 2²⁰`) is indexed by
`(X: byte, Y: byte, Z: 4-bit)` and provides these receivers
(`bitwise.rs:715` `bus_interactions`):

* `ByteAlu[opsel, X, Y] → out` — byte AND/OR/XOR (`bitwise.rs:865-921`; `opsel`
  ∈ {AND, OR, XOR}). The output column is a table column, so a `ByteAlu` send
  **simultaneously range-checks X and Y to be bytes and pins `out` to the exact
  result** — no separate range check needed on any of the three.
* `ARE_BYTES[X, Y]` — range-check two bytes (`bitwise.rs:783`; pass `Y=0` for a
  single byte).
* `IS_HALF[X + 256·Y]` — range-check a 16-bit halfword (`bitwise.rs:798`).
* `IS_B20[...]` — 20-bit range check (`bitwise.rs:813`).
* `HWSL[X + 256·Y, Z] → [SLL, SLLC]` — halfword shift-left (`bitwise.rs:831`),
  where `SLL = (hw << Z) & 0xFFFF`, `SLLC = hw >> (16 - Z)` (`bitwise.rs:135-141`),
  `Z ∈ [0,16)`.
* `MSB8`, `MSB16`, `ZERO` — not needed by BLAKE3.

### 3.1 Op-by-op mapping

| BLAKE3 primitive | Existing contract | How | Cost |
|---|---|---|---|
| **32-bit XOR** (`v[d]⊕v[a]`, `v[b]⊕v[c]`, feed-forward) | `ByteAlu[XOR]` | 4 byte-XOR lookups per 32-bit word, one per byte, exactly as `keccak_rnd` does θ/χ/ι XORs (`keccak_rnd.rs:692-718`). Inputs & output auto-range-checked by the lookup. | 4 sends / 32-bit XOR |
| **`⋙ 16`** | *free* — byte relabeling | rotr16 permutes bytes `[b0,b1,b2,b3] → [b2,b3,b0,b1]`. **VERIFIED** exhaustively (100k random words). No lookup, no column: just re-address the bytes at the next use. | 0 |
| **`⋙ 8`** | *free* — byte relabeling | rotr8 → `[b1,b2,b3,b0]`. **VERIFIED**. | 0 |
| **`⋙ 12`** | `HWSL` (+ `ARE_BYTES`) | rotr12 = rotl20; per the keccak-ρ pattern, HWSL each of the 2 halfwords by `rnc=4`, then a halfword rotate by `rbc=1`, recombining `newlo = SLL_lo + SLLC_hi`, `newhi = SLL_hi + SLLC_lo` (non-overlapping bit ranges ⇒ add = OR), then swap the two halfwords. **VERIFIED** (50k random). Range-check the 4 output bytes with `ARE_BYTES` (as keccak does on ρ outputs, `keccak_rnd.rs:768-790`). | 2 HWSL + 2 ARE_BYTES / rot |
| **`⋙ 7`** | `HWSL` (+ `ARE_BYTES`) | rotr7 = rotl25; same pattern with `rnc=9`, `rbc=1`. **VERIFIED**. `rnc=9 < 16` fits HWSL's 4-bit `Z`. | 2 HWSL + 2 ARE_BYTES / rot |
| **32-bit add mod 2³²** (2-operand `v[c]⊞v[d]`) | carry-bit polynomial constraint + range-check | Exactly `emit_add_pair`'s low half (`templates.rs:334`): with sum `s` committed and range-checked, `carry = (a + b − s)·2⁻³²` is constrained `carry·(1−carry)=0` (`INV_SHIFT_32 = (2³²)⁻¹`, `templates.rs:26`). Sum bytes are range-checked *for free* because `s` immediately feeds an XOR lookup. | 1 poly constraint / add |
| **3-operand add mod 2³²** (`v[a]⊞v[b]⊞mx`) | carry constraint (see §5 open Q) | `a+b+m < 3·2³²` ⇒ carry ∈ {0,1,2}. Either one virtual `k(k−1)(k−2)=0` (deg 3) or two chained `emit_add_pair` steps (deg ≤ 2). See open question O1. | 1–2 poly constraints / add |
| **message schedule** (`permute` between rounds) | *free* — wiring | Fixed compile-time permutation of the 16 input words per round; round `r` references `permute^r`-indexed message columns. No table, exactly like `keccak_rnd` inlines `KECCAK_RHO` offsets as compile-time constants. **Confirmed.** | 0 |
| **IV constants, flags, block_len, counter split** | constants / direct columns | `IV[0..4] → v[8..12]`, `t` split into `v[12]=t mod 2³²`, `v[13]=t>>32`, `v[14]=block_len`, `v[15]=flags`. Constants inlined; counter split is two committed words range-checked. | ~0 |

**No BLAKE3 op lacks an existing contract.** All arithmetic reduces to
`ByteAlu[XOR]`, `HWSL`, `ARE_BYTES`, and the `emit_add_pair` carry template —
every one already exercised by the KECCAK chips. The 32-bit-add carry range
checks fit `ARE_BYTES`/`IS_HALF` (the sum's bytes/halfwords), and the carry
itself is a `{0,1}` (or `{0,1,2}`) polynomial bit, not a table lookup.

### 3.2 Why the two "free" rotations are actually free

`ByteAlu` and `HWSL` operate at byte / halfword granularity, and the working
state is stored as bytes. A rotate-right by a multiple of 8 is a permutation of
byte positions, so the constraint at the *consuming* site simply reads the bytes
in rotated order (the same trick keccak uses implicitly). Only `⋙12` and `⋙7`
cross byte boundaries and therefore need HWSL. This means **half** of BLAKE3's
rotations cost nothing.

---

## 4. I/O column boundary sketch

Analogous to keccak's 200-byte state handoff (`keccak.rs`), the chip's
bus-facing tuple. Recommended **granularity: bytes** — because XOR (the dominant
op) needs byte operands and the two byte-aligned rotations are free at byte
granularity; adds read bytes as a linear combination (`AddOperand::from_dword_bl`,
`templates.rs:191`) so byte storage costs them nothing.

**Chip input** (read from guest memory via the ECALL/MEMW interface, exactly the
keccak pattern `keccak.rs:160-449`: ECALL receiver binds the syscall + timestamp,
a MEMW read of `x10` binds the state pointer, then per-word MEMW reads):

| field | size | granularity |
|---|---|---|
| `h[0..8]` chaining value | 8 words = 32 B | bytes |
| `m[0..16]` message block | 16 words = 64 B | bytes |
| `t` counter | u64 = 8 B | 2 words (lo, hi), byte-stored |
| `block_len` | u32 | 1 word |
| `flags` | u32 | 1 word |

**Chip output** (written back to memory):

| field | size | granularity |
|---|---|---|
| `out[0..16]` | 16 words = 64 B | bytes |

For the truncated (CV-only) call sites the guest reads back `out[0:8]`; the chip
always produces the full 16 words (the XOF root needs them).

**Internal handoff (if one-row-per-round).** If the chip mirrors keccak's
round-chip split, a `Blake3Round` bus carries `(timestamp, round_index,
state[16 words as 64 bytes], message[16 words])` from row `r` to row `r+1`,
mirroring `keccak_rnd`'s `(timestamp, round, start[200])` handoff
(`keccak_rnd.rs:441-515`). Note BLAKE3 must also carry the (round-permuted)
message down the rounds, unlike keccak whose round chip has no message input.

---

## 5. Cost estimate & recommended granularity

Cost model (given): a **committed** cell is expensive; each **bus send** ≈ 1.5
base cells of aux; **max constraint degree 3** is a hard cap.

### Per-round work (8 G calls; each G = 2 three-operand adds, 2 two-operand adds,
4 XORs, 4 rotations of which 2 are free):

| resource | per round | note |
|---|---|---|
| `ByteAlu[XOR]` sends | 8·4·4 = **128** | 4 XORs/G × 4 bytes |
| `HWSL` sends | 8·2·2 = **32** | 2 non-free rots/G × 2 halfwords |
| `ARE_BYTES` (rot-output range checks) | ~**32** | 2 rots/G × 4 bytes ÷ 2-per-send |
| add carry constraints | ~**48** | (16 three-op + 16 two-op adds)/round |
| committed byte-cells (state + G intermediates + carries) | ~**450** | ~10 words/G committed × 8 G × 4 B + input state |

Bus sends/round ≈ 128 + 32 + 32 ≈ **~190**; aux ≈ 190 × 1.5 ≈ **~290** base
cells; committed ≈ **~450**. Total ≈ **~740 cell-equivalents/round**.

### Per compression (7 rounds + feed-forward + I/O):

* XOR lookups: 7·128 + 64 (feed-forward) ≈ **~960**
* HWSL lookups: 7·32 ≈ **~224**
* Range-check sends: ~7·32 + I/O ≈ **~250**
* **Total bus sends ≈ ~1450**, aux ≈ ~2200 base cells
* Committed ≈ 7·450 + I/O ≈ **~3300** base cells
* **Grand total ≈ ~5000–6000 cell-equivalents per compression**, dominated by
  the ~960 byte-XOR lookups.

For scale: a keccak-f permutation is ~24 rounds × 1480 cols. A BLAKE3
compression is roughly **¼–⅓ of one keccak permutation**.

### Recommended layout

BLAKE3 has only **7 rounds** (vs keccak's 24). Two viable shapes:

* **A. One row per round** (keccak-style): ~450–750 columns/row × 7 rows, plus a
  `Blake3Round` internal handoff bus carrying state **and** the permuted message.
  Fewer columns, but the message-carrying handoff is extra bus traffic keccak
  doesn't have.
* **B. One row per compression** (fully unrolled): ~3000–3500 columns in a single
  row; no internal handoff bus, no round-index bookkeeping. The message schedule
  is pure compile-time wiring so unrolling is natural.

**Recommendation: start with B (one row per compression).** With only 7 rounds
the column count (~3k) is comparable to keccak's per-round width, and eliminating
the internal state+message handoff bus removes the biggest source of aux cost and
constraint complexity. Revisit A only if the committed width dominates trace-area
budget. Either way the cell total is the same order (~5–6k).

---

## 6. Open questions for the chip phase

* **O1 — 3-operand add carry granularity (the main one).** `v[a] = v[a] ⊞ v[b] ⊞
  mx` sums three 32-bit values, so the carry-out is in **{0,1,2}**, not {0,1}.
  `emit_add_pair` (`templates.rs:334`) only handles a `{0,1}` carry. Options:
  1. **One virtual carry ∈ {0,1,2}:** commit the sum `s` (range-checked),
     `k = (a+b+m−s)·2⁻³²`, constrain `k(k−1)(k−2)=0`. This is **degree 3** — at
     the cap. It cannot also be `μ`-gated (that would be degree 4). Feasible only
     if padding rows satisfy it ungated (all-zero padding ⇒ `k=0` ⇒ satisfied,
     the keccak padding convention — verify this holds for BLAKE3 padding).
  2. **Two chained adds:** `t = a ⊞ b` (carry ∈ {0,1}), then `a' = t ⊞ mx` (carry
     ∈ {0,1}), each via `emit_add_pair`, at the cost of one extra committed 32-bit
     intermediate `t` per 3-operand add (16 extra words/round). Stays degree ≤ 2,
     so it can be `μ`-gated to degree 3. Simpler and gate-friendly.
  * **Recommendation:** option 2 (chained adds) unless the extra committed width
    is measured to hurt — it keeps every add uniformly `{0,1}`-carry and leaves
    degree headroom for `μ`-gating. Decide with a bench once the chip exists.

* **O2 — carry-bit gating & padding.** Decide whether add-carry and rot
  constraints are `μ`-gated (like `keccak_rnd`'s IS_BIT, `keccak_rnd.rs:914`) or
  rely on all-zero padding rows satisfying them ungated. This interacts with O1's
  degree budget.

* **O3 — one-row-per-round vs unrolled (§5).** Ties to O2 and to whether the
  message schedule is carried on a handoff bus or wired per-row at compile time.

* **O4 — flags coverage of the direct anchor.** Anchor 3 (Plonky3) only checks
  `flags = 0` at the compression level; non-zero flags are validated only through
  the whole-hash anchors 1–2. If the chip is ever exercised on raw compression
  inputs with arbitrary flags outside a valid tree, add a direct differential
  check against the PyPI package's low-level API if/when it exposes `compress`
  (it currently does not).

* **O5 — counter (`t`) width.** The whole-hash anchors drive `t` only up to ~99
  (chunk index) plus small XOF counters. The chip must accept a full u64 `t`
  (`v[12]/v[13]` split). Constants and the split are validated structurally, but
  if the chip supports enormous counters, add a targeted vector. (Plonky3 anchor
  already exercises random full-width u64 `t` — so this is **covered**.)

* **O6 — endianness at the memory boundary.** BLAKE3 words are little-endian;
  the byte-granular I/O sketch (§4) assumes LE byte order in memory. Confirm
  against the guest's `blake3` calling convention when wiring MEMW.

---

## 7. File manifest

```
blake3-oracle/
├── blake3_ref.py                  # reference f (ROUNDS-parameterised) + 6-round variant + tree hasher
├── test_oracle.py                 # anchors 1-3 + 6-round derivation + canonical-vector emitter
├── ORACLE.md                      # this document
├── official_test_vectors.json     # BLAKE3 team vectors (fetched, unmodified)
├── canonical_6round_vectors.json  # 10 canonical Variant-B vectors (generated)
└── venv/                          # python venv with the official `blake3` pkg (anchor 2)
```

No repository files were modified.
