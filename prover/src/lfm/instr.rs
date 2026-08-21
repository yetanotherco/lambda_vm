//! The LFM instruction set: eight algebra-shaped operations, no control flow.
//!
//! Addresses are dense indices assigned in emission order, so every operand
//! address is strictly below its destination (acyclicity by construction —
//! the validator re-checks it anyway). Every write carries its statically
//! known read count `mult`, backfilled by the compiler from the builder's
//! read counters. There is no pc, no branch, no computed address, no halt:
//! the program is a straight line and the dataflow is the execution.
//!
//! Assertions are deliberately not an instruction: `assert_eq(a, b)` lowers
//! to `diff = a - b; _ = diff / ZERO` under the division convention
//! `0/0 = 1, x/0 = error` — the AIR's division constraint `in2·out = in1`
//! with `in2 = 0` forces `in1 = 0`, and the executor errors on a nonzero
//! numerator.

use crate::tables::types::FE;

use super::layout;

/// A write-once memory cell address (dense index into the address space).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Addr(pub u64);

/// Identifies a host-supplied arena (id-addressed word sequence).
pub type ArenaId = u32;

/// Base-field ALU operations. `MulAdd` computes `a·b + c` (the Horner step).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseOp {
    Add,
    Sub,
    Mul,
    Div,
    MulAdd,
}

/// Fp3 ALU operations, on word lanes 0–2. `MulAdd` is `a·b + c`; `MulBase`
/// multiplies an extension element by a base element (3 base muls, not 9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtOp {
    Add,
    Sub,
    Mul,
    Div,
    MulAdd,
    MulBase,
}

/// The four hash-chiplet modes. `Compress`: two digest cells → one digest
/// cell. `Transcript`: the same shape in the Fiat–Shamir domain. `Leaf`: one
/// cell of four FIELD ELEMENTS → one digest cell. `Permute`: three state cells
/// → three state cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashMode {
    Compress,
    /// One step of the Fiat–Shamir transcript chain.
    ///
    /// Structurally identical to [`HashMode::Compress`] — two cells in, one
    /// cell out, same socket, same columns — and DIFFERENT in exactly one
    /// thing: the hash domain. Under BLAKE3 the row's domain tag is the
    /// message word immediately after the lanes (`m[12]` at the RATE-4 lane
    /// count), selected by the preprocessed mode columns, so a
    /// transcript step cannot be replayed as a Merkle parent or the reverse.
    /// Hashers with a single domain (`Test`, `Poseidon`) compute the same
    /// function in both modes; the separation is a property of the hasher, not
    /// of the machine.
    Transcript,
    /// A Merkle LEAF over four arbitrary field elements.
    ///
    /// **This mode implies felt-input semantics**, by decision rather than by
    /// inference. The other modes read both their input cells as digests — four
    /// `u32` lanes each; this one reads its FIRST cell that way, as a chaining
    /// accumulator, and its SECOND as four Goldilocks elements, splitting each
    /// into a checked `lo`/`hi` `u32` pair so eight halves fill the message
    /// lanes above the accumulator. That is what lets FRI data — LDE evaluations
    /// and folded extension elements, none of them `u32` — reach a hash whose
    /// inputs must be `u32`, and it absorbs four felts per compression because
    /// the chaining rides in the message rather than in a separate fold.
    ///
    /// It is also what retires obligation O5 — **under a hasher that separates
    /// the domains.** BLAKE3 does: a leaf is `BLAKE3(…‖"LFML")` and a parent is
    /// `BLAKE3(…‖"LFMC")`, so an internal node cannot be replayed as a leaf
    /// whatever the tree's shape, where before that rested on every eDSL circuit
    /// being fixed-depth — true, but enforced by nothing.
    ///
    /// ⚠ The mode is a machine-level shape, not a guarantee. A single-domain
    /// hasher computes the same function in both modes, so under `Test` and
    /// `Poseidon` O5 still rests on fixed depth exactly as it did before. The
    /// separation is a property of the HASHER; see `LfmHasher::leaf_out`.
    Leaf,
    Permute,
}

impl HashMode {
    /// Whether this mode is the two-cells-in, one-cell-out shape — true for
    /// `Compress` and `Transcript`, which differ only in hash domain.
    ///
    /// Every place that used to match `Compress` for a *shape* reason routes
    /// through here, so adding a domain cannot silently take the permute arm.
    pub const fn is_two_to_one(self) -> bool {
        matches!(self, HashMode::Compress | HashMode::Transcript)
    }

    /// Input cells this mode reads from memory: 2 or 3.
    ///
    /// The `LFM_HASH` bus receives are gated by exactly this, so a mode that
    /// reads fewer cells must not receive the ones it does not read — a row
    /// receiving a cell it never reads would claim a memory read it never makes.
    ///
    /// A `Leaf` reads TWO: its chaining accumulator, then the four felts it
    /// absorbs. It read one until the leaf RATE put the accumulator in the
    /// message rather than in a separate `"LFMC"` fold (COMMIT.md §1.2), which
    /// is what took leaf absorption from 2 felts per compression to 4.
    pub const fn num_input_cells(self) -> usize {
        match self {
            HashMode::Compress | HashMode::Transcript | HashMode::Leaf => 2,
            HashMode::Permute => 3,
        }
    }

    /// Output cells this mode writes: 1 for every hashing mode, 3 for a
    /// permutation.
    pub const fn num_output_cells(self) -> usize {
        match self {
            HashMode::Permute => 3,
            _ => 1,
        }
    }
}

/// Operands of a [`Instr::KeccakF`]: 13 words of `u32`-half keccak state in,
/// 13 out, plus each output's static read count.
///
/// The state's 25 `u64` lanes are not felt-representable (values in `[p, 2^64)`
/// exist), so they travel as 50 `u32` halves packed four to a word; the last
/// word's top two lanes are unused and must be zero. The permutation itself is
/// proved by the production `KECCAK_RND` / `KECCAK_RC` / `BITWISE` chips —
/// `LFM_KECCAK` only binds these words to the `Keccak` bus tokens.
#[derive(Debug, Clone)]
pub struct KeccakOperands {
    pub mode: KeccakMode,
    pub ins: [Addr; 13],
    /// Rate-block words, read only in [`KeccakMode::Absorb`]; `Addr(0)`
    /// placeholders otherwise (the receives are gated by the mode selector, so
    /// the placeholders are never read).
    pub block: [Addr; 9],
    pub outs: [Addr; 13],
    pub mults: [u64; 13],
    /// When set, the row ALSO writes the byte-reversed first 32 bytes of the
    /// output state as two words — the production transcript's `sample()`.
    pub rev: Option<KeccakReversedDigest>,
}

/// The reversed-digest outputs of a keccak row (see `layout::keccak::REV_ADDR0`).
#[derive(Debug, Clone)]
pub struct KeccakReversedDigest {
    pub outs: [Addr; 2],
    pub mults: [u64; 2],
}

/// The adapter's two modes.
///
/// `Permute` is the bare permutation (R1b). `Absorb` XORs a 136-byte rate block
/// into the state's rate region first — the sponge step, with the XOR done by
/// `BYTE_ALU[XOR]` lookups into the same BITWISE table the round chip uses.
/// Doing the XOR here rather than on the LFM side (bit-decompose, recombine) is
/// orders of magnitude cheaper: the adapter already owns byte-granular columns
/// and already talks to BITWISE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeccakMode {
    Permute,
    Absorb,
}

/// Operands of an [`Instr::Blake3`]: 7 words of `u32` input in, 4 words of
/// compression output out, plus each output's static read count.
///
/// The 28 input `u32` words are `h[8] ‖ m[16] ‖ t_lo ‖ t_hi ‖ block_len ‖
/// flags`, four to a machine word — so word 0–1 carry `h`, words 2–5 carry `m`,
/// and word 6 is `(t_lo, t_hi, block_len, flags)`. 28 divides by 4 exactly,
/// so unlike [`KeccakOperands`] there are no unused lane slots. Every lane of
/// every input word must be a canonical value below `2^32`.
///
/// Unlike `KeccakF`, the compression is proved **here** — `LFM_BLAKE3` carries
/// its own AIR — rather than delegated to a hosted family, so there is no tag
/// binding a request token to a reply token and nothing for the admission
/// validator to check for uniqueness.
#[derive(Debug, Clone)]
pub struct Blake3Operands {
    pub ins: [Addr; layout::blake3::IN_WORDS],
    pub outs: [Addr; layout::blake3::OUT_WORDS],
    pub mults: [u64; layout::blake3::OUT_WORDS],
    /// When set, the row ALSO writes the byte-reversed 32-byte digest as two
    /// words — the production transcript's `sample()`. Free on the bus (see
    /// `layout::blake3::REV_ADDR0`).
    pub rev: Option<Blake3ReversedDigest>,
}

/// The reversed-digest outputs of a BLAKE3 row (see `layout::blake3::REV_ADDR0`).
#[derive(Debug, Clone)]
pub struct Blake3ReversedDigest {
    pub outs: [Addr; layout::blake3::DIGEST_WORDS],
    pub mults: [u64; layout::blake3::DIGEST_WORDS],
}

/// One LFM instruction. Operand-field conventions:
///
/// - `c` on the ALU ops is meaningful iff the op is `MulAdd` (and is emitted
///   as address 0 otherwise — the corresponding bus receive is gated by the
///   `MulAdd` selector, so the placeholder is never read).
/// - `Hash` uses `ins[..num_input_cells()]` and `outs[..num_output_cells()]`
///   only — 2/1 for `Compress` and `Transcript`, 1/1 for `Leaf`, 3/3 for
///   `Permute`; the remaining slots are `Addr(0)` placeholders with `mults`
///   fixed to 0, and the validator checks both ends.
/// - `BitDec.bits` lists, low-to-high from bit 0, exactly the bit cells the
///   program consumes; bits beyond `bits.len()` exist as constrained witness
///   columns but get no memory cell.
#[derive(Debug, Clone)]
pub enum Instr {
    Const {
        out: Addr,
        value: [FE; 4],
        mult: u64,
    },
    BaseAlu {
        op: BaseOp,
        out: Addr,
        a: Addr,
        b: Addr,
        c: Addr,
        mult: u64,
    },
    ExtAlu {
        op: ExtOp,
        out: Addr,
        a: Addr,
        b: Addr,
        c: Addr,
        mult: u64,
    },
    Select {
        bit: Addr,
        out_l: Addr,
        out_r: Addr,
        in_l: Addr,
        in_r: Addr,
        mult_l: u64,
        mult_r: u64,
    },
    BitDec {
        input: Addr,
        bits: Vec<(Addr, u64)>,
        /// The value's two BIG-ENDIAN `u32` halves as output cells —
        /// `[high-word half, low-word half]`, i.e. what
        /// `append_field_element` puts on the wire. `None` for a plain
        /// decomposition. The halves are linear forms over the bit columns,
        /// so they cost no extra row and no ALU work.
        halves: Option<[(Addr, u64); 2]>,
    },
    Hash {
        mode: HashMode,
        ins: [Addr; 3],
        outs: [Addr; 3],
        mults: [u64; 3],
    },
    Hint {
        arena: ArenaId,
        index: u32,
        out: Addr,
        mult: u64,
    },
    /// Assemble a word from four base cells (unused lanes take the shared
    /// zero-constant cell). The lane↔word coupling is enforced purely by the
    /// `LFM_LANES` chip's bus tokens — no constraints.
    Pack {
        lanes: [Addr; 4],
        out: Addr,
        mult: u64,
    },
    /// Split a word into four base cells — the only way a hash-state or
    /// digest lane can reach the ALU (discovered as a real ISA gap in
    /// Milestone C: challenges are squeezed as cells but consumed as felts).
    Unpack {
        input: Addr,
        outs: [Addr; 4],
        mults: [u64; 4],
    },
    /// One `keccak-f[1600]` permutation over 13 words of `u32`-half state.
    ///
    /// Boxed: the 13-wide operand arrays are 312 bytes, four times the next
    /// largest variant, and inlining them would quadruple every instruction in
    /// the program vector.
    KeccakF(Box<KeccakOperands>),
    /// One BLAKE3 compression over 7 input words, writing 4 output words.
    ///
    /// Boxed for the reason `KeccakF` is: the operand arrays are ~150 bytes,
    /// twice the next largest variant, and inlining them would grow every
    /// instruction in the program vector.
    Blake3(Box<Blake3Operands>),
    Public {
        addr: Addr,
        index: u32,
    },
}

impl Instr {
    /// The addresses this instruction writes, in ascending order.
    pub fn writes(&self) -> Vec<Addr> {
        match self {
            Instr::Const { out, .. }
            | Instr::BaseAlu { out, .. }
            | Instr::ExtAlu { out, .. }
            | Instr::Hint { out, .. }
            | Instr::Pack { out, .. } => vec![*out],
            Instr::Unpack { outs, .. } => outs.to_vec(),
            Instr::KeccakF(k) => {
                let mut v = k.outs.to_vec();
                if let Some(rev) = &k.rev {
                    v.extend_from_slice(&rev.outs);
                }
                v
            }
            Instr::Blake3(k) => {
                let mut v = k.outs.to_vec();
                if let Some(rev) = &k.rev {
                    v.extend_from_slice(&rev.outs);
                }
                v
            }
            Instr::Select { out_l, out_r, .. } => vec![*out_l, *out_r],
            Instr::BitDec { bits, halves, .. } => bits
                .iter()
                .map(|(a, _)| *a)
                .chain(halves.iter().flat_map(|hs| hs.iter().map(|(a, _)| *a)))
                .collect(),
            Instr::Hash { mode, outs, .. } => outs[..mode.num_output_cells()].to_vec(),
            Instr::Public { .. } => vec![],
        }
    }

    /// The addresses this instruction reads (meaningful operands only, per
    /// the field conventions above).
    pub fn reads(&self) -> Vec<Addr> {
        match self {
            Instr::Const { .. } | Instr::Hint { .. } => vec![],
            Instr::BaseAlu { op, a, b, c, .. } => {
                if *op == BaseOp::MulAdd {
                    vec![*a, *b, *c]
                } else {
                    vec![*a, *b]
                }
            }
            Instr::ExtAlu { op, a, b, c, .. } => {
                if *op == ExtOp::MulAdd {
                    vec![*a, *b, *c]
                } else {
                    vec![*a, *b]
                }
            }
            Instr::Select {
                bit, in_l, in_r, ..
            } => vec![*bit, *in_l, *in_r],
            Instr::BitDec { input, .. } => vec![*input],
            Instr::Hash { mode, ins, .. } => ins[..mode.num_input_cells()].to_vec(),
            Instr::Pack { lanes, .. } => lanes.to_vec(),
            Instr::Unpack { input, .. } => vec![*input],
            Instr::KeccakF(k) => match k.mode {
                KeccakMode::Permute => k.ins.to_vec(),
                KeccakMode::Absorb => {
                    let mut v = k.ins.to_vec();
                    v.extend_from_slice(&k.block);
                    v
                }
            },
            // Every input word is read on every row: the chip has one mode, so
            // there is no gated operand and no placeholder to exclude.
            Instr::Blake3(k) => k.ins.to_vec(),
            Instr::Public { addr, .. } => vec![*addr],
        }
    }
}
