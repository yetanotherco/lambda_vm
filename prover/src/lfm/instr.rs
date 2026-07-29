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

/// The two hash-chiplet modes. `Compress`: two digest cells → one digest
/// cell. `Permute`: three state cells → three state cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashMode {
    Compress,
    Permute,
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

/// One LFM instruction. Operand-field conventions:
///
/// - `c` on the ALU ops is meaningful iff the op is `MulAdd` (and is emitted
///   as address 0 otherwise — the corresponding bus receive is gated by the
///   `MulAdd` selector, so the placeholder is never read).
/// - `Hash` in `Compress` mode uses `ins[0..2]` and `outs[0]` only; the
///   remaining slots are `Addr(0)` placeholders with `mults` fixed to 0.
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
            Instr::Select { out_l, out_r, .. } => vec![*out_l, *out_r],
            Instr::BitDec { bits, .. } => bits.iter().map(|(a, _)| *a).collect(),
            Instr::Hash { mode, outs, .. } => match mode {
                HashMode::Compress => vec![outs[0]],
                HashMode::Permute => outs.to_vec(),
            },
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
            Instr::Hash { mode, ins, .. } => match mode {
                HashMode::Compress => vec![ins[0], ins[1]],
                HashMode::Permute => ins.to_vec(),
            },
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
            Instr::Public { addr, .. } => vec![*addr],
        }
    }
}
