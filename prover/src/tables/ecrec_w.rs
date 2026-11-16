//! ECREC_W: the ECREC chip ([`super::ecrec`], one ECDSA public-key recovery a
//! row) under [`MemoryFormat::Words`](super::fused_memory::MemoryFormat::Words),
//! where RAM is checked by aligned 8-byte words ([`super::word_memory`]).
//!
//! ECREC reads the signature `r ‖ s` as eight doublewords at `x11 + 8i` (at
//! `T`), the hash `z` as four at `x12 + 8i` (`T + 1`), and writes the key and
//! the status as nine at `x10 + 8i` (`T + 2`). ECREC_W is ECREC with each of
//! those 21 memory messages a word message on [`BusId::MemWordMsg`] (a read an
//! access with old = new, a write a write), MEMW_W taking them, and with:
//! - every byte it reads or writes as a word checked in the row (`r`, `s`,
//!   `z`, the key's `x` and `y`: 80 `ARE_BYTES` pairs at `MU`), so each word is
//!   the bytes' one decomposition and the words it writes are canonical;
//! - the status a column, `BAIL = MU − OK`, so the status word's low half is
//!   one column a constraint fixes.
//!
//! Every other interaction and constraint is ECREC's, unchanged.
//!
//! ## Alignment
//!
//! The three pointers are 8-aligned: the executor refuses a call that is not
//! (a completeness rule, the guest's wrapper takes the software path on
//! one). Soundness does not rest on it: a word message's key is the pointer
//! plus `8i`, and only PAGE_W mints tokens, at aligned keys, so a call at an
//! unaligned pointer consumes tokens nothing produced and its block does not
//! verify.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseOperation, BitwiseOperationType};
use super::ecrec::{self, EcrecConstraints, EcrecOperation};
use super::gpack::TraceForm;
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use super::word_memory::{access_message, byte_pair_checks, write_message};

/// Column definitions for the ECREC_W table: ECREC's, then `BAIL`.
pub mod cols {
    /// The status, `MU − OK`.
    pub const BAIL: usize = super::ecrec::cols::NUM_COLUMNS;
    pub const NUM_COLUMNS: usize = BAIL + 1;
}

/// The name the AIR set gives an ECREC_W table: `ECREC_W[i]`.
pub const NAME: &str = "ECREC_W";

/// The byte arrays ECREC_W checks in pairs: `r`, `s`, `z`, the key's `x`,
/// `y` (32 little-endian limb bytes each).
const CHECKED: [usize; 5] = [
    ecrec::cols::R,
    ecrec::cols::S,
    ecrec::cols::Z,
    ecrec::cols::XQ,
    ecrec::cols::YQ,
];

/// The ECREC_W trace of `ops`: ECREC's, with `BAIL` set on each row.
pub fn generate_ecrec_w_trace(
    ops: &[EcrecOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let ecrec = ecrec::generate_ecrec_trace_as(ops, TraceForm::Wide);
    let rows = ecrec.num_rows();
    let mut out = TraceTable::new_main(
        vec![FE::zero(); rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    for row in 0..rows {
        for c in 0..ecrec::cols::NUM_COLUMNS {
            out.main_table.set(row, c, *ecrec.get_main(row, c));
        }
    }
    for (row, op) in ops.iter().enumerate() {
        out.main_table.set_bool(row, cols::BAIL, !op.witness.ok);
    }
    out
}

/// The BITWISE lookups ECREC_W sends beyond ECREC's: the byte pairs of `r`,
/// `s`, `z` and the key, on every row (a bail's key is zero).
pub fn extra_bitwise_ops(op: &EcrecOperation) -> impl Iterator<Item = BitwiseOperation> + '_ {
    let w = &op.witness;
    [&w.r, &w.s, &w.z, &w.x_q, &w.y_q]
        .into_iter()
        .flat_map(|bytes| bytes.chunks_exact(2))
        .map(|p| BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, p[0], p[1]))
}

/// A byte value of ECREC's MEMW tuples as linear terms at `coefficient`: a
/// column, the bail form `MU − OK` (as `BAIL`), or a constant.
fn byte_terms(v: &BusValue, coefficient: i64) -> Vec<LinearTerm> {
    match v {
        BusValue::Packed {
            start_column,
            packing: Packing::Direct,
        } => vec![LinearTerm::Column {
            coefficient,
            column: *start_column,
        }],
        BusValue::Linear(terms) => {
            let is_bail = terms.len() == 2
                && terms.iter().any(|t| {
                    matches!(t, LinearTerm::Column { coefficient: 1, column } if *column == ecrec::cols::MU)
                })
                && terms.iter().any(|t| {
                    matches!(t, LinearTerm::Column { coefficient: -1, column } if *column == ecrec::cols::OK)
                });
            if is_bail {
                return vec![LinearTerm::Column {
                    coefficient,
                    column: cols::BAIL,
                }];
            }
            terms
                .iter()
                .map(|t| match *t {
                    LinearTerm::Column {
                        coefficient: c,
                        column,
                    } => LinearTerm::Column {
                        coefficient: c * coefficient,
                        column,
                    },
                    LinearTerm::Constant(c) => LinearTerm::Constant(c * coefficient),
                    LinearTerm::ColumnUnsigned { .. } => {
                        unreachable!("ECREC's memory bytes are signed forms")
                    }
                })
                .collect()
        }
        BusValue::Packed { packing, .. } => {
            unreachable!("ECREC's memory bytes are columns, not {packing:?}")
        }
    }
}

/// Eight byte values as a word's two halves.
fn word_of(bytes: &[BusValue]) -> [BusValue; 2] {
    let half = |range: std::ops::Range<usize>| {
        let mut terms = Vec::new();
        for (k, j) in range.enumerate() {
            terms.extend(byte_terms(&bytes[j], 1 << (8 * k)));
        }
        if terms.is_empty() {
            BusValue::constant(0)
        } else {
            BusValue::linear(terms)
        }
    };
    [half(0..4), half(4..8)]
}

/// ECREC's MEMW tuple as a word message, or `None` for a register access:
/// a read `[old[8], 0, base_lo, base_hi, value[8], ts_lo, ts_hi, 0, 0, 1]`
/// an access `(base, ts_lo, old, old)`; a write `[0, base_lo, base_hi,
/// value[8], ts_lo, ts_hi, 0, 0, 1]` a write `(base, ts_lo, value)`. The
/// timestamp's high word is ECREC's `TIMESTAMP_1`, which the CPU's `Ecall`
/// send fixes at 0.
fn word_message(bi: &BusInteraction) -> Option<Vec<BusValue>> {
    let constant =
        |v: &BusValue, c: i64| matches!(v, BusValue::Linear(t) if t == &[LinearTerm::Constant(c)]);
    match bi.values.len() {
        24 if constant(&bi.values[8], 0) => {
            let word = word_of(&bi.values[0..8]);
            Some(access_message(
                [bi.values[9].clone(), bi.values[10].clone()],
                bi.values[19].clone(),
                word.clone(),
                word,
            ))
        }
        16 if constant(&bi.values[0], 0) => Some(write_message(
            [bi.values[1].clone(), bi.values[2].clone()],
            bi.values[11].clone(),
            word_of(&bi.values[3..11]),
        )),
        _ => None,
    }
}

/// ECREC_W's interactions: ECREC's, each memory message a word message, then
/// 80 `ARE_BYTES` pairs at `MU`.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut out: Vec<BusInteraction> = ecrec::bus_interactions()
        .into_iter()
        .map(|bi| {
            if bi.bus_id != u64::from(BusId::Memw) {
                return bi;
            }
            match word_message(&bi) {
                Some(values) => {
                    BusInteraction::sender(BusId::MemWordMsg, bi.multiplicity.clone(), values)
                }
                None => bi,
            }
        })
        .collect();
    for start in CHECKED {
        let bytes: Vec<usize> = (start..start + 32).collect();
        out.extend(byte_pair_checks(
            &bytes,
            Multiplicity::Column(ecrec::cols::MU),
        ));
    }
    out
}

/// ECREC's constraints, then `BAIL = MU − OK`.
#[derive(Clone, Copy)]
pub struct EcrecWordConstraints;

/// How many constraints [`EcrecWordConstraints`] emits.
pub const NUM_CONSTRAINTS: usize = ecrec::NUM_CONSTRAINTS + 1;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for EcrecWordConstraints {
    fn max_degree(&self) -> usize {
        EcrecConstraints.max_degree()
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        EcrecConstraints.eval(b);
        let e = b.main(0, cols::BAIL) - (b.main(0, ecrec::cols::MU) - b.main(0, ecrec::cols::OK));
        b.emit_base(ecrec::NUM_CONSTRAINTS, e);
    }
}
