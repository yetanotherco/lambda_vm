//! LOCAL_BYTES: the fixed byte table a group's local lookups are argued
//! against, one in every group that holds a local sender, as the group's last
//! table.
//!
//! `2^16` rows, row `r` the byte pair `(X, Y) = (r mod 256, r div 256)`. Its
//! four leading columns are preprocessed — `X`, `Y`, `X ∧ Y`, `X ⊕ Y`, checked
//! by the verifier against [`preprocessed_columns`] — and three are committed
//! multiplicities: `MU_ARE_BYTES`, `MU_AND`, `MU_XOR`. It receives
//!
//! - `LocalAreBytes[X, Y]·MU_ARE_BYTES`,
//! - `LocalByteAlu[AND, X, Y, X ∧ Y]·MU_AND`,
//! - `LocalByteAlu[XOR, X, Y, X ⊕ Y]·MU_XOR`,
//!
//! the tuples KECCAK_RND sends under `KeccakRndForm::LeanLocal`. It has no
//! constraint: the multiplicities may be anything, and a receiver whose values
//! are all fixed columns balances only lookups of its own rows.
//!
//! Its multiplicities are counted on the host from each sender table's data
//! ([`local_counts_of`]): from a KECCAK_RND table's records by replaying its
//! rounds, or from any table's columns through its interactions
//! ([`count_from_columns`], the reference).

use math::field::element::FieldElement;
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::multilinear_table::CommittedTable;
use stark::trace::TraceTable;

use crate::tables::bitwise::{BitwiseOperation, BitwiseOperationType};
use crate::tables::keccak_rnd::KeccakRndForm;
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, alu_op};

/// The table's height, in variables.
pub const LOG_ROWS: usize = 16;
/// The table's rows: every byte pair.
pub const NUM_ROWS: usize = 1 << LOG_ROWS;
/// The preprocessed columns: `X`, `Y`, `X ∧ Y`, `X ⊕ Y`.
pub const NUM_PRECOMPUTED_COLS: usize = 4;

/// The format's local buses: every lookup on one of these is argued under
/// its group's own challenges, and only LOCAL_BYTES receives them.
pub const LOCAL_BUSES: [u64; 2] = [BusId::LocalAreBytes as u64, BusId::LocalByteAlu as u64];

pub mod cols {
    pub const X: usize = 0;
    pub const Y: usize = 1;
    pub const AND: usize = 2;
    pub const XOR: usize = 3;
    pub const MU_ARE_BYTES: usize = 4;
    pub const MU_AND: usize = 5;
    pub const MU_XOR: usize = 6;
    pub const NUM_COLUMNS: usize = 7;
}

/// The row of the byte pair `(x, y)`.
pub const fn row_of(x: u8, y: u8) -> usize {
    x as usize + 256 * y as usize
}

fn direct(column: usize) -> BusValue {
    BusValue::Packed {
        start_column: column,
        packing: Packing::Direct,
    }
}

/// The table's three receivers.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        BusInteraction::receiver(
            BusId::LocalAreBytes,
            Multiplicity::Column(cols::MU_ARE_BYTES),
            vec![direct(cols::X), direct(cols::Y)],
        ),
        BusInteraction::receiver(
            BusId::LocalByteAlu,
            Multiplicity::Column(cols::MU_AND),
            vec![
                BusValue::constant(alu_op::AND as u64),
                direct(cols::X),
                direct(cols::Y),
                direct(cols::AND),
            ],
        ),
        BusInteraction::receiver(
            BusId::LocalByteAlu,
            Multiplicity::Column(cols::MU_XOR),
            vec![
                BusValue::constant(alu_op::XOR as u64),
                direct(cols::X),
                direct(cols::Y),
                direct(cols::XOR),
            ],
        ),
    ]
}

/// The four preprocessed columns, by row.
pub fn preprocessed_columns() -> Vec<Vec<FE>> {
    let mut columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .map(|_| Vec::with_capacity(NUM_ROWS))
        .collect();
    for r in 0..NUM_ROWS {
        let (x, y) = ((r % 256) as u64, (r / 256) as u64);
        columns[cols::X].push(FE::from(x));
        columns[cols::Y].push(FE::from(y));
        columns[cols::AND].push(FE::from(x & y));
        columns[cols::XOR].push(FE::from(x ^ y));
    }
    columns
}

/// A group's local multiplicities: how many times each LOCAL_BYTES row is
/// looked up by each receiver's tuple, summed over the group's senders.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalCounts {
    pub are_bytes: Vec<u32>,
    pub and: Vec<u32>,
    pub xor: Vec<u32>,
}

impl Default for LocalCounts {
    fn default() -> Self {
        Self {
            are_bytes: vec![0; NUM_ROWS],
            and: vec![0; NUM_ROWS],
            xor: vec![0; NUM_ROWS],
        }
    }
}

impl LocalCounts {
    /// Another sender's counts added in.
    pub fn add(&mut self, other: &Self) -> Result<(), String> {
        for (mine, theirs) in [
            (&mut self.are_bytes, &other.are_bytes),
            (&mut self.and, &other.and),
            (&mut self.xor, &other.xor),
        ] {
            for (m, t) in mine.iter_mut().zip(theirs) {
                *m = m
                    .checked_add(*t)
                    .ok_or("a LOCAL_BYTES multiplicity overflows u32")?;
            }
        }
        Ok(())
    }

    /// `n` lookups of `op` (an `AreBytes`, `ByteAluAnd` or `ByteAluXor`
    /// operation: the only ones a local sender sends).
    pub fn bump_n(&mut self, op: BitwiseOperation, n: u32) -> Result<(), String> {
        let row = row_of(op.x, op.y);
        let column = match op.lookup_type {
            BitwiseOperationType::AreBytes => &mut self.are_bytes,
            BitwiseOperationType::ByteAluAnd => &mut self.and,
            BitwiseOperationType::ByteAluXor => &mut self.xor,
            other => return Err(format!("{other:?} is not a local lookup")),
        };
        column[row] = column[row]
            .checked_add(n)
            .ok_or("a LOCAL_BYTES multiplicity overflows u32")?;
        Ok(())
    }

    /// Whether nothing was counted.
    pub fn is_empty(&self) -> bool {
        [&self.are_bytes, &self.and, &self.xor]
            .iter()
            .all(|column| column.iter().all(|&m| m == 0))
    }

    /// The LOCAL_BYTES table over these counts: its preprocessed columns, then
    /// the three multiplicities.
    pub fn trace(&self) -> TraceTable<GoldilocksField, GoldilocksExtension> {
        let mut columns = preprocessed_columns();
        for column in [&self.are_bytes, &self.and, &self.xor] {
            columns.push(column.iter().map(|&m| FE::from(u64::from(m))).collect());
        }
        TraceTable::from_columns_main(columns, 1)
    }
}

/// KECCAK_RND's rows `skip..skip + rows` of the table `inputs` make (each
/// input a permutation's state, 24 rows a permutation, rows past the last
/// permutation padding) counted under `form`'s local tuples: each real row's
/// round lookups replayed, and each padding row's 200 `AreBytes[0, 0]`.
pub fn count_keccak_rnd_rows<'i>(
    inputs: impl IntoIterator<Item = &'i [u64; 25]>,
    skip: usize,
    rows: usize,
    form: KeccakRndForm,
    counts: &mut LocalCounts,
) -> Result<(), String> {
    if !form.byte_lookups_local() {
        return Err(format!("KECCAK_RND's {form:?} sends no local lookup"));
    }
    let end = skip + rows;
    let mut permutations = 0usize;
    let mut failed: Option<String> = None;
    for (k, input) in inputs.into_iter().enumerate() {
        permutations = k + 1;
        let first = k * 24;
        if first >= end || first + 24 <= skip {
            continue;
        }
        let rounds = skip.max(first) - first..end.min(first + 24) - first;
        crate::tables::trace_builder::emit_keccak_rnd_rounds(input, form, rounds, &mut |op| {
            if failed.is_none()
                && let Err(e) = counts.bump_n(op, 1)
            {
                failed = Some(e);
            }
        });
    }
    if let Some(e) = failed {
        return Err(e);
    }
    let real_end = (permutations * 24).clamp(skip, end);
    let padding = end - real_end;
    let per_row = form.are_bytes_per_row();
    let zero = BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, 0, 0);
    let n = u32::try_from(padding * per_row).map_err(|_| "too many padding rows".to_string())?;
    counts.bump_n(zero, n)
}

/// THE REFERENCE COUNT: a table of `rows` rows whose main column `c` at row
/// `r` is `get(r, c)`, its local SENDS counted through their definitions:
/// each send's multiplicity and tuple evaluated row by row, and the tuple's
/// LOCAL_BYTES row bumped by the multiplicity. A tuple no row holds (a false
/// lookup) is counted nowhere, so its group's local sum is refused.
pub fn count_from_columns(
    interactions: &[BusInteraction],
    rows: usize,
    get: impl Fn(usize, usize) -> u64,
    counts: &mut LocalCounts,
) -> Result<(), String> {
    let sends: Vec<&BusInteraction> = interactions
        .iter()
        .filter(|i| i.is_sender && LOCAL_BUSES.contains(&i.bus_id))
        .collect();
    let byte = |v: &FieldElement<GoldilocksField>| u8::try_from(v.canonical_u64()).ok();
    for row in 0..rows {
        let column = |c: usize| FE::from(get(row, c));
        for send in &sends {
            let m = send.multiplicity.evaluate_with(column).canonical_u64();
            if m == 0 {
                continue;
            }
            let m = u32::try_from(m).map_err(|_| format!("row {row}: multiplicity {m}"))?;
            let tuple: Vec<FE> = send
                .values
                .iter()
                .flat_map(|value| value.combine_from(column))
                .collect();
            let op = if send.bus_id == BusId::LocalAreBytes as u64 {
                match tuple.as_slice() {
                    [x, y] => byte(x).zip(byte(y)).map(|(x, y)| {
                        BitwiseOperation::byte_op(BitwiseOperationType::AreBytes, x, y)
                    }),
                    _ => return Err(format!("a LocalAreBytes tuple of {}", tuple.len())),
                }
            } else {
                match tuple.as_slice() {
                    [op, x, y, z] => byte(x).zip(byte(y)).zip(byte(z)).and_then(|((x, y), z)| {
                        let op = op.canonical_u64();
                        if op == u64::from(alu_op::AND) && z == x & y {
                            Some(BitwiseOperation::byte_op(
                                BitwiseOperationType::ByteAluAnd,
                                x,
                                y,
                            ))
                        } else if op == u64::from(alu_op::XOR) && z == x ^ y {
                            Some(BitwiseOperation::byte_op(
                                BitwiseOperationType::ByteAluXor,
                                x,
                                y,
                            ))
                        } else {
                            None
                        }
                    }),
                    _ => return Err(format!("a LocalByteAlu tuple of {}", tuple.len())),
                }
            };
            if let Some(op) = op {
                counts.bump_n(op, m)?;
            }
        }
    }
    Ok(())
}

/// Whether a table sends a local lookup.
pub fn sends_local(interactions: &[BusInteraction]) -> bool {
    interactions
        .iter()
        .any(|i| i.is_sender && LOCAL_BUSES.contains(&i.bus_id))
}

/// A laid-out table's local sends counted, wherever its data is: a
/// KECCAK_RND table held as its records by replaying its rounds (`form`'s),
/// any other from its packed or its host columns through its interactions
/// ([`count_from_columns`]). `None` for a table with no local send.
pub fn local_counts_of(
    table: &CommittedTable<'_, GoldilocksField, GoldilocksExtension>,
    form: KeccakRndForm,
) -> Result<Option<LocalCounts>, String> {
    let interactions = table.statement().interactions;
    if !sends_local(interactions) {
        return Ok(None);
    }
    let mut counts = LocalCounts::default();
    let rows = 1usize << table.num_vars();
    if let Some(records) = table.records() {
        let multilinear::records::RecordsFill::KeccakRnd { ops, skip } = records.0.fill() else {
            return Err("a local sender held as records other than KECCAK_RND's".to_string());
        };
        count_keccak_rnd_rows(
            ops.iter().map(|op| &op.input),
            skip,
            rows,
            form,
            &mut counts,
        )?;
    } else if let Some(narrow) = table.narrow() {
        count_from_columns(interactions, rows, |r, c| narrow.get(r, c), &mut counts)?;
    } else {
        let columns = table.columns();
        if columns.is_empty() {
            return Err("a local sender with no host columns to count".to_string());
        }
        count_from_columns(
            interactions,
            rows,
            |r, c| columns[c].evals()[r].canonical_u64(),
            &mut counts,
        )?;
    }
    Ok(Some(counts))
}

/// Read a replica's local lookup sources in their builder representation.
#[cfg(feature = "parallel")]
pub(crate) fn local_counts_from_trace(
    trace: &stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>,
    interactions: &[BusInteraction],
    form: KeccakRndForm,
) -> Result<Option<LocalCounts>, String> {
    let rows = trace.main_table.height;
    let mut counts = LocalCounts::default();
    if let Some(records) = trace.records_main() {
        let multilinear::records::RecordsFill::KeccakRnd { ops, skip } = records.fill() else {
            return Err("a local sender held as records other than KECCAK_RND's".to_string());
        };
        count_keccak_rnd_rows(
            ops.iter().map(|op| &op.input),
            skip,
            rows,
            form,
            &mut counts,
        )?;
    } else if let Some(narrow) = trace.narrow_main() {
        count_from_columns(interactions, rows, |r, c| narrow.get(r, c), &mut counts)?;
    } else {
        count_from_columns(
            interactions,
            rows,
            |r, c| trace.main_table.get_row(r)[c].canonical_u64(),
            &mut counts,
        )?;
    }
    Ok(Some(counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::keccak_rnd::{self, KeccakRoundOperation};

    /// Permutations with distinct, scrambled states.
    fn ops(n: usize) -> Vec<KeccakRoundOperation> {
        (0..n as u64)
            .map(|k| {
                let mut input = [0u64; 25];
                for (i, lane) in input.iter_mut().enumerate() {
                    *lane = (k + 1)
                        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                        .rotate_left(i as u32 * 7)
                        ^ (i as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
                }
                KeccakRoundOperation {
                    timestamp: 1000 + 8 * k,
                    input,
                    output: [0; 25],
                }
            })
            .collect()
    }

    /// The reference count of KECCAK_RND's rows `skip..skip + rows` built on
    /// the host, through the local form's interactions.
    fn from_columns(ops: &[KeccakRoundOperation], skip: usize, rows: usize) -> LocalCounts {
        let trace = keccak_rnd::generate_keccak_rnd_rows(ops, skip, rows);
        let interactions = keccak_rnd::bus_interactions_for(KeccakRndForm::LeanLocal);
        let mut counts = LocalCounts::default();
        count_from_columns(
            &interactions,
            rows,
            |r, c| trace.main_table.get(r, c).canonical_u64(),
            &mut counts,
        )
        .unwrap();
        counts
    }

    fn from_records(ops: &[KeccakRoundOperation], skip: usize, rows: usize) -> LocalCounts {
        let mut counts = LocalCounts::default();
        count_keccak_rnd_rows(
            ops.iter().map(|op| &op.input),
            skip,
            rows,
            KeccakRndForm::LeanLocal,
            &mut counts,
        )
        .unwrap();
        counts
    }

    /// The local form is the lean one with its 1,008 byte lookups moved to the
    /// local buses, in place: nothing else changes, and no byte lookup is
    /// left on BITWISE's buses.
    #[test]
    fn the_local_form_moves_every_byte_lookup_and_nothing_else() {
        let lean = keccak_rnd::bus_interactions_for(KeccakRndForm::Lean);
        let local = keccak_rnd::bus_interactions_for(KeccakRndForm::LeanLocal);
        assert_eq!(lean.len(), local.len());
        let mut moved = 0;
        for (l, m) in lean.iter().zip(&local) {
            assert_eq!(l.is_sender, m.is_sender);
            assert_eq!(l.multiplicity, m.multiplicity);
            assert_eq!(l.values, m.values);
            let want = if l.bus_id == BusId::AreBytes as u64 {
                BusId::LocalAreBytes as u64
            } else if l.bus_id == BusId::ByteAlu as u64 {
                BusId::LocalByteAlu as u64
            } else {
                l.bus_id
            };
            assert_eq!(m.bus_id, want);
            moved += usize::from(want != l.bus_id);
        }
        assert_eq!(moved, 1008);
        assert!(
            local
                .iter()
                .all(|i| i.bus_id != BusId::AreBytes as u64 && i.bus_id != BusId::ByteAlu as u64)
        );
        assert!(sends_local(&local));
        assert!(!sends_local(&lean));
    }

    /// The counts from a table's records (its rounds replayed) are the
    /// counts from its columns through its interactions: a whole table, a
    /// chunk that starts and ends inside permutations, and a last chunk past
    /// the last permutation (its padding rows' `AreBytes[0, 0]`). The chunks
    /// add up to the whole table.
    #[test]
    fn local_counts_from_records_equal_the_tables_columns() {
        let ops = ops(5);
        // 120 real rows: a 128-row table, then chunks of 32 rows.
        for (skip, rows) in [(0, 128), (0, 32), (32, 32), (40, 32), (96, 32), (120, 8)] {
            assert_eq!(
                from_records(&ops, skip, rows),
                from_columns(&ops, skip, rows),
                "rows {skip}..{}",
                skip + rows
            );
        }
        let mut chunks = LocalCounts::default();
        for skip in (0..128).step_by(32) {
            chunks.add(&from_records(&ops, skip, 32)).unwrap();
        }
        assert_eq!(chunks, from_records(&ops, 0, 128));
        // A padding row sends 200 AreBytes[0, 0] and nothing else.
        let padding = from_records(&ops, 120, 8);
        assert_eq!(padding.are_bytes[row_of(0, 0)], 8 * 200);
        assert_eq!(
            padding.are_bytes.iter().map(|&m| u64::from(m)).sum::<u64>(),
            1600
        );
        assert!(padding.and.iter().chain(&padding.xor).all(|&m| m == 0));
        // A real row sends its 1,008 lookups.
        let one = from_records(&ops, 0, 1);
        let total: u64 = [&one.are_bytes, &one.and, &one.xor]
            .iter()
            .flat_map(|c| c.iter())
            .map(|&m| u64::from(m))
            .sum();
        assert_eq!(total, 1008);
    }

    /// A form whose byte lookups are not local has nothing to count.
    #[test]
    fn only_the_local_form_counts() {
        let ops = ops(1);
        let mut counts = LocalCounts::default();
        assert!(
            count_keccak_rnd_rows(
                ops.iter().map(|op| &op.input),
                0,
                24,
                KeccakRndForm::Lean,
                &mut counts
            )
            .is_err()
        );
    }

    /// A false tuple is counted nowhere: the XOR result of a sent tuple moved.
    #[test]
    fn a_false_tuple_is_counted_nowhere() {
        let send = vec![BusInteraction::sender(
            BusId::LocalByteAlu,
            Multiplicity::One,
            vec![
                BusValue::constant(alu_op::XOR as u64),
                direct(0),
                direct(1),
                direct(2),
            ],
        )];
        let rows = [[3u64, 5, 6], [3, 5, 7]];
        let mut counts = LocalCounts::default();
        count_from_columns(&send, 2, |r, c| rows[r][c], &mut counts).unwrap();
        assert_eq!(counts.xor[row_of(3, 5)], 1, "the true tuple, once");
        assert_eq!(counts.xor.iter().map(|&m| u64::from(m)).sum::<u64>(), 1);
    }

    /// The fixed columns: row `r` is `(r mod 256, r div 256)` and its AND and
    /// XOR; the receivers read only them (and their multiplicities).
    #[test]
    fn the_fixed_columns_are_every_byte_pair_and_its_and_and_xor() {
        let columns = preprocessed_columns();
        assert_eq!(columns.len(), NUM_PRECOMPUTED_COLS);
        for r in [0usize, 1, 255, 256, 0x1234, NUM_ROWS - 1] {
            let (x, y) = ((r % 256) as u64, (r / 256) as u64);
            assert_eq!(columns[cols::X][r], FE::from(x));
            assert_eq!(columns[cols::Y][r], FE::from(y));
            assert_eq!(columns[cols::AND][r], FE::from(x & y));
            assert_eq!(columns[cols::XOR][r], FE::from(x ^ y));
            assert_eq!(row_of(x as u8, y as u8), r);
        }
        for receiver in bus_interactions() {
            assert!(!receiver.is_sender);
            assert!(LOCAL_BUSES.contains(&receiver.bus_id));
            let mut read = receiver.columns_read();
            if let Multiplicity::Column(m) = receiver.multiplicity {
                read.retain(|&c| c != m);
            }
            assert!(read.iter().all(|&c| c < NUM_PRECOMPUTED_COLS));
        }
    }
}
