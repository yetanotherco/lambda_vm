//! Phase B's overlap ([`PhaseBOverlap`]) on a small block of the multi-table
//! example: each group's openings run beside the next group's argue and the
//! proof is the one phase B proves one group at a time, byte for byte, in any
//! order and under either argue format; the observer hears the same groups,
//! in the same order, with the same shares; a failure on either thread is the
//! prove's error, never a panic; a refusal turns the rest of phase B serial.

use std::sync::Mutex;

use math::field::{
    element::FieldElement, extensions_goldilocks::Degree3GoldilocksExtensionField as Ext,
    goldilocks::GoldilocksField as Fp,
};
use multilinear::{
    Error as MlError,
    mle::Mle,
    stacked_eval::StackedCommitment,
    whir::Domain,
    whir_chain::{ArgueFormat, ChainConfig, ChainFormat, GrindBits},
    whir_commit::TreeDrop,
    whir_hash::{KeccakWhir, WhirHash},
};

use crate::constraints::builder::EmptyConstraints;
use crate::examples::multi_table_lookup::{
    new_add_air_with_lookup, new_cpu_air_with_lookup, new_mul_air_with_lookup,
};
use crate::lookup::{AirWithBuses, NullBoundaryConstraintBuilder};
use crate::multilinear_air::Uniforms;
use crate::multilinear_block::{
    self, ArgueDeviation, BlockCommitted, BlockPrepared, BlockPreparedCheck, BlockProved,
    GroupOpened, GroupStamps, Narrowing, OverlapFaults, PhaseBOverlap,
};
use crate::multilinear_table::{
    CommittedTable, TableLayout, TableStatement, global_layout, global_layouts,
};
use crate::proof::options::ProofOptions;
use crate::traits::AIR;

type FE = FieldElement<Fp>;
type Air = AirWithBuses<Fp, Ext, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
type H = KeccakWhir;

/// The example's three tables (CPU, ADD, MUL), this many times over: the bus
/// balances in each copy, so it balances over the block.
const COPIES: usize = 3;

/// Tables a group: CPU | ADD MUL | CPU | ADD | MUL | CPU ADD | MUL.
const SIZES: [usize; 7] = [1, 2, 1, 1, 1, 2, 1];

/// The prepared stacks: a group and its tables' prefixes, `(table, columns)`.
const PREPARED: [(usize, &[(usize, usize)]); 2] = [(1, &[(1, 1), (2, 2)]), (5, &[(6, 2), (7, 1)])];

fn config(argue: ArgueFormat) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
        format: ChainFormat {
            argue,
            ..ChainFormat::DEFAULT
        },
    }
}

fn airs() -> [Air; 3] {
    let options = ProofOptions::default_test_options();
    [
        new_cpu_air_with_lookup(&options),
        new_add_air_with_lookup(&options),
        new_mul_air_with_lookup(&options),
    ]
}

fn base(values: &[u64]) -> Vec<FE> {
    values.iter().map(|v| FE::from(*v)).collect()
}

/// CPU (8 rows), ADD and MUL (4 rows each), as the multi-table example has
/// them.
fn columns() -> [Vec<Vec<FE>>; 3] {
    [
        vec![
            base(&[1, 0, 1, 0, 1, 1, 0, 0]),
            base(&[0, 1, 0, 1, 0, 0, 1, 1]),
            base(&[1, 2, 3, 4, 5, 6, 7, 8]),
            base(&[10, 20, 30, 40, 50, 60, 70, 80]),
            base(&[11, 40, 33, 160, 55, 66, 490, 640]),
        ],
        vec![
            base(&[1, 3, 5, 6]),
            base(&[10, 30, 50, 60]),
            base(&[11, 33, 55, 66]),
            base(&[1, 1, 1, 1]),
        ],
        vec![
            base(&[2, 4, 7, 8]),
            base(&[20, 40, 70, 80]),
            base(&[40, 160, 490, 640]),
            base(&[1, 1, 1, 1]),
        ],
    ]
}

/// Table `t` of the block: which of the three.
fn kind(t: usize) -> usize {
    t % 3
}

/// Table `t`'s height in variables.
fn num_vars(t: usize) -> usize {
    columns()[kind(t)][0].len().trailing_zeros() as usize
}

fn tables(airs: &[Air; 3]) -> Vec<CommittedTable<'_, Fp, Ext>> {
    let columns = columns();
    (0..3 * COPIES)
        .map(|t| {
            let (air, cols) = (&airs[kind(t)], &columns[kind(t)]);
            CommittedTable::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                cols.len(),
                cols[0].len().trailing_zeros() as usize,
                Uniforms::default(),
                |col| cols[col as usize].clone(),
            )
            .expect("a table")
        })
        .collect()
}

/// Each prepared stack's columns: its tables' prefixes, in its order.
fn prepared_columns() -> Vec<Vec<Mle<Fp>>> {
    let columns = columns();
    PREPARED
        .iter()
        .map(|(_, tables)| {
            tables
                .iter()
                .flat_map(|&(t, n)| columns[kind(t)][..n].iter())
                .map(|values| Mle::new(values.clone()).expect("a column"))
                .collect()
        })
        .collect()
}

/// Each prepared stack committed over its columns.
fn prepared_commitments(
    columns: &[Vec<Mle<Fp>>],
    config: &ChainConfig,
) -> Vec<StackedCommitment<Fp, H>> {
    PREPARED
        .iter()
        .zip(columns)
        .map(|((_, tables), columns)| {
            let shapes: Vec<(usize, usize)> =
                tables.iter().map(|&(t, n)| (n, num_vars(t))).collect();
            let layout = global_layout(&shapes, config.format.stack).expect("a layout");
            StackedCommitment::<Fp, H>::commit(
                layout,
                &multilinear::stacking::borrow(columns),
                None,
                config,
            )
            .expect("a prepared commitment")
        })
        .collect()
}

/// One group's share as the observer was handed it, serialized.
type Heard = (usize, Vec<u8>);

/// The block proved under `overlap`, its groups taken in `order`, with a
/// test's `faults`: the proof and every group's stamps, or the prove's error,
/// and what the observer heard.
fn prove(
    argue: ArgueFormat,
    overlap: PhaseBOverlap,
    order: &[usize],
    faults: OverlapFaults,
) -> (Result<BlockProved<Fp, Ext>, MlError>, Vec<Heard>) {
    let airs = airs();
    let config = config(argue);
    let columns = prepared_columns();
    let commitments = prepared_commitments(&columns, &config);
    let borrowed: Vec<Vec<&Mle<Fp>>> = columns
        .iter()
        .map(|c| multilinear::stacking::borrow(c))
        .collect();
    let prepared: Vec<BlockPrepared<'_, Fp, H>> = PREPARED
        .iter()
        .zip(&commitments)
        .zip(&borrowed)
        .map(|(((group, tables), commitment), columns)| BlockPrepared {
            group: *group,
            tables: tables.to_vec(),
            commitment,
            columns,
        })
        .collect();
    let block = BlockCommitted::commit::<H>(
        tables(&airs),
        &SIZES,
        &config,
        TreeDrop::uniform(1),
        Narrowing::Wide,
        false,
    )
    .expect("phase A")
    .with_phase_b_overlap(overlap)
    .with_overlap_faults(faults);
    let heard: Mutex<Vec<Heard>> = Mutex::new(Vec::new());
    let on_group = |opened: GroupOpened<'_, Fp, Ext>| {
        let share = bincode::serialize(&(
            opened.roots,
            opened.argue,
            opened.tables,
            opened.opening,
            opened.prepared,
        ))
        .expect("serialize");
        heard
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((opened.group, share));
    };
    let mut transcript = H::sponge();
    let proved = multilinear_block::block_prove_in_order::<Fp, Ext, _, H>(
        block,
        &config,
        &mut transcript,
        &prepared,
        &[],
        &|g| g,
        &ArgueDeviation::default(),
        &on_group,
        order,
    );
    (
        proved,
        heard.into_inner().unwrap_or_else(|e| e.into_inner()),
    )
}

fn bytes(proved: &BlockProved<Fp, Ext>) -> Vec<u8> {
    bincode::serialize(&(&proved.0, &proved.1, &proved.2)).expect("serialize")
}

/// The verifier's reading of `proved`, against the shapes and the split it
/// rebuilds itself and the prepared stacks' roots.
fn verify(argue: ArgueFormat, proved: &BlockProved<Fp, Ext>) -> Result<(), MlError> {
    let airs = airs();
    let config = config(argue);
    let columns = columns();
    let prepared_at = |t: usize| {
        PREPARED
            .iter()
            .flat_map(|(_, tables)| tables.iter())
            .find(|&&(table, _)| table == t)
            .map(|&(_, n)| n)
    };
    let layouts: Vec<TableLayout<'_, Fp, Ext>> = (0..3 * COPIES)
        .map(|t| {
            let (air, cols) = (&airs[kind(t)], &columns[kind(t)]);
            TableLayout::new(
                air.constraint_program(),
                air.constraints_meta(),
                air.bus_interactions(),
                cols.len(),
                cols[0].len().trailing_zeros() as usize,
                Uniforms::default(),
            )
            .expect("a layout")
        })
        .collect();
    let statements: Vec<TableStatement<'_, Fp, Ext>> = layouts
        .iter()
        .enumerate()
        .map(|(t, layout)| match prepared_at(t) {
            Some(n) => layout.statement_with_prepared_prefix(n),
            None => layout.statement(),
        })
        .collect();
    let shapes: Vec<(usize, usize)> = (0..3 * COPIES)
        .map(|t| {
            let cols = &columns[kind(t)];
            (cols.len(), cols[0].len().trailing_zeros() as usize)
        })
        .collect();
    let stacks = global_layouts(&shapes, &SIZES, config.format.stack).expect("the stacks");
    let domains: Vec<Domain<Fp>> = stacks
        .iter()
        .map(|layout| Domain::new(layout.n_stack() + config.log_blowup).expect("a domain"))
        .collect();
    let prepared = prepared_columns();
    let commitments = prepared_commitments(&prepared, &config);
    let roots: Vec<Vec<multilinear::whir_commit::Commitment>> =
        commitments.iter().map(StackedCommitment::roots).collect();
    let checks: Vec<BlockPreparedCheck<'_, Fp>> = PREPARED
        .iter()
        .zip(&commitments)
        .zip(&roots)
        .map(
            |(((group, tables), commitment), roots)| BlockPreparedCheck {
                group: *group,
                tables: tables.to_vec(),
                roots,
                layout: commitment.layout(),
                domain: commitment.domain(),
            },
        )
        .collect();
    let mut transcript = H::sponge();
    multilinear_block::block_verify_with::<Fp, Ext, _, H>(
        &proved.0,
        &proved.1,
        &proved.2,
        &checks,
        &statements,
        &stacks,
        &domains,
        &SIZES,
        &FieldElement::<Ext>::zero(),
        &config,
        &mut transcript,
        false,
        crate::multilinear_table::batched::VerifierChecks::ALL,
    )
}

fn in_order() -> Vec<usize> {
    (0..SIZES.len()).collect()
}

const ARGUES: [ArgueFormat; 2] = [ArgueFormat::PerTable, ArgueFormat::BATCHED];

/// The block's proof in group order under each of [`ARGUES`], SHA3-256 of its
/// serialized bytes, as phase B proved it before the overlap existed (one group
/// at a time, at 76a950dc2): one group at a time still proves those bytes.
const BEFORE_THE_OVERLAP: [&str; 2] = [
    "8914fbae3c035670b2163b7ff093bd05991fd3adaf17d558c05246330cca41fe",
    "8dc80a06680a58b72e207af111b9a2c96563c326a4712a96658bff70829c6dc3",
];

fn digest(proved: &BlockProved<Fp, Ext>) -> String {
    use sha3::{Digest, Sha3_256};
    Sha3_256::digest(bytes(proved))
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn stamps(proved: &BlockProved<Fp, Ext>) -> &[GroupStamps] {
    &proved.3
}

/// ★ The overlap moves no byte: under either argue format, with each group's
/// openings beside the next group's argue, the groups taken in order,
/// reversed, or odd ones first, and with the openings' read-backs through
/// pinned staging ([`PhaseBOverlap::Pinned`]), the proof is the one phase B
/// proves one group at a time in group order — the bytes it proved before the
/// overlap existed ([`BEFORE_THE_OVERLAP`]) — and it verifies. Every group but the last one
/// taken opened beside an argue; the last opened alone.
#[test]
fn the_overlap_proves_the_bytes_of_one_group_at_a_time() {
    let n = SIZES.len();
    let reversed: Vec<usize> = (0..n).rev().collect();
    let odd_first: Vec<usize> = (1..n).step_by(2).chain((0..n).step_by(2)).collect();
    for (argue, before) in ARGUES.into_iter().zip(BEFORE_THE_OVERLAP) {
        let (serial, _) = prove(
            argue,
            PhaseBOverlap::Off,
            &in_order(),
            OverlapFaults::default(),
        );
        let serial = serial.expect("one group at a time");
        assert_eq!(digest(&serial), before, "{argue:?}: one group at a time");
        verify(argue, &serial).expect("one group at a time verifies");
        for (what, order, overlap) in [
            ("in order", in_order(), PhaseBOverlap::On),
            ("reversed", reversed.clone(), PhaseBOverlap::On),
            ("odd first", odd_first.clone(), PhaseBOverlap::On),
            ("in order, pinned", in_order(), PhaseBOverlap::Pinned),
            ("reversed, pinned", reversed.clone(), PhaseBOverlap::Pinned),
        ] {
            let (proved, _) = prove(argue, overlap, &order, OverlapFaults::default());
            let proved = proved.unwrap_or_else(|e| panic!("{argue:?} {what}: {e:?}"));
            assert_eq!(
                bytes(&proved),
                bytes(&serial),
                "{argue:?} {what}: the proof"
            );
            verify(argue, &proved).unwrap_or_else(|e| panic!("{argue:?} {what}: {e:?}"));
            let s = stamps(&proved);
            let last = *order.last().expect("a group");
            for &g in &order {
                assert_eq!(s[g].overlapped, g != last, "{argue:?} {what}: group {g}");
                assert!(!s[g].serial_after_refusal, "{argue:?} {what}: group {g}");
            }
            // Each transition's window is stamped on the group it argued:
            // every group taken after the first (0 without a device).
            assert_eq!(
                s[order[0]].transition_reserved, 0,
                "{argue:?} {what}: the first argue ran alone"
            );
        }
    }
}

/// Off, phase B is one group at a time: no group opens beside an argue, none
/// is counted serial after a refusal, and the driver pays every opening.
#[test]
fn with_the_overlap_off_every_group_opens_alone() {
    let (proved, _) = prove(
        ArgueFormat::PerTable,
        PhaseBOverlap::Off,
        &in_order(),
        OverlapFaults::default(),
    );
    let proved = proved.expect("prove");
    for (g, s) in stamps(&proved).iter().enumerate() {
        assert!(!s.overlapped, "group {g}");
        assert!(!s.serial_after_refusal, "group {g}");
        assert_eq!(s.transition_reserved, 0, "group {g}");
        assert!(s.open_paid >= s.open, "group {g}: paid {s:?}");
    }
}

/// The observer hears each group once, in phase B's order, with its share of
/// the proof — the same shares with the overlap on and off — and each share
/// is the proof's.
#[test]
fn the_observer_hears_the_same_shares_in_order() {
    let n = SIZES.len();
    let reversed: Vec<usize> = (0..n).rev().collect();
    for argue in ARGUES {
        let (serial, heard_serial) = prove(
            argue,
            PhaseBOverlap::Off,
            &reversed,
            OverlapFaults::default(),
        );
        let serial = serial.expect("prove");
        let (proved, heard) = prove(
            argue,
            PhaseBOverlap::On,
            &reversed,
            OverlapFaults::default(),
        );
        proved.expect("prove");
        let groups: Vec<usize> = heard.iter().map(|(g, _)| *g).collect();
        assert_eq!(groups, reversed, "{argue:?}: the order heard");
        assert_eq!(heard, heard_serial, "{argue:?}: the shares heard");
        // Each share as the proof carries it.
        let (proof, argues, prepared, _) = &serial;
        let mut table = 0usize;
        let mut first_table = vec![0usize; n];
        for (g, &size) in SIZES.iter().enumerate() {
            first_table[g] = table;
            table += size;
        }
        for (g, share) in &heard {
            let tables = match argue {
                ArgueFormat::PerTable => {
                    &proof.tables[first_table[*g]..first_table[*g] + SIZES[*g]]
                }
                ArgueFormat::Batched { .. } => &proof.tables[..0],
            };
            let k = PREPARED.iter().position(|(group, _)| group == g);
            let expected = bincode::serialize(&(
                &proof.roots,
                argues.get(*g).filter(|_| argue != ArgueFormat::PerTable),
                tables,
                &proof.columns[*g],
                k.map(|k| &prepared[k]),
            ))
            .expect("serialize");
            assert_eq!(share, &expected, "{argue:?}: group {g}'s share");
        }
    }
}

/// ★ A failure is the prove's error, never a panic: an opening that fails
/// beside an argue, an argue that fails beside an opening (the opening
/// thread joined first), both in one transition (the opening's, the earlier
/// group's), and a panic on the opening thread (refused, not resumed). The
/// failed opening is the same error with the overlap off.
#[test]
fn a_failure_beside_an_argue_is_the_proves_error() {
    let failed_opening = MlError::DeviceFailed {
        stage: "a test's failed opening",
    };
    let failed_argue = MlError::DeviceFailed {
        stage: "a test's failed argue",
    };
    let panicked = MlError::DeviceFailed {
        stage: "the previous group's openings",
    };
    for (what, overlap, faults, expected, heard_groups) in [
        (
            "an opening fails",
            PhaseBOverlap::On,
            OverlapFaults {
                open_fails: Some(1),
                ..OverlapFaults::default()
            },
            &failed_opening,
            vec![0],
        ),
        (
            "an opening fails, one at a time",
            PhaseBOverlap::Off,
            OverlapFaults {
                open_fails: Some(1),
                ..OverlapFaults::default()
            },
            &failed_opening,
            vec![0],
        ),
        (
            "an argue fails",
            PhaseBOverlap::On,
            OverlapFaults {
                argue_fails: Some(3),
                ..OverlapFaults::default()
            },
            &failed_argue,
            vec![0, 1, 2],
        ),
        (
            "both, in one transition",
            PhaseBOverlap::On,
            OverlapFaults {
                open_fails: Some(2),
                argue_fails: Some(3),
                ..OverlapFaults::default()
            },
            &failed_opening,
            vec![0, 1],
        ),
        (
            "the opening thread panics",
            PhaseBOverlap::On,
            OverlapFaults {
                open_panics: Some(4),
                ..OverlapFaults::default()
            },
            &panicked,
            vec![0, 1, 2, 3],
        ),
    ] {
        let (proved, heard) = prove(ArgueFormat::PerTable, overlap, &in_order(), faults);
        let err = proved.expect_err(what);
        assert_eq!(format!("{err:?}"), format!("{expected:?}"), "{what}");
        // The groups opened before the failure were heard, and no other.
        let heard: Vec<usize> = heard.iter().map(|(g, _)| *g).collect();
        assert_eq!(heard, heard_groups, "{what}");
    }
}

/// An opening thread slower than the argue beside it moves no byte: what the
/// driver waits for is paid, at least the stall less that argue and the
/// upload beside it.
#[test]
fn a_stalled_opening_thread_is_paid_and_moves_no_byte() {
    let stall = std::time::Duration::from_millis(150);
    let (serial, _) = prove(
        ArgueFormat::PerTable,
        PhaseBOverlap::Off,
        &in_order(),
        OverlapFaults::default(),
    );
    let (proved, _) = prove(
        ArgueFormat::PerTable,
        PhaseBOverlap::On,
        &in_order(),
        OverlapFaults {
            open_stall: Some(stall),
            ..OverlapFaults::default()
        },
    );
    let (serial, proved) = (serial.expect("prove"), proved.expect("prove"));
    assert_eq!(bytes(&proved), bytes(&serial));
    let s = stamps(&proved);
    let order = in_order();
    for w in order.windows(2) {
        let (h, g) = (w[0], w[1]);
        assert!(s[h].overlapped, "group {h}");
        // The opening thread starts as the argue does, so the stall ends
        // within the argue, the upload's wait after it (no card here: none)
        // and what the driver paid.
        let covered = s[h].open_paid + s[g].argue + 0.05;
        assert!(
            covered >= stall.as_secs_f64(),
            "group {h}: paid {:.3} s beside an argue of {:.3} s",
            s[h].open_paid,
            s[g].argue
        );
    }
}

/// ★ A refusal turns the rest of phase B serial: the transition that sees one
/// keeps its overlap, every later group but the last opens alone and is
/// counted, and the proof is the same.
#[test]
fn a_refusal_turns_the_rest_of_phase_b_serial() {
    let n = SIZES.len();
    let (serial, _) = prove(
        ArgueFormat::PerTable,
        PhaseBOverlap::Off,
        &in_order(),
        OverlapFaults::default(),
    );
    let (proved, _) = prove(
        ArgueFormat::PerTable,
        PhaseBOverlap::On,
        &in_order(),
        OverlapFaults {
            refusal_at: Some(2),
            ..OverlapFaults::default()
        },
    );
    let (serial, proved) = (serial.expect("prove"), proved.expect("prove"));
    assert_eq!(bytes(&proved), bytes(&serial));
    let s = stamps(&proved);
    // Groups 0 and 1 opened beside the argues at places 1 and 2; the refusal
    // at place 2 sends groups 2..n-1 to the driver, the last one as always.
    let overlapped: Vec<usize> = (0..n).filter(|&g| s[g].overlapped).collect();
    let serial_after: Vec<usize> = (0..n).filter(|&g| s[g].serial_after_refusal).collect();
    assert_eq!(overlapped, vec![0, 1]);
    assert_eq!(serial_after, (2..n - 1).collect::<Vec<_>>());
}
