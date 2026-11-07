//! The argument's challenge tables on the card, against the host they replace
//! (`LAMBDA_VM_ARGUE_DEVICE_TABLES`): the zerocheck's `eq` weights, and the
//! claim reduce's shift tables and batched columns.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test argue_device_tables -- --test-threads=1
//! ```
//!
//! Needs a GPU. The arms switch process-wide overrides and read process-wide
//! counters, so every test takes one lock as well.
#![cfg(feature = "cuda")]

use std::sync::Arc;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::Error;
use multilinear::batch::{Rule, Weight};
use multilinear::claim_reduce::{self, FactorSource};
use multilinear::constraint_argument::{self, ConstraintCore, FactorKind, TraceData};
use multilinear::eq::{eq_eval, eq_evals, eq_mle, shift_evals};
use multilinear::gpu::{self, TableFault};
use multilinear::mle::Mle;
use multilinear::program::Builder;

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Every override this file sets, put back when the test ends — a failed
/// assertion included, so one red test cannot leave the next one armed.
struct Restore {
    _serial: std::sync::MutexGuard<'static, ()>,
}

impl Drop for Restore {
    fn drop(&mut self) {
        gpu::force_argue_device_tables(None);
        gpu::force_argue_device_columns(None);
        gpu::force_argue_xcheck(None);
        gpu::force_table_fault(None);
    }
}

fn serial() -> Restore {
    Restore {
        _serial: SERIAL.lock().unwrap_or_else(|e| e.into_inner()),
    }
}

/// A column with no structure a kernel could accidentally satisfy.
fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| {
                FE::from(
                    i.wrapping_add(seed)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(seed.wrapping_mul(6364136223846793005))
                        >> 11,
                )
            })
            .collect(),
    )
    .unwrap()
}

fn point(num_vars: usize, seed: u64) -> Vec<EE> {
    (0..num_vars as u64)
        .map(|i| {
            EE::new([
                FE::from(seed + 3 * i + 1),
                FE::from(2 * i + 7),
                FE::from(seed * seed + i),
            ])
        })
        .collect()
}

fn raw(values: &[EE]) -> Vec<u64> {
    values
        .iter()
        .flat_map(|v| gpu::ext3_raw(v).unwrap())
        .collect()
}

fn ext(cells: &[u64]) -> Vec<EE> {
    cells
        .chunks_exact(3)
        .map(gpu::ext3_from_raw::<Ext>)
        .collect()
}

/// The program a session carries when a test only reads its tables: the
/// rounds never run, but a session is built with one.
fn any_program() -> gpu::Lowered {
    let mut b = Builder::<Ext>::new();
    let v = b.var(0);
    gpu::lower(&b.finish(v).unwrap()).unwrap()
}

/// The claim reduce's tables built on the card against the host's, table by
/// table: the shift table at every offset — the first one no shift, so
/// `eq(alpha)` is built in place, or a shift, so it is built apart; offsets
/// past the cube, which wrap — and the batched column of every offset, one
/// member and many, from a run that starts past column 0 of a mixed store.
#[test]
fn the_card_builds_the_reduce_tables_the_host_builds() {
    let _serial = serial();
    math_cuda::device::backend().expect("needs a GPU");
    let lowered = any_program();
    let shapes: [(usize, usize, &[usize]); 5] = [
        (1, 2, &[0, 1]),
        (6, 3, &[0, 1, 2]),
        (10, 16, &[1, 3]),
        (12, 40, &[0, 1, 2, 7, (1 << 12) + 3]),
        (15, 64, &[0, 1]),
    ];
    for (vars, width, offsets) in shapes {
        let rows = 1usize << vars;
        let lead: Vec<Mle<F>> = (0..2).map(|k| column(vars + 1, 700 + k)).collect();
        let table: Vec<Mle<F>> = (0..width)
            .map(|k| column(vars, 13 * k as u64 + 5))
            .collect();
        let stored: Vec<Vec<u64>> = lead
            .iter()
            .chain(&table)
            .map(|c| c.evals().iter().map(|v| *v.value()).collect())
            .collect();
        let stored_ref: Vec<&[u64]> = stored.iter().map(Vec::as_slice).collect();
        let store = math_cuda::columns::DeviceColumns::upload(&stored_ref)
            .expect("the card takes the store");
        let alpha = point(vars, 29 + vars as u64);

        // Offset `i` reads every third column from `i`, each with a weight.
        let members: Vec<Vec<(usize, EE)>> = (0..offsets.len())
            .map(|i| {
                (i..width)
                    .step_by(3)
                    .map(|c| (c, EE::from(1000 + 17 * c as u64 + i as u64)))
                    .collect()
            })
            .collect();
        let groups: Vec<(usize, Vec<u64>, Vec<u64>)> = offsets
            .iter()
            .zip(&members)
            .map(|(offset, members)| {
                let columns = members.iter().map(|(c, _)| *c as u64).collect();
                let weights = members
                    .iter()
                    .flat_map(|(_, w)| gpu::ext3_raw(w).unwrap())
                    .collect();
                (*offset, columns, weights)
            })
            .collect();
        let session = math_cuda::sumcheck::reduce_session(
            &store,
            lead.len(),
            width,
            &raw(&alpha),
            &groups,
            &lowered.nodes,
            &lowered.consts,
            lowered.num_slots,
            lowered.root_slot,
        )
        .unwrap_or_else(|e| panic!("2^{vars} × {width}: {e:?}"));

        for (i, (offset, members)) in offsets.iter().zip(&members).enumerate() {
            assert_eq!(
                ext(&session.factor(2 * i).unwrap()),
                shift_evals(&alpha, *offset),
                "2^{vars} × {width}: the shift table for offset {offset}"
            );
            let batched: Vec<EE> = (0..rows)
                .map(|y| {
                    members
                        .iter()
                        .fold(EE::zero(), |acc, (c, w)| acc + &table[*c].evals()[y] * w)
                })
                .collect();
            assert_eq!(
                ext(&session.factor(2 * i + 1).unwrap()),
                batched,
                "2^{vars} × {width}: the batched column for offset {offset}"
            );
        }
    }
}

/// The zerocheck's weights built on the card from their points, beside a
/// weight uploaded as a table and the trace's own factors, which the building
/// must leave as they were.
#[test]
fn the_card_builds_the_zerocheck_weights_the_host_builds() {
    let _serial = serial();
    math_cuda::device::backend().expect("needs a GPU");
    let lowered = any_program();
    for vars in [1usize, 5, 12, 16] {
        let rows = 1usize << vars;
        let factors: Vec<Vec<u64>> = (0..3u64)
            .map(|k| {
                raw(&(0..rows as u64)
                    .map(|i| EE::from(i * 31 + k))
                    .collect::<Vec<_>>())
            })
            .collect();
        let factor_refs: Vec<&[u64]> = factors.iter().map(Vec::as_slice).collect();
        let device = math_cuda::sumcheck::DeviceFactors::upload(&factor_refs)
            .expect("the card takes the factors");
        let (a, b) = (point(vars, 3), point(vars, 7));
        let table: Vec<EE> = (0..rows as u64).map(|i| EE::from(i * i + 9)).collect();
        let (raw_a, raw_b, raw_table) = (raw(&a), raw(&b), raw(&table));
        let session = device
            .session_with(
                &[
                    math_cuda::sumcheck::Extra::Eq(&raw_a),
                    math_cuda::sumcheck::Extra::Table(&raw_table),
                    math_cuda::sumcheck::Extra::Eq(&raw_b),
                ],
                &lowered.nodes,
                &lowered.consts,
                lowered.num_slots,
                lowered.root_slot,
            )
            .unwrap_or_else(|e| panic!("2^{vars}: {e:?}"));
        for (k, factor) in factors.iter().enumerate() {
            assert_eq!(
                &session.factor(k).unwrap(),
                factor,
                "2^{vars}: trace factor {k}"
            );
        }
        assert_eq!(
            ext(&session.factor(3).unwrap()),
            eq_evals(&a),
            "2^{vars}: eq(a)"
        );
        assert_eq!(
            ext(&session.factor(4).unwrap()),
            table,
            "2^{vars}: the table"
        );
        assert_eq!(
            ext(&session.factor(5).unwrap()),
            eq_evals(&b),
            "2^{vars}: eq(b)"
        );
    }
}

// ---------------------------------------------------------------------------
// The argument: a zerocheck, two claims on shifted reads, and the reduction.
// ---------------------------------------------------------------------------

/// Rows `2^VARS`. Six columns of it are 6·2^14 cells, past the device-columns
/// threshold, so every knob here has something to move.
const VARS: usize = 14;
const WIDTH: usize = 6;
/// Factor slots: the six columns, column 3 a row ahead, column 4 two rows
/// ahead, then the two weights.
const AHEAD_1: usize = WIDTH;
const AHEAD_2: usize = WIDTH + 1;
const EQ_R: usize = WIDTH + 2;
const EQ_ROW: usize = WIDTH + 3;

/// `c2 = c0 · c1` row by row — the zerocheck's constraint — and the rest free.
fn trace_columns() -> Vec<Mle<F>> {
    let (c0, c1) = (column(VARS, 3), column(VARS, 5));
    let c2 = Mle::new(
        c0.evals()
            .iter()
            .zip(c1.evals())
            .map(|(a, b)| a * b)
            .collect(),
    )
    .unwrap();
    let mut columns = vec![c0, c1, c2];
    columns.extend((3..WIDTH as u64).map(|k| column(VARS, 40 + k)));
    columns
}

fn kinds() -> Vec<FactorKind> {
    (0..WIDTH)
        .map(FactorKind::direct)
        .chain([FactorKind::shifted(3, 1), FactorKind::shifted(4, 2)])
        .collect()
}

/// The zerocheck `eq(r)·(c0·c1 − c2)` and the claims `eq(row)·c3[+1]` and
/// `eq(row)·c4[+2]`: as programs, which the device runs, or as closures,
/// which it cannot — so the zerocheck is turned down and its weights are built
/// on the host after all.
fn rules(compiled: bool) -> Vec<Rule<'static, Ext>> {
    if !compiled {
        return vec![
            Rule::new(3, |v: &[EE]| v[EQ_R] * (v[0] * v[1] - v[2])),
            Rule::new(2, |v: &[EE]| v[EQ_ROW] * v[AHEAD_1]),
            Rule::new(2, |v: &[EE]| v[EQ_ROW] * v[AHEAD_2]),
        ];
    }
    let zerocheck = {
        let mut b = Builder::<Ext>::new();
        let (w, x, y, z) = (b.var(EQ_R), b.var(0), b.var(1), b.var(2));
        let product = b.mul(x, y);
        let constraint = b.sub(product, z);
        let root = b.mul(w, constraint);
        b.finish(root).unwrap()
    };
    let claim = |slot: usize| {
        let mut b = Builder::<Ext>::new();
        let (w, v) = (b.var(EQ_ROW), b.var(slot));
        let root = b.mul(w, v);
        b.finish(root).unwrap()
    };
    vec![
        Rule::compiled(3, zerocheck),
        Rule::compiled(2, claim(AHEAD_1)),
        Rule::compiled(2, claim(AHEAD_2)),
    ]
}

fn r_point() -> Vec<EE> {
    point(VARS, 41)
}

fn row_point() -> Vec<EE> {
    point(VARS, 43)
}

fn claims(columns: &[Mle<F>]) -> Vec<EE> {
    let row = row_point();
    let ahead = |column, offset| {
        claim_reduce::materialize(columns, &FactorSource::shifted(column, offset))
            .unwrap()
            .evaluate_in(&row)
            .unwrap()
    };
    vec![EE::zero(), ahead(3, 1), ahead(4, 2)]
}

fn transcript() -> DefaultTranscript<Ext> {
    DefaultTranscript::<Ext>::new(b"argue-device-tables")
}

/// One arm: how it ran, and what it left.
#[derive(Clone, Copy)]
struct Arm {
    /// The epoch's columns and the table's factors on the card.
    resident: bool,
    /// `LAMBDA_VM_ARGUE_DEVICE_TABLES`: the weights go as points.
    tables: bool,
    /// `LAMBDA_VM_ARGUE_DEVICE_COLUMNS`.
    columns: bool,
    /// Programs, which a device runs, or closures.
    compiled: bool,
}

struct Argued {
    core: ConstraintCore<Ext>,
    point: Vec<EE>,
    state: [u8; 32],
    next: EE,
    /// Challenge tables and column values the card made during the arm.
    tables_on_card: u64,
    columns_on_card: u64,
}

fn argue(columns: &[Mle<F>], arm: Arm) -> Result<Argued, Error> {
    gpu::force_argue_device_tables(Some(arm.tables));
    gpu::force_argue_device_columns(Some(arm.columns));
    let mut trace = TraceData::new(columns.to_vec(), kinds(), Vec::new())?;
    if arm.resident {
        // A taller table first, so the run starts past column 0 of a store of
        // mixed heights, as an epoch's does.
        let lead: Vec<Mle<F>> = (0..2).map(|k| column(VARS + 1, 300 + k)).collect();
        let all: Vec<&Mle<F>> = lead.iter().chain(columns).collect();
        let store = gpu::upload_columns(&all).expect("the card takes the columns");
        trace.set_resident(Arc::new(store), lead.len());
        trace
            .reside_from_columns()
            .expect("the card takes the factors");
    }
    let (r, row) = (r_point(), row_point());
    let weights = if arm.tables {
        vec![Weight::Eq(r), Weight::Eq(row)]
    } else {
        vec![Weight::Table(eq_mle(&r)?), Weight::Table(eq_mle(&row)?)]
    };
    let (tables, cards) = (gpu::argue_tables_on_card(), gpu::evaluate_calls());
    let mut t = transcript();
    let (core, point) = constraint_argument::prove_core::<F, Ext, _>(
        &trace,
        weights,
        rules(arm.compiled),
        &claims(columns),
        &mut t,
    )?;
    Ok(Argued {
        core,
        point,
        state: t.state(),
        next: t.sample_field_element(),
        tables_on_card: gpu::argue_tables_on_card() - tables,
        columns_on_card: gpu::evaluate_calls() - cards,
    })
}

/// Every value the core carries, in order, each list led by its length — the
/// canonical content of its bytes. The field marker derives no `PartialEq`,
/// so the proof structs' own `==` is not there to use.
fn canonical(core: &ConstraintCore<Ext>) -> Vec<EE> {
    let mut out = Vec::new();
    let mut list = |values: &[EE]| {
        out.push(EE::from(values.len() as u64));
        out.extend_from_slice(values);
    };
    for round in &core.sumcheck.rounds {
        list(&round.evaluations);
    }
    list(&core.factor_values);
    for round in &core.reduce.sumcheck.rounds {
        list(&round.evaluations);
    }
    list(&core.reduce.column_values);
    out
}

/// The verifier's side, whole: the batch against the closed-form weights, the
/// reduction, and — the commitment's half, done here directly — the column
/// values at the reduced point.
fn verify(columns: &[Mle<F>], core: &ConstraintCore<Ext>) -> Result<(), Error> {
    let (r, row) = (r_point(), row_point());
    let reduced = constraint_argument::verify_core(
        core,
        &kinds(),
        WIDTH,
        VARS,
        &rules(true),
        &claims(columns),
        |at: &[EE]| Ok(vec![eq_eval(&r, at)?, eq_eval(&row, at)?]),
        &mut transcript(),
    )?;
    for (k, (column, value)) in columns.iter().zip(&reduced.column_values).enumerate() {
        if column.evaluate_in(&reduced.point)? != *value {
            return Err(Error::ColumnOpeningRejected { column: k });
        }
    }
    Ok(())
}

const TODAY: Arm = Arm {
    resident: true,
    tables: false,
    columns: false,
    compiled: true,
};

/// ★ The stage's identity. The same argument with its tables built on the
/// card or on the host — alone, beside the columns on the card, with the
/// zerocheck turned down (closure rules), with nothing resident — is the same
/// proof, point and transcript as today's, and verifies. And each arm took the
/// path it is named for, so no comparison is of a path with itself.
#[test]
fn the_argument_proves_the_same_bytes_with_its_tables_on_the_card() {
    let _serial = serial();
    let columns = trace_columns();
    let today = argue(&columns, TODAY).expect("today's arm proves");
    verify(&columns, &today.core).expect("today's arm verifies");
    assert_eq!(
        (today.tables_on_card, today.columns_on_card),
        (0, 0),
        "today: nothing on the card but the rounds"
    );

    // (arm, tables on the card, column values on the card)
    let arms = [
        // The zerocheck's two weights and the reduce's three offsets × 2.
        (
            Arm {
                tables: true,
                ..TODAY
            },
            8,
            0,
        ),
        (
            Arm {
                columns: true,
                ..TODAY
            },
            0,
            6,
        ),
        (
            Arm {
                tables: true,
                columns: true,
                ..TODAY
            },
            8,
            6,
        ),
        // Closures: the zerocheck is turned down, and its weights are built
        // on the host; the reduce's program is its own, so its tables are not.
        (
            Arm {
                tables: true,
                compiled: false,
                ..TODAY
            },
            6,
            0,
        ),
        // Nothing resident: every table on the host, as it always was.
        (
            Arm {
                tables: true,
                columns: true,
                resident: false,
                ..TODAY
            },
            0,
            0,
        ),
    ];
    for (arm, tables, cards) in arms {
        let what = format!(
            "resident {} · tables {} · columns {} · compiled {}",
            arm.resident, arm.tables, arm.columns, arm.compiled
        );
        let argued = argue(&columns, arm).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!(
            (argued.tables_on_card, argued.columns_on_card),
            (tables, cards),
            "{what}: the path taken (tables, columns on the card)"
        );
        assert_eq!(
            canonical(&argued.core),
            canonical(&today.core),
            "{what}: the proof moved"
        );
        assert_eq!(argued.point, today.point, "{what}: the reduced point moved");
        assert_eq!(argued.state, today.state, "{what}: the transcripts parted");
        assert_eq!(argued.next, today.next, "{what}: the next challenge moved");
        verify(&columns, &argued.core).unwrap_or_else(|e| panic!("{what}: {e:?}"));
    }
}

/// ⛔ THE NEGATIVE CONTROLS: a table the card built wrong is seen by every
/// check — the identity, the verifier, and `LAMBDA_VM_ARGUE_XCHECK`, which
/// refuses to prove — and, with nothing wrong, the cross-check proves the
/// same bytes and says it compared every table.
#[test]
fn a_wrong_table_on_the_card_is_seen_by_every_check() {
    let _serial = serial();
    let columns = trace_columns();
    let today = argue(&columns, TODAY).expect("today's arm proves");
    let on_card = Arm {
        tables: true,
        ..TODAY
    };

    for (fault, rejected) in [
        (TableFault::Eq, Error::BatchMismatch),
        (TableFault::Batched, Error::ShiftedReadMismatch),
    ] {
        gpu::force_table_fault(Some(fault));
        let faulted = argue(&columns, on_card).unwrap_or_else(|e| panic!("{fault:?}: {e:?}"));
        assert_eq!(
            faulted.tables_on_card, 8,
            "{fault:?}: the fault is on the card's path"
        );
        assert_ne!(
            canonical(&faulted.core),
            canonical(&today.core),
            "{fault:?}: the identity must see the corrupted table"
        );
        assert_eq!(
            verify(&columns, &faulted.core),
            Err(rejected),
            "{fault:?}: the verifier must reject the proof over a corrupted table"
        );

        gpu::force_argue_xcheck(Some(true));
        let stage = match fault {
            TableFault::Eq => "eq weight",
            TableFault::Batched => "reduce tables",
        };
        assert_eq!(
            argue(&columns, on_card).err(),
            Some(Error::DeviceFailed { stage }),
            "{fault:?}: the cross-check must refuse a table the host disagrees with"
        );
        gpu::force_argue_xcheck(None);
        gpu::force_table_fault(None);
    }

    gpu::force_argue_xcheck(Some(true));
    let checked = gpu::argue_table_xchecks();
    let clean = argue(&columns, on_card).expect("the cross-checked arm proves");
    assert_eq!(
        gpu::argue_table_xchecks() - checked,
        8,
        "the cross-check must count every table it compared"
    );
    assert_eq!(canonical(&clean.core), canonical(&today.core));
    assert_eq!(clean.state, today.state);
    verify(&columns, &clean.core).expect("the cross-checked arm verifies");
}
