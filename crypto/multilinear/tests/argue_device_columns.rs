//! The claim reduce's column values on the card, against the host they replace
//! (`LAMBDA_VM_ARGUE_DEVICE_COLUMNS`).
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test argue_device_columns -- --test-threads=1
//! ```
//!
//! Needs a GPU. The arms switch process-wide overrides and read process-wide
//! counters, so every test takes one lock as well.
//!
//! The reference is the host's own `Mle::evaluate_in` at heights below 2^16,
//! where it never asks the device — so what is compared is the card against the
//! code it replaces, not against itself.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::Error;
use multilinear::claim_reduce::{self, FactorSource, ReduceProof};
use multilinear::gpu;
use multilinear::mle::Mle;

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
        gpu::force_argue_device_columns(None);
        gpu::force_argue_xcheck(None);
        gpu::force_column_value_fault(false);
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

/// The batched evaluation of a short table — read where it lies in a store it
/// shares with a taller table, and uploaded — against the host, column by
/// column: every height the knob moves onto the card, from one variable up,
/// and widths up to the widest precompile's and past it.
///
/// The two device arms read the same cells with the same kernels, so they
/// must agree bit for bit; the host agrees with them as field elements.
#[test]
fn the_card_values_short_columns_as_the_host_does() {
    let _serial = serial();
    math_cuda::device::backend().expect("needs a GPU");
    let shapes = [
        (1usize, 3usize),
        (2, 7),
        (3, 1),
        (5, 2000),
        (8, 256),
        (10, 64),
        (12, 1480),
        (13, 5),
        (15, 1480),
    ];
    for (vars, width) in shapes {
        let rows = 1usize << vars;
        assert!(rows < 1 << 16, "the host reference must stay on the host");
        // A taller table first, so the run under test starts past column 0 of
        // a store whose columns are not all one height — as an epoch's are.
        let lead: Vec<Mle<F>> = (0..2).map(|k| column(vars + 1, 900 + k)).collect();
        let table: Vec<Mle<F>> = (0..width)
            .map(|k| column(vars, 31 * k as u64 + 7))
            .collect();
        let raw = |c: &Mle<F>| c.evals().iter().map(|v| *v.value()).collect::<Vec<u64>>();
        let stored: Vec<Vec<u64>> = lead.iter().chain(&table).map(raw).collect();
        let stored_ref: Vec<&[u64]> = stored.iter().map(Vec::as_slice).collect();
        let store = math_cuda::columns::DeviceColumns::upload(&stored_ref)
            .expect("the card takes the store");
        assert!(store.is_run(lead.len(), width), "the table is a run");

        let at = point(vars, vars as u64 * 101 + width as u64);
        let raw_point: Vec<u64> = at.iter().flat_map(|c| gpu::ext3_raw(c).unwrap()).collect();
        let resident = math_cuda::sumcheck::evaluate_many_base(
            math_cuda::columns::Columns::Device {
                store: &store,
                first: lead.len(),
                width,
            },
            &raw_point,
        )
        .unwrap_or_else(|e| panic!("2^{vars} × {width}, resident: {e:?}"));
        let uploaded = math_cuda::sumcheck::evaluate_many_base(
            math_cuda::columns::Columns::Host(&stored_ref[lead.len()..]),
            &raw_point,
        )
        .unwrap_or_else(|e| panic!("2^{vars} × {width}, uploaded: {e:?}"));
        assert_eq!(resident.len(), width);
        assert_eq!(
            resident, uploaded,
            "2^{vars} × {width}: the resident and uploaded evaluations differ"
        );
        for (k, (column, value)) in table.iter().zip(&resident).enumerate() {
            let host = column.evaluate_in(&at).expect("the host evaluation");
            assert_eq!(
                gpu::ext3_from_raw::<Ext>(value),
                host,
                "2^{vars} × {width}: column {k} differs from the host"
            );
        }
    }
}

/// A table's reduction: its columns, each read once unshifted and every third
/// one read a step ahead too, and the honest claims about them.
struct Case {
    columns: Vec<Mle<F>>,
    sources: Vec<FactorSource>,
    values: Vec<EE>,
    alpha: Vec<EE>,
}

fn case(num_vars: usize, width: usize) -> Case {
    let columns: Vec<Mle<F>> = (0..width)
        .map(|k| column(num_vars, 17 * k as u64 + 3))
        .collect();
    let sources: Vec<FactorSource> = (0..width)
        .map(FactorSource::direct)
        .chain((0..width).step_by(3).map(|c| FactorSource::shifted(c, 1)))
        .collect();
    let alpha = point(num_vars, 5);
    let values = sources
        .iter()
        .map(|s| {
            claim_reduce::materialize(&columns, s)
                .unwrap()
                .evaluate_in(&alpha)
                .unwrap()
        })
        .collect();
    Case {
        columns,
        sources,
        values,
        alpha,
    }
}

/// What one prove of a reduction left, and what it moved.
struct Reduced {
    proof: ReduceProof<Ext>,
    point: Vec<EE>,
    transcript: DefaultTranscript<Ext>,
    /// Columns the batched device evaluation returned, and columns the host
    /// walked, during this prove.
    on_card: u64,
    on_host: u64,
}

fn transcript() -> DefaultTranscript<Ext> {
    DefaultTranscript::<Ext>::new(b"argue-device-columns")
}

fn reduce(
    c: &Case,
    resident: Option<(&gpu::ResidentColumns, usize)>,
    on: bool,
) -> Result<Reduced, Error> {
    gpu::force_argue_device_columns(Some(on));
    let (card, host) = (gpu::evaluate_calls(), gpu::host_evaluate_calls());
    let mut transcript = transcript();
    let (proof, point) = claim_reduce::prove(
        &c.columns,
        &c.sources,
        &c.values,
        &c.alpha,
        resident,
        &mut transcript,
    )?;
    Ok(Reduced {
        proof,
        point,
        transcript,
        on_card: gpu::evaluate_calls() - card,
        on_host: gpu::host_evaluate_calls() - host,
    })
}

fn verify(c: &Case, proof: &ReduceProof<Ext>) -> Result<(), Error> {
    claim_reduce::verify(
        proof,
        &c.sources,
        &c.values,
        &c.alpha,
        c.columns.len(),
        &mut transcript(),
    )
    .map(|_| ())
}

/// Puts `c`'s columns on the card behind a table of another height, and says
/// where they start — the layout an epoch's store has.
fn resident_store(c: &Case) -> (gpu::ResidentColumns, usize) {
    let lead: Vec<Mle<F>> = (0..3)
        .map(|k| column(c.columns[0].num_vars() + 2, 500 + k))
        .collect();
    let all: Vec<&Mle<F>> = lead.iter().chain(&c.columns).collect();
    (
        gpu::upload_columns(&all).expect("the card takes the columns"),
        lead.len(),
    )
}

/// The first index at which two proofs' column values differ, or `None`.
fn first_difference(a: &Reduced, b: &Reduced) -> Option<usize> {
    let (x, y) = (&a.proof.column_values, &b.proof.column_values);
    if x.len() != y.len() {
        return Some(x.len().min(y.len()));
    }
    (0..x.len()).find(|&k| x[k] != y[k])
}

/// A reduction's round messages, compared value by value: the field marker
/// derives no `PartialEq`, so the proof struct's own `==` is not there to use.
fn rounds(r: &Reduced) -> Vec<&[EE]> {
    r.proof
        .sumcheck
        .rounds
        .iter()
        .map(|round| round.evaluations.as_slice())
        .collect()
}

/// ★ The stage's identity: a resident short table reduced with the knob off
/// (its columns walked on the host) and on (evaluated on the card) gives the
/// same proof, the same point and the same transcript — and each arm took the
/// path it is named for, so the comparison is not of a path with itself.
#[test]
fn a_resident_short_table_reduces_to_the_same_proof_on_the_card() {
    let _serial = serial();
    // 64 columns of 2^10 rows: 2^16 cells, the knob's threshold exactly, in
    // a table no column of which the default would put on the card.
    let c = case(10, 64);
    let (store, first) = resident_store(&c);
    let resident = Some((&store, first));

    let mut off = reduce(&c, resident, false).expect("the knob off proves");
    let mut on = reduce(&c, resident, true).expect("the knob on proves");

    assert_eq!(
        (off.on_card, off.on_host),
        (0, 64),
        "knob off: the host must walk every column"
    );
    assert_eq!(
        (on.on_card, on.on_host),
        (64, 0),
        "knob on: the card must take the whole table"
    );
    assert_eq!(
        first_difference(&off, &on),
        None,
        "a column value moved onto the card changed"
    );
    assert_eq!(rounds(&off), rounds(&on), "the rounds moved");
    assert_eq!(off.point, on.point, "the reduced point moved");
    assert_eq!(
        off.transcript.state(),
        on.transcript.state(),
        "the transcripts parted"
    );
    assert_eq!(
        off.transcript.sample_field_element(),
        on.transcript.sample_field_element(),
        "the next challenge moved"
    );
    // Informational: rkyv writes raw limbs, and a value has two raw forms below
    // 2^32 − 1, so equal values need not be equal rkyv bytes. The serde form —
    // the transcript's, and the canonical one — is what is gated above.
    let raw = |r: &Reduced| rkyv::to_bytes::<rkyv::rancor::Error>(&r.proof).unwrap();
    let rkyv_equal = raw(&off).as_slice() == raw(&on).as_slice();
    eprintln!("argue device columns: rkyv bytes equal across the arms: {rkyv_equal}");
    verify(&c, &off.proof).expect("the knob-off proof verifies");
    verify(&c, &on.proof).expect("the knob-on proof verifies");
}

/// Where the knob stops: a resident table one column short of the threshold
/// stays on the host, and a table that is not resident stays there however
/// wide — it would have to be uploaded, which is what the height pays for.
#[test]
fn the_knob_takes_only_resident_tables_of_enough_cells() {
    let _serial = serial();
    let short = case(10, 63);
    let (store, first) = resident_store(&short);
    let below = reduce(&short, Some((&store, first)), true).expect("proves");
    assert_eq!(
        (below.on_card, below.on_host),
        (0, 63),
        "63 × 2^10 is under 2^16 cells: the host walks it"
    );

    let wide = case(10, 96);
    let away = reduce(&wide, None, true).expect("proves");
    assert_eq!(
        (away.on_card, away.on_host),
        (0, 96),
        "a table the card does not hold stays on the host"
    );
    verify(&short, &below.proof).expect("verifies");
    verify(&wide, &away.proof).expect("verifies");
}

/// ⛔ THE NEGATIVE CONTROLS: the checks above and the block's cross-check can
/// each see a card value that is wrong.
///
/// The fault adds one to the first column's device value. The identity test's
/// comparison must name that column, the proof must fail verification, and
/// `LAMBDA_VM_ARGUE_XCHECK` must refuse to prove at all — while, without the
/// fault, the cross-check proves and says how many columns it compared.
#[test]
fn a_wrong_card_value_is_seen_by_every_check() {
    let _serial = serial();
    let c = case(10, 64);
    let (store, first) = resident_store(&c);
    let resident = Some((&store, first));
    let off = reduce(&c, resident, false).expect("the knob off proves");

    gpu::force_column_value_fault(true);
    let faulted = reduce(&c, resident, true).expect("the faulted arm proves");
    assert_eq!(faulted.on_card, 64, "the fault is on the card's path");
    assert_eq!(
        first_difference(&off, &faulted),
        Some(0),
        "the identity comparison must name the corrupted column"
    );
    assert_eq!(
        verify(&c, &faulted.proof),
        Err(Error::ShiftedReadMismatch),
        "a corrupted column value must fail verification"
    );

    gpu::force_argue_xcheck(Some(true));
    let checked = gpu::argue_xchecks();
    assert_eq!(
        reduce(&c, resident, true).err(),
        Some(Error::DeviceFailed {
            stage: "column evaluation"
        }),
        "the cross-check must refuse a card value the host disagrees with"
    );
    assert_eq!(
        gpu::argue_xchecks(),
        checked,
        "a refused table is not counted"
    );

    gpu::force_column_value_fault(false);
    let clean = reduce(&c, resident, true).expect("the cross-checked arm proves");
    assert_eq!(
        gpu::argue_xchecks() - checked,
        64,
        "the cross-check must count every column it compared"
    );
    assert_eq!(first_difference(&off, &clean), None);
    verify(&c, &clean.proof).expect("the cross-checked proof verifies");
}
