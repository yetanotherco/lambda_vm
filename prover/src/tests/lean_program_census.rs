//! Census (ignored by default): every VM table's zerocheck program as the WHIR
//! argue runs it, today's against the lean one — its steps, its roots, and the
//! live set that sizes a device round's per-thread slot file, with the thread
//! ceiling that live set leaves under the round kernel's slot budget. Lane
//! I-GFS, S0 of the narrow-round lever (`thoughts/zf/gap2/fix2/I-GFS.md` §15).
//!
//! It also checks, at a random point, that the two programs are one polynomial.
//!
//! `cargo test -p lambda-vm-prover --lib tests::lean_program_census -- --ignored --nocapture`

use math::field::element::FieldElement;
use stark::multilinear_air::{Uniforms, beta_powers};
use stark::multilinear_table::{TableLayout, weight_slots};
use stark::proof::options::GoldilocksCubicProofOptions;
use stark::traits::AIR;

use multilinear::program::Program;

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::*;

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type EE = FieldElement<Ext3>;

fn point(n: usize, seed: u64) -> Vec<EE> {
    (0..n as u64)
        .map(|i| {
            let x = i.wrapping_add(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            EE::new([
                FieldElement::from(x >> 7),
                FieldElement::from(x >> 11),
                FieldElement::from(x >> 3),
            ])
        })
        .collect()
}

// ── S1a: the zerocheck batches on demand (`LAMBDA_VM_ARGUE_LEAN_PROGRAM`) ────

pub(crate) type VmAir = Box<dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>>;

/// Every VM table's AIR, named as the census names them.
pub(crate) fn vm_airs() -> Vec<(VmAir, &'static str)> {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");
    vec![
        (Box::new(create_cpu_air(&opts, 0)), "CPU"),
        (Box::new(create_bitwise_air(&opts)), "BITWISE"),
        (Box::new(create_lt_air(&opts)), "LT"),
        (Box::new(create_shift_air(&opts)), "SHIFT"),
        (Box::new(create_eq_air(&opts)), "EQ"),
        (Box::new(create_bytewise_air(&opts)), "BYTEWISE"),
        (Box::new(create_store_air(&opts)), "STORE"),
        (Box::new(create_memw_air(&opts)), "MEMW"),
        (Box::new(create_memw_aligned_air(&opts)), "MEMW_A"),
        (Box::new(create_memw_register_air(&opts)), "MEMW_R"),
        (Box::new(create_load_air(&opts)), "LOAD"),
        (Box::new(create_decode_air(&opts)), "DECODE"),
        (Box::new(create_mul_air(&opts)), "MUL"),
        (Box::new(create_dvrm_air(&opts)), "DVRM"),
        (Box::new(create_branch_air(&opts)), "BRANCH"),
        (Box::new(create_halt_air(&opts)), "HALT"),
        (Box::new(create_hint_air(&opts)), "HINT"),
        (Box::new(create_copy_air(&opts)), "COPY"),
        (Box::new(create_page_air(&opts, 0x1000)), "PAGE"),
        (Box::new(create_register_air(&opts, 0)), "REGISTER"),
        (Box::new(create_keccak_air(&opts)), "KECCAK"),
        (Box::new(create_keccak_rnd_air(&opts)), "KECCAK_RND"),
        (Box::new(create_keccak_rc_air(&opts)), "KECCAK_RC"),
        (Box::new(create_ecsm_air(&opts)), "ECSM"),
        (Box::new(create_ecdas_air(&opts)), "ECDAS"),
        (
            Box::new(crate::continuation::l2g_memory_air(
                &opts,
                crate::tables::local_to_global::epoch_label(1),
            )),
            "L2G",
        ),
    ]
}

/// A VM table's zerocheck batch as `multilinear_table::prove` builds it — the
/// constraint rule beside the bus's two, combined — with its round degree and
/// the factors it reads.
struct Batch {
    program: Program<Ext3>,
    degree: usize,
    factors: usize,
}

fn batch_of(
    air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
    label: &str,
) -> Batch {
    const NUM_VARS: usize = 10;
    let layout = TableLayout::<Gl, Ext3>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        air.trace_layout().0,
        NUM_VARS,
        Uniforms::default(),
    )
    .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let shape = layout.shape();
    let betas = beta_powers(&EE::from(0x5eed_u64), shape.num_roots());
    let (weight_r, weight_z) = weight_slots(layout.kinds().len());
    let zerocheck = shape
        .program(&betas, weight_r)
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let slots = layout.slot_of().to_vec();
    let interactions = stark::multilinear_logup::interactions(
        air.bus_interactions(),
        slots.len(),
        &EE::from(0x2a_u64),
        &EE::from(0x3b_u64),
        |col| {
            slots
                .get(col)
                .copied()
                .ok_or(multilinear::Error::UnknownPolynomial {
                    index: col,
                    len: slots.len(),
                })
        },
    )
    .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let claim = point(
        multilinear::logup::input_layer_vars(interactions.len(), NUM_VARS),
        7,
    );
    let bus = multilinear::logup::claim_statements(&interactions, &claim, NUM_VARS, weight_z)
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let program = multilinear::program::combine(
        &[
            &zerocheck,
            bus.numerator.program().expect("compiled"),
            bus.denominator.program().expect("compiled"),
        ],
        &point(3, 11),
    )
    .unwrap_or_else(|e| panic!("{label}: {e:?}"));
    let degree = (shape.degree() + 1)
        .max(bus.numerator.degree())
        .max(bus.denominator.degree());
    Batch {
        program,
        degree,
        factors: weight_z + 1,
    }
}

/// ★ Host parity, every VM table: each zerocheck batch on demand is the same
/// polynomial as today's at random points, and holds no more values a thread.
/// The four big batches — and only they — are over the gate, and on demand
/// every one of them falls under it: a first round of at least 64 k threads.
/// The byte gate's EQ fixture is not big, so it keeps today's program.
#[test]
fn every_vm_batch_on_demand_is_the_same_program() {
    let mut big = Vec::new();
    for (air, label) in vm_airs() {
        let batch = batch_of(&*air, label);
        assert!(
            (1..=math_cuda::sumcheck::MAX_NODES).contains(&batch.degree),
            "{label}: a round of degree {} is not one the card takes",
            batch.degree
        );
        let demand = batch.program.on_demand();
        let mut scratch = Vec::new();
        for seed in [1u64, 29, 311] {
            let values = point(batch.factors, seed);
            assert_eq!(
                demand.eval(&values, &mut scratch),
                batch.program.eval(&values, &mut scratch),
                "{label}: on demand is another polynomial (seed {seed})"
            );
        }
        let today = multilinear::gpu::lower(&batch.program).expect("today's lowers");
        let lean = multilinear::gpu::lower(&demand).expect("on demand lowers");
        assert!(
            lean.num_slots <= today.num_slots,
            "{label}: on demand holds {} values a thread, today {}",
            lean.num_slots,
            today.num_slots
        );
        if multilinear::gpu::is_big_batch(today.num_slots) {
            assert!(
                !multilinear::gpu::is_big_batch(lean.num_slots),
                "{label}: on demand still holds {} values a thread",
                lean.num_slots
            );
            big.push(label);
        }
    }
    assert_eq!(
        big,
        ["KECCAK", "KECCAK_RND", "ECSM", "ECDAS"],
        "the big batches"
    );
}

/// ★ Device parity, every big batch — the VM's four, found
/// by the gate: on the card, the program on demand — its slot file sized for
/// every node from the first round — gives today's round sums, round by round
/// from a 2^14 cube (whose first round is past KECCAK_RND's 8,096-thread
/// ceiling) down to the host crossover; and a shadow session walking today's
/// program over the lean session's own factors ([`SumcheckSession::shadow`],
/// what `LAMBDA_VM_ARGUE_XCHECK` runs) agrees with both. Needs a device, and
/// says so rather than passing without one.
///
/// ```text
/// cargo test --release -p lambda-vm-prover --features cuda --lib \
///     tests::lean_program_census::every_big_batch_rounds_the_same_on_demand_on_the_card -- --exact --nocapture
/// ```
#[cfg(feature = "cuda")]
#[test]
fn every_big_batch_rounds_the_same_on_demand_on_the_card() {
    use math_cuda::sumcheck::{DeviceFactors, thread_ceiling};

    const CUBE: usize = 1 << 14;
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    if math_cuda::device::backend().is_err() {
        eprintln!("lean program: SKIPPED, no device");
        return;
    }
    let canonical = |sums: Vec<u64>| -> Vec<u64> {
        sums.into_iter()
            .map(|x| if x >= P { x - P } else { x })
            .collect()
    };
    let mut state = 0x243F_6A88_85A3_08D3_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % P
    };
    let vm = vm_airs();
    let every: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
        String,
    )> = vm
        .iter()
        .map(|(air, label)| (&**air, label.to_string()))
        .collect();
    let mut big = Vec::new();
    for (air, label) in every {
        let label = label.as_str();
        let batch = batch_of(air, label);
        let today = multilinear::gpu::lower(&batch.program).expect("today's lowers");
        if !multilinear::gpu::is_big_batch(today.num_slots) {
            continue;
        }
        big.push(label.to_string());
        let demand = multilinear::gpu::lower(&batch.program.on_demand()).expect("on demand lowers");
        let factors: Vec<Vec<u64>> = (0..batch.factors)
            .map(|_| (0..CUBE * 3).map(|_| next()).collect())
            .collect();
        let raw: Vec<&[u64]> = factors.iter().map(Vec::as_slice).collect();
        let held_factors = DeviceFactors::upload(&raw).expect("today's factors go up");
        let lean_factors = DeviceFactors::upload(&raw).expect("the lean session's factors go up");
        let mut held = held_factors
            .session_with(
                &[],
                &today.nodes,
                &today.consts,
                today.num_slots,
                today.root_slot,
            )
            .expect("today's session");
        let mut lean = lean_factors
            .session_spread(
                &[],
                &demand.nodes,
                &demand.consts,
                demand.num_slots,
                demand.root_slot,
                batch.degree,
            )
            .expect("the session on demand");
        let mut shadow = lean
            .shadow(
                &today.nodes,
                &today.consts,
                today.num_slots,
                today.root_slot,
            )
            .expect("the shadow");
        let t: Vec<u64> = (1..=batch.degree as u64)
            .flat_map(|node| [node, 0, 0])
            .collect();
        let rounds = (CUBE / 32).trailing_zeros() as usize;
        for round in 0..rounds {
            let expected = canonical(held.round(&t).expect("today's round"));
            let got = canonical(lean.round(&t).expect("the round on demand"));
            shadow.follow(&lean);
            let beside = canonical(shadow.round(&t).expect("the shadow's round"));
            assert_eq!(got, expected, "{label}: round {round} on demand");
            assert_eq!(beside, expected, "{label}: the shadow's round {round}");
            let r = [next(), next(), next()];
            held.fold(&r).expect("today's fold");
            lean.fold(&r).expect("the fold on demand");
        }
        eprintln!(
            "lean program: {label}: {rounds} rounds of degree {} equal on demand · slots {} → {} · \
             thread ceiling {} → {}",
            batch.degree,
            today.num_slots,
            demand.num_slots,
            thread_ceiling(today.num_slots),
            thread_ceiling(demand.num_slots),
        );
    }
    assert_eq!(
        big,
        ["KECCAK", "KECCAK_RND", "ECSM", "ECDAS"],
        "the big batches"
    );
}
