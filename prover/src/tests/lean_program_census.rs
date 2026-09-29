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

use multilinear::program::{Builder, Op, Program};

use crate::tables::types::{GoldilocksExtension, GoldilocksField};
use crate::test_utils::*;

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;
type EE = FieldElement<Ext3>;

/// The round kernel's slot budget and thread cap (`math-cuda/src/sumcheck.rs`).
const SLOT_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
const MAX_THREADS: u64 = 1 << 20;

fn ceiling(num_slots: usize) -> u64 {
    (SLOT_BUDGET_BYTES / (num_slots as u64 * 24).max(1)).min(MAX_THREADS)
}

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

/// `program` with every read of a factor or a constant emitted again at each
/// step that uses it, instead of once and kept: what the slot file would hold
/// if a column value were reloaded rather than held (a census variant; the
/// lowering has no such pass).
fn rematerialized(program: &Program<Ext3>) -> Program<Ext3> {
    let mut b = Builder::<Ext3>::new();
    let steps = program.steps();
    // A value's node, or the step to emit again at each use when it is a read.
    let mut node: Vec<Option<u32>> = Vec::with_capacity(steps.len());
    let fresh = |b: &mut Builder<Ext3>, node: &[Option<u32>], k: u32| -> u32 {
        match (&steps[k as usize], node[k as usize]) {
            (Op::Var(i), _) => b.var(*i as usize),
            (Op::Fixed(c), _) => b.fixed(*c),
            (_, Some(at)) => at,
            _ => unreachable!("an operand precedes its user"),
        }
    };
    for op in steps {
        let at = match op {
            Op::Var(_) | Op::Fixed(_) => None,
            Op::Add(a, x) => {
                let (a, x) = (fresh(&mut b, &node, *a), fresh(&mut b, &node, *x));
                Some(b.add(a, x))
            }
            Op::Sub(a, x) => {
                let (a, x) = (fresh(&mut b, &node, *a), fresh(&mut b, &node, *x));
                Some(b.sub(a, x))
            }
            Op::Mul(a, x) => {
                let (a, x) = (fresh(&mut b, &node, *a), fresh(&mut b, &node, *x));
                Some(b.mul(a, x))
            }
            Op::Neg(a) => {
                let a = fresh(&mut b, &node, *a);
                Some(b.neg(a))
            }
        };
        node.push(at);
    }
    let root = fresh(&mut b, &node, program.root());
    b.finish_verbatim(root).expect("the root is emitted")
}

fn census(air: &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>, label: &str) {
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
    let roots = shape.num_roots();
    let betas = beta_powers(&EE::from(0x5eed_u64), roots);
    let (weight_r, weight_z) = weight_slots(layout.kinds().len());
    let today = shape.program(&betas, weight_r).expect("today's program");
    let lean = shape
        .program_lean(&betas, weight_r)
        .expect("the lean program");
    let values = point(weight_z + 1, roots as u64);
    let mut scratch = Vec::new();
    assert_eq!(
        today.eval(&values, &mut scratch),
        lean.eval(&values, &mut scratch),
        "{label}: the lean program is another polynomial"
    );

    // The batch the zerocheck's session runs: the constraint rule beside the
    // bus's two, as `multilinear_table::prove` builds them.
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
    let lambdas = point(3, 11);
    let batch = |zerocheck: &multilinear::program::Program<Ext3>| {
        multilinear::program::combine(
            &[
                zerocheck,
                bus.numerator.program().expect("compiled"),
                bus.denominator.program().expect("compiled"),
            ],
            &lambdas,
        )
        .expect("the batch combines")
    };
    let (batch_today, batch_lean) = (batch(&today), batch(&lean));

    // The bus's two rules folded the same way: each interaction's side into
    // the running sum as soon as it is emitted (a prototype of
    // `logup::claim_statements`' `weighted`, same weights, same padding).
    let (interaction_point, _) = claim.split_at(claim.len() - NUM_VARS);
    let weights = multilinear::eq::eq_evals(interaction_point);
    let padding = weights[interactions.len()..]
        .iter()
        .fold(EE::zero(), |acc, w| acc + w);
    let lean_side = |sides: Vec<&multilinear::logup::Affine<Ext3>>, constant: Option<EE>| {
        let mut b = multilinear::program::Builder::<Ext3>::new();
        let mut sum: Option<u32> = None;
        for (side, w) in sides.into_iter().zip(&weights) {
            let mut term = side.emit(&mut b);
            if *w != EE::one() {
                let c = b.fixed(*w);
                term = b.mul(c, term);
            }
            sum = Some(match sum {
                Some(acc) => b.add(acc, term),
                None => term,
            });
        }
        if let Some(constant) = constant {
            let c = b.fixed(constant);
            sum = Some(match sum {
                Some(acc) => b.add(acc, c),
                None => c,
            });
        }
        let sum = sum.unwrap_or_else(|| b.fixed(EE::zero()));
        let row = b.var(weight_z);
        let root = b.mul(row, sum);
        b.finish(root).expect("the lean side finishes")
    };
    let numerator = lean_side(interactions.iter().map(|i| &i.numerator).collect(), None);
    let denominator = lean_side(
        interactions.iter().map(|i| &i.denominator).collect(),
        Some(padding),
    );
    for (lean, today) in [
        (&numerator, bus.numerator.program().expect("compiled")),
        (&denominator, bus.denominator.program().expect("compiled")),
    ] {
        assert_eq!(
            lean.eval(&values, &mut scratch),
            today.eval(&values, &mut scratch),
            "{label}: the lean bus side is another polynomial"
        );
    }
    let batch_all_lean =
        multilinear::program::combine(&[&lean, &numerator, &denominator], &lambdas)
            .expect("the batch combines");
    assert_eq!(
        batch_all_lean.eval(&values, &mut scratch),
        batch_today.eval(&values, &mut scratch),
        "{label}: the lean batch is another polynomial"
    );

    let low = |program: &multilinear::program::Program<Ext3>| {
        multilinear::gpu::lower(program).expect("lowers")
    };
    let (zc_today, zc_lean) = (low(&today), low(&lean));
    let (all_today, all_lean) = (low(&batch_today), low(&batch_lean));
    let both = low(&batch_all_lean);
    let remat_program = rematerialized(&batch_all_lean);
    assert_eq!(
        remat_program.eval(&values, &mut scratch),
        batch_today.eval(&values, &mut scratch),
        "{label}: the rematerialized batch is another polynomial"
    );
    let remat = low(&remat_program);
    println!(
        "{label:12} width {:5} · roots {roots:5} · interactions {:4} || batch steps {:6} · slots {:5} || \
         zc lean {:5} · all lean {:5} · + reads reloaded {:5} (steps {:6}) || ceiling {:8} → {:8} threads",
        air.trace_layout().0,
        interactions.len(),
        all_today.nodes.len() / 2,
        all_today.num_slots,
        all_lean.num_slots,
        both.num_slots,
        remat.num_slots,
        remat.nodes.len() / 2,
        ceiling(all_today.num_slots),
        ceiling(remat.num_slots),
    );
    let _ = (zc_today, zc_lean);
}

#[test]
#[ignore = "a printing census for the lean-program design, not a gate"]
fn the_zerocheck_programs_and_their_live_sets() {
    let opts = GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 valid");

    census(&create_cpu_air(&opts), "CPU");
    census(&create_bitwise_air(&opts), "BITWISE");
    census(&create_lt_air(&opts), "LT");
    census(&create_shift_air(&opts), "SHIFT");
    census(&create_eq_air(&opts), "EQ");
    census(&create_bytewise_air(&opts), "BYTEWISE");
    census(&create_store_air(&opts), "STORE");
    census(&create_memw_air(&opts), "MEMW");
    census(&create_memw_aligned_air(&opts), "MEMW_A");
    census(&create_memw_register_air(&opts), "MEMW_R");
    census(&create_load_air(&opts), "LOAD");
    census(&create_decode_air(&opts), "DECODE");
    census(&create_mul_air(&opts), "MUL");
    census(&create_dvrm_air(&opts), "DVRM");
    census(&create_branch_air(&opts), "BRANCH");
    census(&create_halt_air(&opts), "HALT");
    census(&create_hint_air(&opts), "HINT");
    census(&create_commit_air(&opts), "COMMIT");
    census(&create_page_air(&opts, 0x1000), "PAGE");
    census(&create_register_air(&opts), "REGISTER");
    census(&create_keccak_air(&opts), "KECCAK");
    census(&create_keccak_rnd_air(&opts), "KECCAK_RND");
    census(&create_keccak_rc_air(&opts), "KECCAK_RC");
    census(&create_ecsm_air(&opts), "ECSM");
    census(&create_ecdas_air(&opts), "ECDAS");
    census(
        &crate::continuation::l2g_memory_air(&opts, crate::tables::local_to_global::epoch_label(1)),
        "L2G",
    );
}
