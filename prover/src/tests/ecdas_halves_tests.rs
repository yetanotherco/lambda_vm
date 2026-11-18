//! ECDAS's two forms ([`EcdasForm`]): today's byte-position relations
//! ([`EcdasForm::Bytes`]) and the same three relations written over 16-bit
//! limb positions ([`EcdasForm::Halves`]). On the built AIRs, over rows, no
//! proof.
//!
//! - The byte form is pinned byte for byte (its captured program).
//! - The limb form's shape: 515 columns, 101 constraints, 242 interactions;
//!   its byte checks are exactly the byte form's, two to a lookup; each
//!   carry's two parts each checked once; the constraint program's fused
//!   steps against the byte form's.
//! - Completeness: the carry bound computed over every byte value, real
//!   steps (ECSM, near-order scalars, ECREC's chains) and random steps.
//! - Forgeries, each refused, each beside a control that lets it through
//!   when the one piece that refuses it is removed: carries solved in the
//!   field (refused by the `LO` checks, and with `LO` in range by the `HI`
//!   checks), a relation off by `2^512` (refused by the closing rule
//!   `C_31 = 0`), and a byte of 256 (refused by its pair's check).

use math::field::element::FieldElement;
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Signed, Zero};
use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
use stark::frame::Frame;
use stark::lookup::{BusInteraction, BusValue, Multiplicity};
use stark::table::TableView;
use stark::traits::{AIR, TransitionEvaluationContext};

use crate::tables::ecdas::{
    EcdasConstraints, EcdasForm, EcdasHalvesConstraints, EcdasOperation, HALVES_CARRIES,
    HALVES_CARRY_OFFSET, HALVES_POSITIONS, HalvesLimbs, R_BYTES, bus_interactions_for, cols,
    generate_ecdas_rows_in, halves_carries, halves_chain, halves_checked_bytes, halves_cols as h,
    split_carry,
};
use crate::tables::gpack::TraceForm;
use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::test_utils::{create_ecdas_air, create_ecdas_air_for, create_ecdas_halves_air};
use ecsm::{EcdasStep, P_BYTES};

type Trace = stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>;

/// The Goldilocks prime.
const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;

/// The BLAKE3 digest of an AIR's captured constraint program (its
/// constraints and its LogUp emission), as the recursion reads it.
fn program_digest(
    air: &dyn AIR<Field = GoldilocksField, FieldExtension = GoldilocksExtension, PublicInputs = ()>,
) -> String {
    let artifact = stark::constraint_ir::ConstraintArtifact::capture(air);
    let bytes = artifact.to_bytes().expect("the artifact serializes");
    blake3::hash(&bytes).to_hex().to_string()
}

/// ★ The byte form's AIR — its constraint program and its interactions — is
/// the one before the limb form existed, byte for byte (taken at the commit
/// the limb form was added on), and the format presets keep it.
#[test]
fn the_byte_forms_air_is_pinned() {
    let opts = stark::proof::options::ProofOptions::default_test_options();
    assert_eq!(
        program_digest(&create_ecdas_air(&opts)),
        "cdd4c4e6927f9f89b22c5e150e79499d7e2fe6ebf7a125f46913222c9619c4b4"
    );
    assert_eq!(
        program_digest(&*create_ecdas_air_for(&opts, EcdasForm::Bytes, "ECDAS")),
        "cdd4c4e6927f9f89b22c5e150e79499d7e2fe6ebf7a125f46913222c9619c4b4"
    );
    assert_eq!(EcdasForm::default(), EcdasForm::Bytes);
    assert_eq!(
        crate::block_whir::BlockFormat::pinned().ecdas,
        EcdasForm::Bytes
    );
    assert_eq!(
        crate::block_whir::BlockFormat::production().ecdas,
        EcdasForm::Bytes
    );
}

/// `BLOCK_WHIR_ECDAS`: unset is the pinned preset's form, `halves` and
/// `bytes` name a form, anything else is refused.
#[test]
fn the_ecdas_knob_reads_halves_or_bytes() {
    use crate::lfm::whir_block_tree::ecdas_setting;
    assert_eq!(
        ecdas_setting(None),
        Ok(crate::block_whir::BlockFormat::pinned().ecdas)
    );
    assert_eq!(ecdas_setting(Some("halves")), Ok(EcdasForm::Halves));
    assert_eq!(ecdas_setting(Some(" bytes ")), Ok(EcdasForm::Bytes));
    for bad in ["", "1", "on", "Halves", "half", "limbs"] {
        assert!(ecdas_setting(Some(bad)).is_err(), "{bad:?}");
    }
}

/// The two forms' shapes, each its own AIR prototype however they are built
/// in one process, both named `ECDAS`.
#[test]
fn the_two_forms_shapes_and_prototypes() {
    let opts = stark::proof::options::ProofOptions::default_test_options();
    for form in [
        EcdasForm::Halves,
        EcdasForm::Bytes,
        EcdasForm::Halves,
        EcdasForm::Bytes,
    ] {
        let air = create_ecdas_air_for(&opts, form, "ECDAS");
        let (columns, constraints, interactions) = match form {
            EcdasForm::Bytes => (521, 200, 388),
            EcdasForm::Halves => (515, 101, 242),
        };
        assert_eq!(air.trace_layout().0, columns, "{form:?}");
        assert_eq!(form.num_columns(), columns, "{form:?}");
        assert_eq!(air.bus_interactions().len(), interactions, "{form:?}");
        assert_eq!(bus_interactions_for(form).len(), interactions, "{form:?}");
        let meta = match form {
            EcdasForm::Bytes => EcdasConstraints.meta().len(),
            EcdasForm::Halves => EcdasHalvesConstraints.meta().len(),
        };
        assert_eq!(meta, constraints, "{form:?}");
        assert_eq!(form.num_constraints(), constraints, "{form:?}");
        assert_eq!(air.name(), "ECDAS", "{form:?}");
    }
    assert_eq!(EcdasHalvesConstraints.max_degree(), 3);
    assert_eq!(EcdasConstraints.max_degree(), 3);
    assert_eq!(create_ecdas_halves_air(&opts).name(), "ECDAS");
}

/// The column an interaction value reads, if it is one column.
fn column(v: &BusValue) -> Option<usize> {
    match v {
        BusValue::Packed { start_column, .. } => Some(*start_column),
        _ => None,
    }
}

fn is_zero_constant(v: &BusValue) -> bool {
    v.combine_from(|_| FE::from(7u64)) == vec![FE::zero()]
}

/// The byte form's single checks, the column each checks, mapped to the limb
/// form's layout (the arrays after `Q0` moved).
fn byte_forms_checked_columns_in_halves_layout() -> Vec<usize> {
    let to_halves = |c: usize| match c {
        c if (cols::Q1..cols::Q1 + 33).contains(&c) => h::Q1 + (c - cols::Q1),
        c if (cols::Q2..cols::Q2 + 33).contains(&c) => h::Q2 + (c - cols::Q2),
        c if c < cols::C0 => c,
        c => panic!("the byte form checks column {c} as a byte"),
    };
    bus_interactions_for(EcdasForm::Bytes)
        .iter()
        .filter(|i| i.bus_id == u64::from(BusId::AreBytes))
        .map(|i| {
            assert_eq!(i.values.len(), 2);
            assert!(is_zero_constant(&i.values[1]), "a single byte check");
            to_halves(column(&i.values[0]).expect("a byte column"))
        })
        .collect()
}

/// ★ The limb form checks exactly the byte form's 196 bytes, each once, two
/// to a lookup; every `LO` once on `IS_HALF`; every `HI` once on
/// `ARE_BYTES`, two to a lookup and the last with the constant 0; every
/// check at the row's `μ`. The bus receive and send and the `Bit` send are the
/// byte form's, at the moved `NEXT_OP`.
#[test]
fn the_limb_forms_checks_are_the_byte_forms_paired_and_the_carries_split() {
    let halves = bus_interactions_for(EcdasForm::Halves);
    let mut bytes = Vec::new();
    let mut his = Vec::new();
    let mut los = Vec::new();
    let hi_cols = |c: usize| {
        h::RELATIONS
            .iter()
            .any(|&(_, hi, _)| (hi..hi + 31).contains(&c))
    };
    let mut zero_partners = 0;
    for i in &halves {
        let bus = BusId::try_from(i.bus_id).expect("a bus");
        match bus {
            BusId::AreBytes => {
                assert!(i.is_sender);
                assert!(matches!(i.multiplicity, Multiplicity::Column(c) if c == h::MU));
                let a = column(&i.values[0]).expect("a column");
                let b = column(&i.values[1]);
                if hi_cols(a) {
                    his.push(a);
                    match b {
                        Some(b) => {
                            assert!(hi_cols(b));
                            his.push(b);
                        }
                        None => {
                            assert!(is_zero_constant(&i.values[1]));
                            zero_partners += 1;
                        }
                    }
                } else {
                    bytes.push(a);
                    bytes.push(b.expect("a byte pairs with a byte"));
                }
            }
            BusId::IsHalfword => {
                assert!(matches!(i.multiplicity, Multiplicity::Column(c) if c == h::MU));
                los.push(column(&i.values[0]).expect("a column"));
            }
            BusId::Ecdas | BusId::Bit => {}
            other => panic!("ECDAS sends to {other:?}"),
        }
    }
    assert_eq!(bytes, halves_checked_bytes());
    let mut sorted = bytes.clone();
    sorted.sort_unstable();
    let mut want = byte_forms_checked_columns_in_halves_layout();
    assert_eq!(want.len(), 196);
    want.sort_unstable();
    assert_eq!(sorted, want, "the byte form's checked bytes, each once");
    let mut want_los: Vec<usize> = h::RELATIONS
        .iter()
        .flat_map(|&(lo, _, _)| lo..lo + HALVES_CARRIES)
        .collect();
    los.sort_unstable();
    want_los.sort_unstable();
    assert_eq!(los, want_los);
    let mut want_his: Vec<usize> = h::RELATIONS
        .iter()
        .flat_map(|&(_, hi, _)| hi..hi + HALVES_CARRIES)
        .collect();
    his.sort_unstable();
    want_his.sort_unstable();
    assert_eq!(his, want_his);
    assert_eq!(zero_partners, 3, "each relation's 31st HI pairs with 0");

    // The bus and Bit interactions are the byte form's, with NEXT_OP moved.
    let key = |i: &BusInteraction, next_op: usize| {
        format!(
            "{} {} {:?} {:?}",
            i.bus_id, i.is_sender, i.multiplicity, i.values
        )
        .replace(&format!("{next_op}"), "NEXT_OP")
    };
    let others = |form: EcdasForm, next_op: usize, mu: usize| -> Vec<String> {
        bus_interactions_for(form)
            .iter()
            .filter(|i| i.bus_id == u64::from(BusId::Ecdas) || i.bus_id == u64::from(BusId::Bit))
            .map(|i| key(i, next_op).replace(&format!("Column({mu})"), "Column(MU)"))
            .collect()
    };
    assert_eq!(
        others(EcdasForm::Bytes, cols::NEXT_OP, cols::MU),
        others(EcdasForm::Halves, h::NEXT_OP, h::MU)
    );
}

// =============================================================================
// The constraint program's size
// =============================================================================

/// The fused steps of an AIR's constraint program, as the argue lowers it
/// at 2^17 rows (`multilinear_table::prove`).
fn fused_steps(
    air: &dyn AIR<Field = GoldilocksField, FieldExtension = GoldilocksExtension, PublicInputs = ()>,
) -> (usize, usize) {
    use stark::multilinear_air::Uniforms;
    use stark::multilinear_table::TableLayout;
    let layout = TableLayout::<GoldilocksField, GoldilocksExtension>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        air.trace_layout().0,
        17,
        Uniforms::default(),
    )
    .expect("the layout");
    let shape = layout.shape();
    let constraints = multilinear::fused::Constraints::<GoldilocksField>::from_extension(
        &shape.steps_as_ops(),
        shape.root_steps(),
        shape.selector_of_root(),
        shape.degree(),
    )
    .expect("a base-field program");
    (shape.steps_as_ops().len(), constraints.steps().len())
}

/// ★ The limb form's constraint program against the byte form's: the fused
/// steps 21,133 → 4,500…7,500, the degree unchanged.
#[test]
fn the_limb_forms_program_is_a_quarter_of_the_byte_forms() {
    let opts = stark::proof::options::ProofOptions::default_test_options();
    let (bytes_ops, bytes_steps) = fused_steps(&create_ecdas_air(&opts));
    let (halves_ops, halves_steps) = fused_steps(&create_ecdas_halves_air(&opts));
    let nodes = |a: &dyn AIR<
        Field = GoldilocksField,
        FieldExtension = GoldilocksExtension,
        PublicInputs = (),
    >| a.constraint_program().nodes.len();
    eprintln!(
        "ECDAS FUSED STEPS: bytes {bytes_steps} (shape ops {bytes_ops}, IR nodes {}), halves \
         {halves_steps} (shape ops {halves_ops}, IR nodes {}), ratio {:.3}",
        nodes(&create_ecdas_air(&opts)),
        nodes(&create_ecdas_halves_air(&opts)),
        halves_steps as f64 / bytes_steps as f64
    );
    assert_eq!(bytes_steps, 21_133, "the byte form's measured steps");
    assert!(
        (4_500..=7_500).contains(&halves_steps),
        "the limb form's steps {halves_steps}"
    );
}

// =============================================================================
// Rows, the constraints and the checks
// =============================================================================

/// Every constraint of `form` on one row (the prover folder). The limb form
/// with `top` reads each relation's `C_31` from its own column (a mutant).
fn eval_row(form: EcdasForm, row: &[FE], top: Option<usize>) -> Vec<FE> {
    let n = form.num_constraints();
    let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
        vec![row.to_vec()],
        vec![vec![]],
    )]);
    let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
    let offset_e = FieldElement::<GoldilocksExtension>::zero();
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
    let mut base = vec![FE::zero(); n];
    let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
    match form {
        EcdasForm::Bytes => EcdasConstraints.eval(&mut folder),
        EcdasForm::Halves => match top {
            None => EcdasHalvesConstraints.eval(&mut folder),
            Some(first) => EcdasHalvesConstraints::eval_with_top_columns(&mut folder, first),
        },
    }
    base
}

/// What a verifier would refuse a row for: a constraint that does not vanish,
/// or a range check whose value is not in its table. The Ecdas and Bit buses
/// balance against other rows and are not read here.
fn refusal(
    form: EcdasForm,
    interactions: &[BusInteraction],
    row: &[FE],
    top: Option<usize>,
) -> Option<String> {
    let failing: Vec<usize> = eval_row(form, row, top)
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != FE::zero())
        .map(|(i, _)| i)
        .collect();
    if !failing.is_empty() {
        return Some(format!("constraints {failing:?}"));
    }
    for (n, i) in interactions.iter().enumerate() {
        let mult = match i.multiplicity {
            Multiplicity::Column(c) => row[c],
            ref other => panic!("ECDAS has no {other:?} multiplicity"),
        };
        if mult == FE::zero() {
            continue;
        }
        let values: Vec<u64> = i
            .values
            .iter()
            .flat_map(|v| v.combine_from(|c| row[c]))
            .map(|v| v.canonical())
            .collect();
        let bound = match BusId::try_from(i.bus_id).expect("a bus") {
            BusId::AreBytes => 1 << 8,
            BusId::IsHalfword => 1 << 16,
            _ => continue,
        };
        if values.iter().any(|&v| v >= bound) {
            return Some(format!("interaction {n} {values:?}"));
        }
    }
    None
}

fn row_of(trace: &Trace, r: usize) -> Vec<FE> {
    (0..trace.main_table.width)
        .map(|c| *trace.main_table.get(r, c))
        .collect()
}

/// Both forms' tables of `steps` hold every constraint and every check on
/// every row, padding rows included; the limb form's carries are exactly its
/// relations' integer chains.
fn assert_both_forms_hold(steps: &[EcdasStep], label: &str) {
    let ops: Vec<EcdasOperation> = steps
        .iter()
        .map(|s| EcdasOperation {
            timestamp: 444,
            step: s.clone(),
        })
        .collect();
    let rows = ops.len().next_power_of_two().max(4) + 4;
    let rows = rows.next_power_of_two();
    for form in [EcdasForm::Bytes, EcdasForm::Halves] {
        let trace = generate_ecdas_rows_in(&ops, rows, TraceForm::Wide, form, |o| {
            std::borrow::Cow::Borrowed(o)
        });
        let interactions = bus_interactions_for(form);
        for r in 0..trace.num_rows() {
            let row = row_of(&trace, r);
            assert_eq!(
                refusal(form, &interactions, &row, None),
                None,
                "{label} {form:?} row {r}"
            );
        }
    }
    for s in steps {
        for t in HalvesLimbs::of(s).terms() {
            assert!(halves_chain(&t).1, "{label}: an exact chain");
        }
    }
}

fn gx_le() -> [u8; 32] {
    let mut be = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    be.reverse();
    be
}

fn steps_for(k_le: &[u8; 32]) -> Vec<EcdasStep> {
    ecsm::compute_witness(k_le, &gx_le())
        .expect("a witness")
        .steps
}

fn k_le(v: u64) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&v.to_le_bytes());
    k
}

/// ★ ECSM's steps over many scalars (doubles and adds), a near-order
/// scalar's full ladder, and an ECDSA recovery's two chains and addition hold
/// in both forms.
#[test]
fn real_steps_hold_in_both_forms() {
    for k in [2u64, 3, 5, 7, 0xFF, 0xABCD, 1_000_003, u64::MAX] {
        assert_both_forms_hold(&steps_for(&k_le(k)), &format!("k={k}"));
    }
    let mut near = ecsm::N_BYTES;
    near[0] -= 1;
    assert_both_forms_hold(&steps_for(&near), "N-1");
    let mut sig = [0u8; 64];
    let hex32 = |s: &str| -> [u8; 32] {
        std::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
    };
    sig[..32].copy_from_slice(&hex32(
        "3284bae4a1b693dfbe8ac67400e4891f3b53681c5159a315561051f3d2c57139",
    ));
    sig[32..].copy_from_slice(&hex32(
        "97d4b2f2f728a54ed562a12237f6b11d32d626e7fcc24c3470b0729e095bd057",
    ));
    let msg = hex32("c0ffee0011223344556677889900aabbccddeeff00112233445566778899aabb");
    let w = ecsm::compute_ecrec_witness(&sig, &msg, 0);
    let steps: Vec<EcdasStep> = w
        .chain1
        .iter()
        .chain(&w.chain2)
        .chain(w.add.as_ref())
        .cloned()
        .collect();
    assert!(steps.len() > 400, "{} steps", steps.len());
    assert_both_forms_hold(&steps, "ECREC");
}

// =============================================================================
// The carry bound, computed
// =============================================================================

/// Each relation's `S'_k` range over every byte value of every operand (and
/// `op`, `μ` bits): `(min, max)` a position.
fn term_ranges() -> [[(i64, i64); HALVES_POSITIONS]; 3] {
    const M: i64 = 0xFFFF;
    let p: [i64; 16] =
        std::array::from_fn(|i| i64::from(P_BYTES[2 * i]) + 256 * i64::from(P_BYTES[2 * i + 1]));
    let r: [i64; 17] = std::array::from_fn(|i| {
        i64::from(R_BYTES[2 * i])
            + if 2 * i + 1 < 33 {
                256 * i64::from(R_BYTES[2 * i + 1])
            } else {
                0
            }
    });
    let pairs = |k: usize| (k.saturating_sub(15)..=k.min(15)).count() as i64;
    let lin = |k: usize| if k < 16 { M } else { 0 };
    let rp = |k: usize| -> i64 {
        (k.saturating_sub(15)..=k.min(16))
            .map(|j| r[j] * p[k - j])
            .sum()
    };
    let qp = |k: usize| -> i64 {
        (k.saturating_sub(15)..=k.min(16))
            .map(|j| if j < 16 { M } else { 255 } * p[k - j])
            .sum()
    };
    let mut out = [[(0, 0); HALVES_POSITIONS]; 3];
    #[allow(clippy::needless_range_loop)]
    for k in 0..HALVES_POSITIONS {
        let c = pairs(k) * M * M;
        // op = 1: Σ λ·Δx in [−c, c], + yA − yG; op = 0: 2·Σ λ·yA − 3·Σ xA·xA.
        let add = (-c - lin(k), c + lin(k));
        let dbl = (-3 * c, 2 * c);
        out[0][k] = (add.0.min(dbl.0) - qp(k), add.1.max(dbl.1) + rp(k));
        // Σ λ·λ − (xA + xG + xR) or − (2·xA + xR).
        out[1][k] = (-3 * lin(k) - qp(k), c + rp(k));
        // Σ λ·(xA − xR) − yA − yR.
        out[2][k] = (-c - 2 * lin(k) - qp(k), c + rp(k));
    }
    out
}

/// ★ The carry bound, computed over every byte value: every carry a step can
/// have lies in `[−2^23, 2^23)` (so `C + 2^23` is a 16-bit `LO` and an 8-bit
/// `HI`), with a bit to spare; every `|S'_k| < 2^39`; and every constraint's
/// terms stay below half the field, so each is an equality over the
/// integers. Real steps' carries lie inside the computed ranges.
#[test]
fn the_carry_bound_is_computed() {
    let ranges = term_ranges();
    let mut worst_term = 0i64;
    let mut worst_carry = [0i64; 3];
    for (rel, terms) in ranges.iter().enumerate() {
        let (mut lo, mut hi) = (0i64, 0i64);
        for (k, &(smin, smax)) in terms.iter().enumerate() {
            worst_term = worst_term.max(smin.abs()).max(smax.abs());
            if k == HALVES_CARRIES {
                break;
            }
            lo = (lo + smin).div_euclid(1 << 16);
            hi = (hi + smax).div_euclid(1 << 16);
            worst_carry[rel] = worst_carry[rel].max(-lo).max(hi);
            assert!(
                -HALVES_CARRY_OFFSET <= lo && hi < HALVES_CARRY_OFFSET,
                "relation {rel} position {k}: [{lo}, {hi}]"
            );
            assert!(
                -HALVES_CARRY_OFFSET / 2 <= lo && hi < HALVES_CARRY_OFFSET / 2,
                "relation {rel} position {k}: no bit to spare"
            );
        }
    }
    eprintln!(
        "ECDAS HALVES BOUND: max |S'| = 2^{:.3}, max |C| (λ, xR, yR) = {worst_carry:?}",
        (worst_term as f64).log2()
    );
    assert!(worst_term < 1 << 39);
    // No wrap: 2^16·|C_k| + |C_{k−1}| + |S'_k| < (p − 1) / 2.
    let most = (1u128 << 16) * (1 << 23) + (1 << 23) + worst_term as u128;
    assert!(most < u128::from(GOLDILOCKS / 2));

    // Real steps lie inside.
    for k in [3u64, 0xABCD, u64::MAX] {
        for s in steps_for(&k_le(k)) {
            for (rel, t) in HalvesLimbs::of(&s).terms().iter().enumerate() {
                for (k, &v) in t.iter().enumerate() {
                    assert!(ranges[rel][k].0 <= v && v <= ranges[rel][k].1);
                }
            }
        }
    }
}

// =============================================================================
// Steps over the integers: random honest steps, and the forgeries
// =============================================================================

fn p_big() -> BigInt {
    BigInt::from(ecsm::p())
}

fn r_big() -> BigInt {
    BigInt::from_bytes_le(num_bigint::Sign::Plus, &R_BYTES)
}

/// `v`'s `N` little-endian bytes as integers; `v` must be in `[0, 2^(8N))`.
fn bytes_of<const N: usize>(v: &BigInt) -> [u64; N] {
    assert!(!v.is_negative(), "{v}");
    let (_, b) = v.to_bytes_le();
    assert!(b.len() <= N, "{v} past {N} bytes");
    std::array::from_fn(|i| u64::from(b.get(i).copied().unwrap_or(0)))
}

fn big_of(bytes: &[u64]) -> BigInt {
    bytes
        .iter()
        .rev()
        .fold(BigInt::zero(), |acc, &b| (acc << 8) + BigInt::from(b))
}

/// A row's operands as integer bytes (a test may put one out of `0..256`).
#[derive(Clone, Debug)]
struct Operands {
    xg: [u64; 32],
    yg: [u64; 32],
    xa: [u64; 32],
    ya: [u64; 32],
    xr: [u64; 32],
    yr: [u64; 32],
    lambda: [u64; 32],
    q: [[u64; 33]; 3],
    op: u64,
    round: u64,
}

impl Operands {
    /// A step from its integers, `λ` given: `op = 1` adds (`yG` is made to fit
    /// `λ`), `op = 0` doubles (`yA` is made to fit `λ`); `xR`, `yR` and the
    /// quotients follow, every relation exact over the integers.
    fn honest(op: u64, xa: &BigInt, ya: &BigInt, xg: &BigInt, lambda: &BigInt) -> Self {
        Self::honest_with(op, xa, ya, xg, lambda, bytes_of(lambda))
    }

    /// [`Self::honest`] with `λ`'s bytes as given (their integer must be `λ`;
    /// one may be past `0..256`).
    fn honest_with(
        op: u64,
        xa: &BigInt,
        ya: &BigInt,
        xg: &BigInt,
        lambda: &BigInt,
        lambda_bytes: [u64; 32],
    ) -> Self {
        assert_eq!(&big_of(&lambda_bytes), lambda);
        let p = p_big();
        let r = r_big();
        let m = |v: BigInt| v.mod_floor(&p);
        let (ya, yg) = if op == 1 {
            (ya.clone(), m(lambda * (xg - xa) + ya))
        } else {
            let two_l = m(lambda * 2);
            let inv = two_l.modpow(&(&p - 2), &p);
            (m(xa * xa * 3 * inv), BigInt::zero())
        };
        let quotient = |num: BigInt| -> BigInt {
            let (q, rem) = num.div_rem(&p);
            assert!(rem.is_zero(), "an exact relation");
            &r + q
        };
        let q0 = if op == 1 {
            quotient(lambda * (xg - xa) + &ya - &yg)
        } else {
            quotient(lambda * &ya * 2 - xa * xa * 3)
        };
        let lin = if op == 1 { xa + xg } else { xa * 2 };
        let xr = m(lambda * lambda - &lin);
        let q1 = quotient(lambda * lambda - &lin - &xr);
        let yr = m(lambda * (xa - &xr) - &ya);
        let q2 = quotient(lambda * (xa - &xr) - &ya - &yr);
        Self {
            xg: bytes_of(xg),
            yg: bytes_of(&yg),
            xa: bytes_of(xa),
            ya: bytes_of(&ya),
            xr: bytes_of(&xr),
            yr: bytes_of(&yr),
            lambda: lambda_bytes,
            q: [bytes_of(&q0), bytes_of(&q1), bytes_of(&q2)],
            op,
            round: 17,
        }
    }

    fn limbs(&self) -> HalvesLimbs {
        let l16 = |b: &[u64; 32]| -> [i64; 16] {
            std::array::from_fn(|i| (b[2 * i] + 256 * b[2 * i + 1]) as i64)
        };
        let l17 = |b: &[u64; 33]| -> [i64; 17] {
            std::array::from_fn(|i| {
                (b[2 * i]
                    + if 2 * i + 1 < 33 {
                        256 * b[2 * i + 1]
                    } else {
                        0
                    }) as i64
            })
        };
        HalvesLimbs {
            xg: l16(&self.xg),
            yg: l16(&self.yg),
            xa: l16(&self.xa),
            ya: l16(&self.ya),
            xr: l16(&self.xr),
            yr: l16(&self.yr),
            lambda: l16(&self.lambda),
            q: [l17(&self.q[0]), l17(&self.q[1]), l17(&self.q[2])],
            op: self.op as i64,
            mu: 1,
        }
    }

    /// The limb form's row: these operands and `carries` (`LO`, `HI` a
    /// relation), with `extra` columns past the table's (zero).
    fn row(&self, carries: &[[(FE, FE); HALVES_CARRIES]; 3], extra: usize) -> Vec<FE> {
        let mut row = vec![FE::zero(); h::NUM_COLUMNS + extra];
        let put = |row: &mut [FE], start: usize, b: &[u64]| {
            for (i, &v) in b.iter().enumerate() {
                row[start + i] = FE::from(v);
            }
        };
        put(&mut row, h::XG, &self.xg);
        put(&mut row, h::YG, &self.yg);
        put(&mut row, h::XA, &self.xa);
        put(&mut row, h::YA, &self.ya);
        put(&mut row, h::XR, &self.xr);
        put(&mut row, h::YR, &self.yr);
        put(&mut row, h::LAMBDA, &self.lambda);
        for (&(lo, hi, q), (qb, c)) in h::RELATIONS.iter().zip(self.q.iter().zip(carries)) {
            put(&mut row, q, qb);
            for (k, &(l, high)) in c.iter().enumerate() {
                row[lo + k] = l;
                row[hi + k] = high;
            }
        }
        row[h::ROUND] = FE::from(self.round);
        row[h::OP] = FE::from(self.op);
        row[h::TIMESTAMP_0] = FE::from(444u64);
        row[h::MU] = FE::one();
        row
    }

    /// The integer carries' `(LO, HI)`, each relation's chain asserted exact.
    fn integer_carries(&self) -> [[(FE, FE); HALVES_CARRIES]; 3] {
        self.limbs().terms().map(|t| {
            let (c, exact) = halves_chain(&t);
            assert!(exact, "an exact chain");
            c.map(|c| {
                let (l, high) = split_carry(c);
                (FE::from(l), fe(high))
            })
        })
    }

    /// The honest row.
    fn honest_row(&self) -> Vec<FE> {
        self.row(&self.integer_carries(), 0)
    }
}

fn fe(v: i64) -> FE {
    if v >= 0 {
        FE::from(v as u64)
    } else {
        -FE::from(v.unsigned_abs())
    }
}

/// Deterministic SplitMix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value below `P`; every 16th one at an edge (0, 1, P − 1, P − 2^k).
    fn below_p(&mut self) -> BigInt {
        let p = p_big();
        match self.next() % 16 {
            0 => match self.next() % 4 {
                0 => BigInt::zero(),
                1 => BigInt::one(),
                2 => &p - 1,
                _ => &p - (BigInt::one() << (self.next() % 256) as usize),
            },
            _ => {
                let v = (0..4).fold(BigInt::zero(), |acc, _| {
                    (acc << 64) + BigInt::from(self.next())
                });
                v.mod_floor(&p)
            }
        }
    }
}

/// A random honest step: random operands below `P` (edges among them), a
/// random `λ` (not zero when it doubles), `op` either.
fn random_step(rng: &mut Rng) -> Operands {
    let op = rng.next() % 2;
    let (xa, ya, xg) = (rng.below_p(), rng.below_p(), rng.below_p());
    let mut lambda = rng.below_p();
    if op == 0 && lambda.is_zero() {
        lambda = BigInt::one();
    }
    Operands::honest(op, &xa, &ya, &xg, &lambda)
}

/// ★ 100,000 random honest steps (edges among their operands): every chain
/// exact and every carry inside the computed bound (so inside its checks);
/// 2,000 of them as rows, every constraint and check holding.
#[test]
fn random_honest_steps_hold() {
    let interactions = bus_interactions_for(EcdasForm::Halves);
    let ranges = term_ranges();
    let mut rng = Rng(0xEC_DA5);
    let mut most = [0i64; 3];
    for n in 0..100_000 {
        let s = random_step(&mut rng);
        for (rel, t) in s.limbs().terms().iter().enumerate() {
            for (k, &v) in t.iter().enumerate() {
                assert!(ranges[rel][k].0 <= v && v <= ranges[rel][k].1, "{n}");
            }
            let (c, exact) = halves_chain(t);
            assert!(exact, "{n}");
            for c in c {
                most[rel] = most[rel].max(c.abs());
                let (_, high) = split_carry(c);
                assert!((0..256).contains(&high), "{n}: carry {c}");
            }
        }
        if n < 2_000 {
            assert_eq!(
                refusal(EcdasForm::Halves, &interactions, &s.honest_row(), None),
                None,
                "{n}"
            );
        }
    }
    eprintln!("ECDAS HALVES RANDOM: max |C| (λ, xR, yR) = {most:?}");
}

/// `interactions` less those `drop` names.
fn without(
    interactions: &[BusInteraction],
    drop: impl Fn(&BusInteraction) -> bool,
) -> Vec<BusInteraction> {
    let kept: Vec<BusInteraction> = interactions.iter().filter(|i| !drop(i)).cloned().collect();
    assert!(
        kept.len() < interactions.len(),
        "the mutant removes something"
    );
    kept
}

fn on(bus: BusId) -> impl Fn(&BusInteraction) -> bool {
    move |i| i.bus_id == u64::from(bus)
}

fn are_bytes_on(cols: &[usize]) -> impl Fn(&BusInteraction) -> bool + '_ {
    move |i| {
        i.bus_id == u64::from(BusId::AreBytes)
            && i.values
                .iter()
                .filter_map(column)
                .any(|c| cols.contains(&c))
    }
}

/// A row whose relations hold in the field but not over the integers: `λ`
/// one more than the true slope's, each relation's quotient chosen so its
/// integer total is a nonzero multiple of the field's prime, and each carry
/// solved in the field, `C_k = (C_{k−1} + S'_k) / 2^16`. `put` lays each
/// carry out as `(LO, HI)`.
fn field_solved(put: impl Fn(FE) -> (FE, FE)) -> (Operands, Vec<FE>) {
    let step = &steps_for(&k_le(0xABCD))[5];
    let mut s = Operands {
        xg: step.x_g.map(u64::from),
        yg: step.y_g.map(u64::from),
        xa: step.x_a.map(u64::from),
        ya: step.y_a.map(u64::from),
        xr: step.x_r.map(u64::from),
        yr: step.y_r.map(u64::from),
        lambda: step.lambda.map(u64::from),
        q: [[0; 33]; 3],
        op: u64::from(step.op),
        round: u64::from(step.round),
    };
    s.lambda = bytes_of(&(big_of(&s.lambda) + 1));
    let shift = FE::from(1u64 << 16);
    let inv = shift.inv().expect("2^16 is invertible");
    // The integer total of `terms` in the field: Σ 2^{16k}·S'_k.
    let total = |t: &[i64; HALVES_POSITIONS]| {
        t.iter()
            .rev()
            .fold(FE::zero(), |acc, &v| acc * shift + fe(v))
    };
    let p_in_field = big_of(&P_BYTES.map(u64::from)).mod_floor(&BigInt::from(GOLDILOCKS));
    let p_in_field = FE::from(u64::try_from(p_in_field).expect("below the prime"));
    // With q = 0: A_r; then q_r = A_r / P in the field makes the total vanish.
    let free = s.limbs().terms();
    for (rel, t) in free.iter().enumerate() {
        let q = total(t) * p_in_field.inv().expect("P is not a multiple of the prime");
        let mut b = [0u64; 33];
        b[..8].copy_from_slice(&q.canonical().to_le_bytes().map(u64::from));
        s.q[rel] = b;
    }
    let terms = s.limbs().terms();
    let mut carries = [[(FE::zero(), FE::zero()); HALVES_CARRIES]; 3];
    for (rel, t) in terms.iter().enumerate() {
        assert_eq!(total(t), FE::zero(), "a multiple of the prime");
        assert!(!halves_chain(t).1, "not a relation over the integers");
        let mut c = FE::zero();
        for (k, out) in carries[rel].iter_mut().enumerate() {
            c = (c + fe(t[k])) * inv;
            *out = put(c);
        }
    }
    let row = s.row(&carries, 0);
    (s, row)
}

/// ★ T1. Carries solved in the field, each `HI` 0 and `LO = C + 2^23`: every
/// constraint holds, and the `LO` checks refuse it; without them (the
/// control) the row is accepted.
#[test]
fn field_solved_carries_are_refused_by_the_lo_checks() {
    let offset = fe(HALVES_CARRY_OFFSET);
    let (_, row) = field_solved(|c| (c + offset, FE::zero()));
    let all = bus_interactions_for(EcdasForm::Halves);
    let why = refusal(EcdasForm::Halves, &all, &row, None).expect("refused");
    assert!(why.starts_with("interaction"), "{why}");
    let control = without(&all, on(BusId::IsHalfword));
    assert_eq!(refusal(EcdasForm::Halves, &control, &row, None), None);
}

/// ★ T2. The same forgery with every `LO` 0 and `HI = (C + 2^23) / 2^16`:
/// refused by the `HI` checks; without them (the control) accepted.
#[test]
fn field_solved_carries_with_lo_in_range_are_refused_by_the_hi_checks() {
    let offset = fe(HALVES_CARRY_OFFSET);
    let inv = FE::from(1u64 << 16).inv().expect("invertible");
    let (_, row) = field_solved(|c| (FE::zero(), (c + offset) * inv));
    let all = bus_interactions_for(EcdasForm::Halves);
    assert!(refusal(EcdasForm::Halves, &all, &row, None).is_some());
    let his: Vec<usize> = h::RELATIONS
        .iter()
        .flat_map(|&(_, hi, _)| hi..hi + HALVES_CARRIES)
        .collect();
    let control = without(&all, are_bytes_on(&his));
    assert_eq!(control.len(), all.len() - 48);
    assert_eq!(refusal(EcdasForm::Halves, &control, &row, None), None);
}

/// A step whose relation `rel` (0 λ, 1 xR, 2 yR) is off by exactly `2^512`
/// and whose other two hold: λ's by `yG` and `q0`, xR's by `xR` and `q1`, yR's
/// by `yR` and `q2`, each pair the quotient and remainder by `P` of the total
/// less `2^512`. The relation's value is then `2^512 ≢ 0 mod P`: λ is not the
/// slope, or `xR`, `yR` are not the sum's.
fn off_by_two_to_the_512(rel: usize) -> Operands {
    let p = p_big();
    let r = r_big();
    let honest = steps_for(&k_le(0xABCD));
    let step = honest.iter().find(|s| s.op == 1).expect("an add");
    let big = |b: &[u8]| big_of(&b.iter().map(|&v| u64::from(v)).collect::<Vec<_>>());
    let (xa, ya, xg) = (big(&step.x_a), big(&step.y_a), big(&step.x_g));
    let two_512: BigInt = BigInt::one() << 512;
    let mut lambda = big(&step.lambda);
    if rel == 0 {
        lambda += 1;
    }
    let mut s = Operands::honest(1, &xa, &ya, &xg, &lambda);
    match rel {
        0 => {
            // λ·Δx + yA + R·P − 2^512 = yG + q0·P.
            let x: BigInt = &lambda * (&xg - &xa) + &ya + &r * &p - &two_512;
            let (q0, yg) = x.div_mod_floor(&p);
            s.yg = bytes_of(&yg);
            s.q[0] = bytes_of(&q0);
        }
        1 => {
            // λ² − xA − xG + R·P − 2^512 = xR + q1·P; yR's relation re-made for that xR.
            let x: BigInt = &lambda * &lambda - &xa - &xg + &r * &p - &two_512;
            let (q1, xr) = x.div_mod_floor(&p);
            let y: BigInt = &lambda * (&xa - &xr) - &ya;
            let (q2, yr) = y.div_mod_floor(&p);
            s.xr = bytes_of(&xr);
            s.q[1] = bytes_of(&q1);
            s.yr = bytes_of(&yr);
            s.q[2] = bytes_of(&(&r + q2));
        }
        _ => {
            // λ·(xA − xR) − yA + R·P − 2^512 = yR + q2·P.
            let xr = big_of(&s.xr);
            let y: BigInt = &lambda * (&xa - &xr) - &ya + &r * &p - &two_512;
            let (q2, yr) = y.div_mod_floor(&p);
            s.yr = bytes_of(&yr);
            s.q[2] = bytes_of(&q2);
        }
    }
    let t = s.limbs().terms();
    for (k, t) in t.iter().enumerate() {
        let total = t
            .iter()
            .rev()
            .fold(BigInt::zero(), |acc, &v| (acc << 16) + BigInt::from(v));
        let want = if k == rel {
            two_512.clone()
        } else {
            BigInt::zero()
        };
        assert_eq!(total, want, "relation {k}'s total");
    }
    assert_ne!(two_512.mod_floor(&p), BigInt::zero());
    s
}

/// ★ T3, each relation. A relation off by exactly `2^512`
/// ([`off_by_two_to_the_512`]): every carry below the top is an in-range
/// integer and the top carry is 1. The closing rule `C_31 = 0` refuses it,
/// the one identity at the relation's top failing (36 λ, 68 xR, 100 yR); the
/// mutant that keeps a `C_31` column (the control) accepts it with that
/// column 1. Every check passes either way.
#[test]
fn a_relation_off_by_two_to_the_512_is_refused_by_the_closing_rule() {
    let all = bus_interactions_for(EcdasForm::Halves);
    for rel in 0..3 {
        let s = off_by_two_to_the_512(rel);
        let t = s.limbs().terms();
        let mut carries = [[(FE::zero(), FE::zero()); HALVES_CARRIES]; 3];
        for (k, t) in t.iter().enumerate() {
            let (c, exact) = halves_chain(t);
            assert_eq!(exact, k != rel, "relation {k}'s chain");
            if k == rel {
                assert_eq!(c[HALVES_CARRIES - 1] + t[HALVES_POSITIONS - 1], 1 << 16);
            }
            for (j, &c) in c.iter().enumerate() {
                let (l, high) = split_carry(c);
                assert!((0..256).contains(&high));
                carries[k][j] = (FE::from(l), fe(high));
            }
        }
        let row = s.row(&carries, 0);
        let why = refusal(EcdasForm::Halves, &all, &row, None).expect("refused");
        assert_eq!(
            why,
            format!(
                "constraints [{}]",
                5 + HALVES_POSITIONS * rel + HALVES_POSITIONS - 1
            ),
            "relation {rel}"
        );
        // The mutant: each relation's C_31 a column of its own (this one's 1).
        let mut mutant = s.row(&carries, 3);
        mutant[h::NUM_COLUMNS + rel] = FE::one();
        assert_eq!(
            refusal(EcdasForm::Halves, &all, &mutant, Some(h::NUM_COLUMNS)),
            None,
            "relation {rel}: the control"
        );
    }
}

/// The failing constraints of the limb form on `row`.
fn failing(row: &[FE]) -> Vec<usize> {
    eval_row(EcdasForm::Halves, row, None)
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != FE::zero())
        .map(|(i, _)| i)
        .collect()
}

/// ★ Every carry identity is load-bearing, one at a time: on honest rows (an
/// add and a double), `LO_k + 1` of relation `r`, and the carry `C_k` moved by
/// ±1 and by ±2^22 (the gap between the honest range and the checks'),
/// re-split so every check passes, each breaks exactly the two identities
/// that read `C_k` — `k`'s (`2^16·C_k − C_{k−1} − S'_k`) and `k + 1`'s — at
/// `5 + 32r + k` and `5 + 32r + k + 1`. So a carry in range but off its chain
/// is refused, and an identity emitted as zero would leave its index out.
#[test]
fn each_carry_identity_reads_its_two_carries() {
    let mut rng = Rng(21);
    for op in [1, 0] {
        let s = loop {
            let s = random_step(&mut rng);
            if s.op == op {
                break s;
            }
        };
        let row = s.honest_row();
        assert!(failing(&row).is_empty(), "op {op}: the honest row");
        let all = bus_interactions_for(EcdasForm::Halves);
        let chains = s.limbs().terms().map(|t| halves_chain(&t).0);
        for (r, &(lo, hi, _)) in h::RELATIONS.iter().enumerate() {
            for k in 0..HALVES_CARRIES {
                let at = 5 + HALVES_POSITIONS * r + k;
                // LO_k + 1, as it stands.
                let mut bumped = row.clone();
                bumped[lo + k] += FE::one();
                assert_eq!(
                    failing(&bumped),
                    vec![at, at + 1],
                    "op {op} relation {r} carry {k}: LO + 1"
                );
                // C_k moved by ±1 and by ±2^22 (into the gap between the
                // honest range and the checks'), re-split: every check passes,
                // and the two identities that read C_k alone refuse it.
                for d in [1i64, -1, 1 << 22, -(1 << 22)] {
                    let c = chains[r][k] + d;
                    let (l, high) = split_carry(c);
                    assert!((0..256).contains(&high), "carry {c} within the checks");
                    let mut moved = row.clone();
                    moved[lo + k] = FE::from(l);
                    moved[hi + k] = fe(high);
                    assert_eq!(
                        refusal(EcdasForm::Halves, &all, &moved, None),
                        Some(format!("constraints [{at}, {}]", at + 1)),
                        "op {op} relation {r} carry {k}: C_k {d:+}"
                    );
                }
            }
        }
    }
}

/// ★ The five flag constraints of the limb form, each the only one (or among
/// those) to refuse its row: `NEXT_OP = 2` on a double → `[2]`; `NEXT_OP = 1`
/// on a padding row (op 0, every carry zero) → `[4]`; `OP = NEXT_OP = 1` → 3
/// among them; `OP = 2` → 1 among them; `MU = 2` → 0 among them.
#[test]
fn each_flag_constraint_refuses_its_row() {
    let mut rng = Rng(23);
    let step = |op: u64, rng: &mut Rng| loop {
        let s = random_step(rng);
        if s.op == op {
            break s;
        }
    };
    let double = step(0, &mut rng).honest_row();
    let add = step(1, &mut rng).honest_row();
    assert!(failing(&double).is_empty() && failing(&add).is_empty());
    // A padding row: everything zero but each carry's zero split; op 0.
    let mut padding = vec![FE::zero(); h::NUM_COLUMNS];
    let (_, hi_zero) = split_carry(0);
    for &(_, hi, _) in &h::RELATIONS {
        for k in 0..HALVES_CARRIES {
            padding[hi + k] = fe(hi_zero);
        }
    }
    assert!(failing(&padding).is_empty(), "the padding row holds");

    let mut row = double.clone();
    row[h::NEXT_OP] = FE::from(2u64);
    assert_eq!(failing(&row), vec![2], "NEXT_OP = 2 on a double");

    let mut row = padding.clone();
    row[h::NEXT_OP] = FE::one();
    assert_eq!(failing(&row), vec![4], "NEXT_OP = 1 on a padding row");

    let mut row = add.clone();
    row[h::NEXT_OP] = FE::one();
    assert!(failing(&row).contains(&3), "OP = NEXT_OP = 1");

    let mut row = add.clone();
    row[h::OP] = FE::from(2u64);
    assert!(failing(&row).contains(&1), "OP = 2");

    let mut row = add;
    row[h::MU] = FE::from(2u64);
    assert!(failing(&row).contains(&0), "MU = 2");
}

/// The limb form's captured program (constraints and LogUp emission), pinned
/// as built at the commit that added it. Moved by any change to the limb
/// form's constraint body or interactions.
#[test]
fn the_limb_forms_air_is_pinned() {
    let opts = stark::proof::options::ProofOptions::default_test_options();
    assert_eq!(
        program_digest(&create_ecdas_halves_air(&opts)),
        "f1bf6c4e32dba2a323b17780313069093061dc7606e6b7f38fadff6efafb8b09"
    );
}

/// The constants' check: the offset is what makes the honest carries fit.
/// Without it (`LO + 2^16·HI` read as the carry itself) an honest step with a
/// negative carry has no in-range `(LO, HI)`.
#[test]
fn the_offset_is_load_bearing_for_completeness() {
    let mut rng = Rng(7);
    let negative = |s: &Operands| {
        s.limbs()
            .terms()
            .iter()
            .flat_map(|t| halves_chain(t).0)
            .any(|c| c < 0)
    };
    let s = (0..10_000)
        .map(|_| random_step(&mut rng))
        .find(negative)
        .expect("a step with a negative carry");
    // `split_carry` with offset 0: some HI is negative.
    let unshifted = |c: i64| {
        let lo = c.rem_euclid(1 << 16);
        (c - lo) >> 16
    };
    assert!(
        s.limbs()
            .terms()
            .iter()
            .flat_map(|t| halves_chain(t).0)
            .any(|c| !(0..256).contains(&unshifted(c)))
    );
    // With it, every one fits.
    assert_eq!(
        refusal(
            EcdasForm::Halves,
            &bus_interactions_for(EcdasForm::Halves),
            &s.honest_row(),
            None
        ),
        None
    );
}

/// ★ T5. A byte of 256 or more: each sampled check is the only thing that
/// refuses it. λ's third byte (λ_2 + 256, λ_3 − 1: the same limbs, so every
/// constraint holds); `q0`'s 32nd byte (with the 33rd − 1); `ROUND`; λ's last
/// byte (λ + 2^256, a step whose relations hold for that `λ`); the outputs
/// xR_0, yR_0, yR_1 (across a limb) and xR + P (its top cell past 255);
/// `q1_4`; `q2_0`. The pairs that straddle two arrays, (q1_32, yR_0) and
/// (yR_31, q2_0), are among them.
/// Each with all checks is refused; without its one pair (the control),
/// accepted.
#[test]
fn each_sampled_byte_check_is_load_bearing() {
    let mut rng = Rng(11);
    let base = loop {
        let s = random_step(&mut rng);
        if s.op == 1
            && s.lambda[3] > 0
            && s.q[0][32] > 0
            && s.xr[1] > 0
            && s.yr[1] > 0
            && s.yr[2] > 0
            && s.q[1][5] > 0
            && s.q[2][1] > 0
            && big_of(&s.xr) + p_big() >= BigInt::one() << 256
        {
            break s;
        }
    };
    let all = bus_interactions_for(EcdasForm::Halves);
    let mut cases: Vec<(&str, usize, Vec<FE>)> = Vec::new();

    let mut s = base.clone();
    s.lambda[2] += 256;
    s.lambda[3] -= 1;
    cases.push(("λ_2", h::LAMBDA + 2, s.honest_row()));

    let mut s = base.clone();
    s.q[0][31] += 256;
    s.q[0][32] -= 1;
    cases.push(("q0_31", h::Q0 + 31, s.honest_row()));

    let mut s = base.clone();
    s.round = 256 + 17;
    cases.push(("ROUND", h::ROUND, s.honest_row()));

    // λ + 2^256, its last byte past 255: a step whose relations hold over
    // the integers for that λ (yG made to fit it).
    let mut bytes = base.lambda;
    bytes[31] += 256;
    let lambda = big_of(&bytes);
    let s = Operands::honest_with(
        1,
        &big_of(&base.xa),
        &big_of(&base.ya),
        &big_of(&base.xg),
        &lambda,
        bytes,
    );
    cases.push(("λ_31", h::LAMBDA + 31, s.honest_row()));

    // The outputs, which leave the row (to the next step and to memory): a
    // byte of xR and of yR re-encoded within the same integer, yR across a limb
    // boundary, and xR + P (q1 − 1, q2 − λ: the relations hold over the
    // integers), a non-canonical output past 2^256 in its top cell.
    let mut s = base.clone();
    s.xr[0] += 256;
    s.xr[1] -= 1;
    cases.push(("xR_0", h::XR, s.honest_row()));
    let mut s = base.clone();
    s.yr[0] += 256;
    s.yr[1] -= 1;
    cases.push(("yR_0 (the pair q1_32, yR_0)", h::YR, s.honest_row()));
    let mut s = base.clone();
    s.yr[1] += 256;
    s.yr[2] -= 1;
    cases.push(("yR_1 (across a limb)", h::YR + 1, s.honest_row()));
    let mut s = base.clone();
    let xr = big_of(&s.xr) + p_big();
    let low = bytes_of::<32>(&(&xr - ((&xr >> 248usize) << 248usize)));
    s.xr = std::array::from_fn(|i| {
        if i < 31 {
            low[i]
        } else {
            u64::try_from(&xr >> 248usize).expect("a small top cell")
        }
    });
    assert_eq!(big_of(&s.xr), xr);
    s.q[1] = bytes_of(&(big_of(&s.q[1]) - 1));
    s.q[2] = bytes_of(&(big_of(&s.q[2]) - big_of(&s.lambda)));
    cases.push(("xR + P (top cell)", h::XR + 31, s.honest_row()));
    // A quotient byte of xR's and of yR's relation, the latter in the pair
    // (yR_31, q2_0) that straddles two arrays.
    let mut s = base.clone();
    s.q[1][4] += 256;
    s.q[1][5] -= 1;
    cases.push(("q1_4", h::Q1 + 4, s.honest_row()));
    let mut s = base.clone();
    s.q[2][0] += 256;
    s.q[2][1] -= 1;
    cases.push(("q2_0 (the pair yR_31, q2_0)", h::Q2, s.honest_row()));

    for (name, col, row) in cases {
        assert!(row[col].canonical() >= 256, "{name}");
        let why = refusal(EcdasForm::Halves, &all, &row, None).expect(name);
        assert!(why.starts_with("interaction"), "{name}: {why}");
        let control = without(&all, are_bytes_on(&[col]));
        assert_eq!(control.len(), all.len() - 1, "{name}: one pair");
        assert_eq!(
            refusal(EcdasForm::Halves, &control, &row, None),
            None,
            "{name}: the control"
        );
    }
}

/// The limb form's table of real steps carries exactly the integer chains'
/// carries, and its padding rows' are zero.
#[test]
fn the_tables_carries_are_the_chains() {
    let steps = steps_for(&k_le(0xABCD));
    let ops: Vec<EcdasOperation> = steps
        .iter()
        .map(|s| EcdasOperation {
            timestamp: 9,
            step: s.clone(),
        })
        .collect();
    let rows = ops.len().next_power_of_two() * 2;
    for form in [TraceForm::Wide, TraceForm::Narrow, TraceForm::Narrow] {
        let trace = generate_ecdas_rows_in(&ops, rows, form, EcdasForm::Halves, |o| {
            std::borrow::Cow::Borrowed(o)
        });
        let columns: Vec<Vec<FE>> = match trace.narrow_main() {
            Some(n) => (0..n.cols())
                .map(|c| n.column(c).into_iter().map(FE::from).collect())
                .collect(),
            None => (0..trace.main_table.width)
                .map(|c| (0..rows).map(|r| *trace.main_table.get(r, c)).collect())
                .collect(),
        };
        let get = |r: usize, c: usize| columns[c][r];
        for r in 0..rows {
            let want = if r < steps.len() {
                halves_carries(&steps[r])
            } else {
                [[0; HALVES_CARRIES]; 3]
            };
            for (&(lo, hi, _), carries) in h::RELATIONS.iter().zip(want) {
                for (k, c) in carries.into_iter().enumerate() {
                    let got = get(r, lo + k) + FE::from(1u64 << 16) * get(r, hi + k)
                        - fe(HALVES_CARRY_OFFSET);
                    assert_eq!(got, fe(c), "{form:?} row {r}");
                }
            }
        }
    }
}

// =============================================================================
// Proved blocks
// =============================================================================

fn block_format(ecdas: EcdasForm) -> crate::block_whir::BlockFormat {
    crate::block_whir::BlockFormat {
        argue: multilinear::whir_chain::ArgueFormat::BATCHED,
        ecdas,
        ..crate::tests::block_whir_p1_tests::many_groups(stark::proof::options::BaseFormat::P1_WHIR)
    }
}

fn prove(elf: &[u8], format: &crate::block_whir::BlockFormat) -> crate::block_whir::BlockWhirProof {
    let mut o = crate::tests::block_whir_p1_tests::options(0);
    o.window_log2 = Some(3);
    crate::block_whir::prove_block_whir_with(
        elf,
        &[],
        &stark::proof::options::ProofOptions::default_test_options(),
        format,
        &o,
        &crate::block_whir::Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verify(
    proof: &crate::block_whir::BlockWhirProof,
    elf: &[u8],
    format: &crate::block_whir::BlockFormat,
) -> bool {
    crate::block_whir::verify_block_whir(
        proof,
        elf,
        &stark::proof::options::ProofOptions::default_test_options(),
        format,
    )
    .unwrap_or(false)
}

/// ★ T6 and the honest path on proved blocks: an ECSM block and an ECDSA
/// recovery block prove and verify under the limb form (their BITWISE
/// counts the limb form's lookups, or the bus would not balance); the byte
/// form's verifier refuses each, and the limb form's verifier refuses the
/// byte form's proof of the same program.
#[test]
fn a_limb_form_block_proves_and_each_form_refuses_the_other() {
    for guest in ["test_ecsm", "test_ecrec"] {
        let elf = crate::test_utils::asm_elf_bytes(guest);
        let halves = block_format(EcdasForm::Halves);
        let bytes = block_format(EcdasForm::Bytes);
        let proof = prove(&elf, &halves);
        assert!(proof.table_counts.ecdas > 0, "{guest}: ECDAS tables");
        assert!(verify(&proof, &elf, &halves), "{guest}: the limb form");
        assert!(
            !verify(&proof, &elf, &bytes),
            "{guest}: the byte form refuses it"
        );
        let todays = prove(&elf, &bytes);
        assert!(verify(&todays, &elf, &bytes), "{guest}: today's proof");
        assert!(
            !verify(&todays, &elf, &halves),
            "{guest}: the limb form refuses today's"
        );
        eprintln!(
            "ECDAS HALVES BLOCK {guest}: {} ECDAS tables; verified under halves, refused under \
             bytes and back",
            proof.table_counts.ecdas
        );
    }
}

/// The whole-run build lays ECDAS out in the byte form alone: it refuses the
/// limb form rather than prove a table its BITWISE does not count.
#[test]
fn the_whole_run_build_refuses_the_limb_form() {
    let elf = crate::test_utils::asm_elf_bytes("test_ecsm");
    let refused = crate::block_whir::prove_block_whir_with(
        &elf,
        &[],
        &stark::proof::options::ProofOptions::default_test_options(),
        &block_format(EcdasForm::Halves),
        &crate::tests::block_whir_p1_tests::options(0),
        &crate::block_whir::Deviations::default(),
    );
    let Err(err) = refused else {
        panic!("the whole-run build proved the limb form");
    };
    assert!(format!("{err:?}").contains("ECDAS"), "{err:?}");
}
