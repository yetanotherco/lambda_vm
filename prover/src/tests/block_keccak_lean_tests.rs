//! KECCAK_RND without the byte checks its D lookups imply
//! ([`KeccakRndForm::Lean`]): an opt-in
//! format value, [`KeccakRndForm::Full`] the production constant.
//!
//! Pinned here: the knob reads `full` or `lean` and nothing else; the lean
//! table is today's less exactly the 20 `Cxz_left` checks; every `Cxz_left`
//! byte still enters a `BYTE_ALU[XOR]` lookup, which is what makes the checks
//! redundant; the θ identity rules out the one extra value, by parity (and an
//! identity without its factor 2 would not); the verifier presets hold `Full`;
//! and a lean block proves and verifies while each form's verifier refuses
//! the other's proof.

use crate::block_whir::{
    BlockFormat, BlockWhirProof, Deviations, prove_block_whir_with, verify_block_whir,
};
use crate::lfm::whir_block_tree::keccak_rnd_setting;
use crate::tables::bitwise::BitwiseOperationType;
use crate::tables::keccak_rnd::{self, KeccakRndForm, cols};
use crate::tables::types::BusId;
use crate::test_utils::asm_elf_bytes;
use crate::tests::block_whir_p1_tests::{many_groups, options};
use math::field::element::FieldElement;
use multilinear::whir_chain::ArgueFormat;
use stark::lookup::{BusInteraction, BusValue, LinearTerm};
use stark::proof::options::{BaseFormat, ProofOptions};

type F = crate::tables::types::GoldilocksField;

/// `BLOCK_WHIR_KECCAK_RND`: unset is the pinned preset's form, `full` and
/// `lean` name a form, anything else is refused. The type's own default stays
/// `Full`, the form of every build outside the block format.
#[test]
fn the_keccak_rnd_knob_reads_full_or_lean() {
    assert_eq!(
        keccak_rnd_setting(None),
        Ok(BlockFormat::pinned().keccak_rnd)
    );
    assert_eq!(keccak_rnd_setting(Some("full")), Ok(KeccakRndForm::Full));
    assert_eq!(keccak_rnd_setting(Some(" lean ")), Ok(KeccakRndForm::Lean));
    for bad in ["", "1", "on", "Lean", "off"] {
        assert!(keccak_rnd_setting(Some(bad)).is_err(), "{bad:?}");
    }
    assert_eq!(KeccakRndForm::default(), KeccakRndForm::Full);
}

/// An interaction as text (its bus, side, multiplicity and values), for
/// comparing lists: `BusInteraction` has no `Debug` of its own.
fn key(i: &BusInteraction) -> String {
    format!(
        "{} {} {:?} {:?}",
        i.bus_id, i.is_sender, i.multiplicity, i.values
    )
}

/// The ARE_BYTES pair a sender checks, as its two columns, if it is one.
fn are_bytes_pair(i: &BusInteraction) -> Option<(usize, usize)> {
    if i.bus_id != u64::from(BusId::AreBytes) {
        return None;
    }
    match i.values.as_slice() {
        [
            BusValue::Packed {
                start_column: a, ..
            },
            BusValue::Packed {
                start_column: b, ..
            },
        ] => Some((*a, *b)),
        _ => None,
    }
}

/// The lean table is today's, in today's order, less exactly the 20 checks on
/// `Cxz_left`'s byte pairs: 1,031 interactions become 1,011, and a row sends
/// 200 byte checks instead of 220, padding rows included.
#[test]
fn the_lean_form_drops_exactly_the_cxz_left_byte_checks() {
    let full = keccak_rnd::bus_interactions_for(KeccakRndForm::Full);
    let lean = keccak_rnd::bus_interactions_for(KeccakRndForm::Lean);
    assert_eq!(full.len(), 1031);
    assert_eq!(lean.len(), 1011);
    assert_eq!(keccak_rnd::bus_interactions().len(), full.len());
    let cxz_left = |c: usize| (cols::CXZ_LEFT..cols::CXZ_LEFT + 40).contains(&c);
    let kept: Vec<String> = full
        .iter()
        .filter(|i| !are_bytes_pair(i).is_some_and(|(a, b)| cxz_left(a) && cxz_left(b)))
        .map(key)
        .collect();
    let lean_debug: Vec<String> = lean.iter().map(key).collect();
    assert_eq!(
        kept, lean_debug,
        "lean = full less the Cxz_left pairs, in order"
    );
    assert!(
        lean.iter()
            .filter_map(are_bytes_pair)
            .all(|(a, b)| !cxz_left(a) && !cxz_left(b)),
        "no Cxz_left byte check is left"
    );
    assert_eq!(
        KeccakRndForm::Full.are_bytes_per_row(),
        keccak_rnd::ARE_BYTES_PER_ROW
    );
    assert_eq!(KeccakRndForm::Lean.are_bytes_per_row(), 200);
    assert_eq!(
        lean.iter().filter(|i| are_bytes_pair(i).is_some()).count(),
        KeccakRndForm::Lean.are_bytes_per_row()
    );
    // 2 calls = 48 rows, padded to 64: 16 padding rows.
    assert_eq!(
        keccak_rnd::padding_are_bytes_for(2, KeccakRndForm::Full),
        16 * 220
    );
    assert_eq!(
        keccak_rnd::padding_are_bytes_for(2, KeccakRndForm::Lean),
        16 * 200
    );
    assert_eq!(keccak_rnd::padding_are_bytes(2), 16 * 220);
}

/// The premise that makes the dropped checks redundant: for every lane `x` and
/// byte `b`, a `BYTE_ALU[XOR]` lookup weighted by μ takes `Cxz_left[x][b]`
/// with coefficient 1 in its first-operand slot's neighbour — the second
/// operand — plus at most the one carry bit of the halfword below (even `b`)
/// and nothing else. Its operands are bytes, so `Cxz_left[x][b]` is a byte, or
/// `−1` beside a carry of 1 (even `b`).
#[test]
fn every_cxz_left_byte_enters_a_d_xor_lookup() {
    let lean = keccak_rnd::bus_interactions_for(KeccakRndForm::Lean);
    let xor = format!(
        "{:?}",
        BusValue::constant(crate::tables::types::alu_op::XOR as u64)
    );
    for x in 0..5 {
        for b in 0..8 {
            let left = cols::cxz_left(x, b);
            let carry = cols::cxz_right_bit_for_byte(b).map(|hw| cols::cxz_right_bit(x, hw));
            let found = lean.iter().any(|i| {
                if i.bus_id != u64::from(BusId::ByteAlu)
                    || !matches!(i.multiplicity, stark::lookup::Multiplicity::Column(c) if c == cols::MU)
                {
                    return false;
                }
                let [op, _, second, _] = i.values.as_slice() else {
                    return false;
                };
                if format!("{op:?}") != xor {
                    return false;
                }
                let mut want = vec![LinearTerm::Column {
                    coefficient: 1,
                    column: left,
                }];
                if let Some(c) = carry {
                    want.push(LinearTerm::Column {
                        coefficient: 1,
                        column: c,
                    });
                }
                format!("{second:?}") == format!("{:?}", BusValue::linear(want))
            });
            assert!(found, "Cxz_left[{x}][{b}] enters a μ-weighted D XOR lookup");
            assert_eq!(
                carry.is_some(),
                b % 2 == 0,
                "a carry beside the even bytes only"
            );
        }
    }
    // And the lookup's table holds bytes: BYTE_ALU is indexed by two bytes.
    assert_eq!(
        crate::tables::bitwise::NUM_ROWS,
        256 * 256 * 16,
        "(X: Byte, Y: Byte, Z: B4)"
    );
    let _ = BitwiseOperationType::ByteAluXor;
}

/// The θ identity's residue `2·in − 2¹⁶·r − (lo + 256·hi)` over the field.
fn theta_residue(input: u64, r: u64, hi: u64, lo: FieldElement<F>, two: u64) -> FieldElement<F> {
    FieldElement::<F>::from(two * input)
        - FieldElement::<F>::from(r << 16)
        - FieldElement::<F>::from(256 * hi)
        - lo
}

/// ★ The parity argument, exhaustively: with the low byte `−1` — the one value
/// the D lookup leaves an even `Cxz_left` byte besides `[0, 255]`, when the
/// carry beside it is 1 — no halfword `in`, carry `r` and byte `hi` satisfies
/// the θ identity. And every honest halfword's low byte is even, so the D
/// lookup's operand `lo + carry` stays a byte.
#[test]
fn a_low_byte_of_minus_one_breaks_the_theta_identity() {
    let minus_one = -FieldElement::<F>::one();
    let zero = FieldElement::<F>::zero();
    for input in 0..1u64 << 16 {
        let shifted = (input << 1) & 0xFFFF;
        let (lo, hi, r) = (shifted & 0xFF, shifted >> 8, input >> 15);
        assert_eq!(theta_residue(input, r, hi, FieldElement::from(lo), 2), zero);
        assert_eq!(lo % 2, 0, "in = {input}: the honest low byte is even");
        for r in 0..2u64 {
            for hi in 0..256u64 {
                assert_ne!(
                    theta_residue(input, r, hi, minus_one, 2),
                    zero,
                    "in = {input}, r = {r}, hi = {hi}: lo = −1 passes"
                );
            }
        }
    }
}

/// The argument's mutation: the same identity without its factor 2 admits a low
/// byte of `−1` (`in = 255`, `r = 0`, `hi = 1`), so the parity of `2·in` is
/// what the dropped checks' redundancy rests on.
#[test]
fn without_the_factor_two_a_low_byte_of_minus_one_would_pass() {
    let minus_one = -FieldElement::<F>::one();
    assert_eq!(
        theta_residue(255, 0, 1, minus_one, 1),
        FieldElement::<F>::zero()
    );
}

/// The verifier's presets hold `Lean`: a full proof verifies only under a
/// format that names `Full`.
#[test]
fn the_verifiers_keccak_rnd_form_is_a_named_constant() {
    assert_eq!(BlockFormat::pinned().keccak_rnd, KeccakRndForm::Lean);
    assert_eq!(BlockFormat::production().keccak_rnd, KeccakRndForm::Lean);
    assert_eq!(
        BlockFormat::pinned()
            .with_keccak_rnd(KeccakRndForm::Full)
            .keccak_rnd,
        KeccakRndForm::Full
    );
}

fn format(keccak_rnd: KeccakRndForm) -> BlockFormat {
    BlockFormat {
        argue: ArgueFormat::BATCHED,
        keccak_rnd,
        ..many_groups(BaseFormat::P1_WHIR)
    }
}

fn prove(elf: &[u8], format: &BlockFormat) -> BlockWhirProof {
    let mut o = options(0);
    o.window_log2 = Some(3);
    prove_block_whir_with(
        elf,
        &[],
        &ProofOptions::default_test_options(),
        format,
        &o,
        &Deviations::default(),
    )
    .expect("prove")
    .0
}

fn verify(proof: &BlockWhirProof, elf: &[u8], format: &BlockFormat) -> bool {
    verify_block_whir(proof, elf, &ProofOptions::default_test_options(), format).unwrap_or(false)
}

/// ★ A lean block proves and verifies (its BITWISE counts the lean table's
/// byte checks, or its bus would not balance); today's verifier refuses it and
/// the lean verifier refuses today's proof. A block with KECCAK_RND.
#[test]
fn a_lean_block_proves_and_verifies() {
    let elf = asm_elf_bytes("test_keccak");
    let lean = format(KeccakRndForm::Lean);
    let full = format(KeccakRndForm::Full);
    let proof = prove(&elf, &lean);
    assert!(verify(&proof, &elf, &lean), "the lean verifier");
    assert!(!verify(&proof, &elf, &full), "today's verifier refuses it");
    let todays = prove(&elf, &full);
    assert!(
        verify(&todays, &elf, &full),
        "today's verifier, today's proof"
    );
    assert!(
        !verify(&todays, &elf, &lean),
        "the lean verifier refuses today's"
    );
    eprintln!(
        "lean keccak block: {} tables, {} groups; verified under lean, refused under full and back",
        proof.table_num_vars.len(),
        proof.groups.len()
    );
}

/// Each form is its own AIR prototype: built in any order in one process, the
/// lean AIR has 1,011 interactions and the full one 1,031, both named
/// `KECCAK_RND`. (One shared prototype served the first form built to every
/// later caller: a lean proof then verified under today's format.)
#[test]
fn the_two_forms_never_share_an_air_prototype() {
    use stark::traits::AIR;
    let opts = ProofOptions::default_test_options();
    for form in [
        KeccakRndForm::Lean,
        KeccakRndForm::Full,
        KeccakRndForm::Lean,
        KeccakRndForm::Full,
    ] {
        let air = crate::test_utils::create_keccak_rnd_air_for(&opts, form);
        let want = match form {
            KeccakRndForm::Full => 1031,
            KeccakRndForm::Lean => 1011,
        };
        assert_eq!(air.bus_interactions().len(), want, "{form:?}");
        assert_eq!(air.name(), "KECCAK_RND", "{form:?}");
    }
}

/// The lean form's θ step checked on the real AIR: the real constraint set's
/// entries on real trace rows and the real interaction lists, not a model.
mod on_the_real_air {
    use crate::tables::keccak_rnd::{
        self, KeccakRndConstraints, KeccakRndForm, KeccakRoundOperation, cols,
        generate_keccak_rnd_trace,
    };
    use crate::tables::types::{BusId, FE, GoldilocksExtension, GoldilocksField, alu_op};
    use math::field::element::FieldElement;
    use stark::constraints::builder::{ConstraintSet, ProverEvalFolder};
    use stark::frame::Frame;
    use stark::lookup::{BusInteraction, Multiplicity};
    use stark::table::TableView;
    use stark::traits::TransitionEvaluationContext;

    /// θ identity `(x, hw)` sits at `20 + 4x + hw`; IS_BIT on `Cxz_right[x][hw]`
    /// at `4x + hw`.
    const THETA_BASE: usize = 20;

    fn ops() -> Vec<KeccakRoundOperation> {
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s
        };
        (0..2)
            .map(|t| KeccakRoundOperation {
                timestamp: 1000 + t,
                input: std::array::from_fn(|_| next()),
                output: [0; 25],
            })
            .collect()
    }

    fn row_of(
        trace: &stark::trace::TraceTable<GoldilocksField, GoldilocksExtension>,
        r: usize,
    ) -> Vec<FE> {
        (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(r, c))
            .collect()
    }

    /// Every one of the 141 constraints on one row (the prover folder).
    fn eval_row(main: &[FE]) -> Vec<FE> {
        let n = KeccakRndConstraints.meta().len();
        let frame = Frame::<GoldilocksField, GoldilocksExtension>::new(vec![TableView::new(
            vec![main.to_vec()],
            vec![vec![]],
        )]);
        let no_e: Vec<FieldElement<GoldilocksExtension>> = vec![];
        let offset_e = FieldElement::<GoldilocksExtension>::zero();
        let ctx =
            TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_e, &no_e, &offset_e);
        let mut base = vec![FE::zero(); n];
        let mut ext = vec![FieldElement::<GoldilocksExtension>::zero(); n];
        let mut folder = ProverEvalFolder::new(&ctx, &mut base, &mut ext);
        KeccakRndConstraints.eval(&mut folder);
        base
    }

    fn failing(main: &[FE]) -> Vec<usize> {
        eval_row(main)
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != FE::zero())
            .map(|(i, _)| i)
            .collect()
    }

    fn multiplicity(i: &BusInteraction, main: &[FE]) -> FE {
        match &i.multiplicity {
            Multiplicity::One => FE::one(),
            Multiplicity::Column(c) => main[*c],
            other => panic!("KECCAK_RND has no {other:?} multiplicity"),
        }
    }

    fn values(i: &BusInteraction, main: &[FE]) -> Vec<FE> {
        i.values
            .iter()
            .flat_map(|v| v.combine_from(|c| main[c]))
            .collect()
    }

    /// Whether a ByteAlu / AreBytes tuple is a row of BITWISE, and whether its
    /// operands alone are (an output can be recomputed, operands cannot).
    fn in_bitwise(bus: u64, v: &[FE]) -> (bool, bool) {
        let byte = |x: &FE| x.canonical() < 256;
        if bus == u64::from(BusId::AreBytes) {
            let ok = v.iter().all(byte);
            return (ok, ok);
        }
        assert_eq!(bus, u64::from(BusId::ByteAlu));
        let operands = byte(&v[1]) && byte(&v[2]);
        if !operands {
            return (false, false);
        }
        let (x, y) = (v[1].canonical(), v[2].canonical());
        let out = match v[0].canonical() as u8 {
            alu_op::AND => x & y,
            alu_op::OR => x | y,
            alu_op::XOR => x ^ y,
            _ => return (false, true),
        };
        (v[3].canonical() == out, true)
    }

    /// The byte-bus sends of a row whose tuple is not a BITWISE row, with their
    /// multiplicity: `(interaction index, μ, operands are bytes, values)`.
    fn bad_sends(
        interactions: &[BusInteraction],
        only: Option<&[usize]>,
        main: &[FE],
    ) -> Vec<(usize, FE, bool, Vec<u64>)> {
        let byte_bus = |b: u64| b == u64::from(BusId::ByteAlu) || b == u64::from(BusId::AreBytes);
        let all: Vec<usize> = (0..interactions.len()).collect();
        only.unwrap_or(&all)
            .iter()
            .filter_map(|&k| {
                let i = &interactions[k];
                if !byte_bus(i.bus_id) {
                    return None;
                }
                assert!(i.is_sender);
                let m = multiplicity(i, main);
                if m == FE::zero() {
                    return None;
                }
                let v = values(i, main);
                let (row, operands) = in_bitwise(i.bus_id, &v);
                (!row).then(|| (k, m, operands, v.iter().map(|x| x.canonical()).collect()))
            })
            .collect()
    }

    fn reads(i: &BusInteraction, cols_: &[usize]) -> bool {
        i.values
            .iter()
            .flat_map(|v| v.column_indices())
            .any(|c| cols_.contains(&c))
    }

    fn next(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// Every other `(lo, hi, r)` that keeps all 141 constraints on a real row
    /// sends a D lookup whose operand is not a byte under the lean form (and an
    /// `ARE_BYTES` pair off a byte under the full form): the θ decomposition is
    /// unique on the real AIR.
    #[test]
    fn every_other_decomposition_breaks_a_d_operand() {
        let trace = generate_keccak_rnd_trace(&ops());
        let lean = keccak_rnd::bus_interactions_for(KeccakRndForm::Lean);
        let full = keccak_rnd::bus_interactions_for(KeccakRndForm::Full);
        let are_bytes = u64::from(BusId::AreBytes);
        let (mut tried, mut kept_constraints) = (0u64, 0u64);
        for r in 0..48 {
            let honest = row_of(&trace, r);
            assert_eq!(honest[cols::MU], FE::one());
            for x in 0..5 {
                for hw in 0..4 {
                    let (c_lo, c_hi) = (cols::cxz_left(x, 2 * hw), cols::cxz_left(x, 2 * hw + 1));
                    let c_r = cols::cxz_right_bit(x, hw);
                    let touched = [c_lo, c_hi, c_r];
                    let lean_idx: Vec<usize> = (0..lean.len())
                        .filter(|&k| reads(&lean[k], &touched))
                        .collect();
                    let full_idx: Vec<usize> = (0..full.len())
                        .filter(|&k| reads(&full[k], &touched))
                        .collect();
                    let input = honest[cols::cxz(x, 3, 2 * hw)].canonical()
                        + 256 * honest[cols::cxz(x, 3, 2 * hw + 1)].canonical();
                    for rr in 0..2u64 {
                        for hi in 0..256u64 {
                            let lo = FE::from(2 * input) - FE::from(rr << 16) - FE::from(256 * hi);
                            if (lo, FE::from(hi), FE::from(rr))
                                == (honest[c_lo], honest[c_hi], honest[c_r])
                            {
                                continue;
                            }
                            tried += 1;
                            let mut main = honest.clone();
                            main[c_lo] = lo;
                            main[c_hi] = FE::from(hi);
                            main[c_r] = FE::from(rr);
                            if !failing(&main).is_empty() {
                                continue;
                            }
                            kept_constraints += 1;
                            let bad_lean = bad_sends(&lean, Some(&lean_idx), &main);
                            assert!(
                                bad_lean.iter().any(|(k, _, operands, _)| !operands
                                    && lean[*k].bus_id == u64::from(BusId::ByteAlu)),
                                "row {r} x {x} hw {hw} lo {} hi {hi} r {rr}: no D operand off a byte",
                                lo.canonical()
                            );
                            let bad_full = bad_sends(&full, Some(&full_idx), &main);
                            assert!(
                                bad_full.iter().any(|(k, ..)| full[*k].bus_id == are_bytes),
                                "the full form's byte check also refuses it"
                            );
                        }
                    }
                }
            }
        }
        eprintln!(
            "θ decomposition search: {tried} mutations, {kept_constraints} keep all 141 constraints, \
             every one sends a D lookup with a non-byte operand under Lean"
        );
        assert_eq!(tried, 48 * 20 * 511);
        assert_eq!(
            kept_constraints, tried,
            "θ identity and IS_BIT hold by construction"
        );
    }

    /// The mutation: with the D lookups that read `Cxz_left` gone, a forged
    /// decomposition (`lo − 256`, `hi + 1`) keeps every constraint and sends
    /// nothing off BITWISE. So the D lookups are what refuse it under the lean form.
    #[test]
    fn the_harness_sees_a_missing_d_lookup() {
        let trace = generate_keccak_rnd_trace(&ops());
        let left: Vec<usize> = (cols::CXZ_LEFT..cols::CXZ_LEFT + 40).collect();
        let without_d: Vec<BusInteraction> = keccak_rnd::bus_interactions_for(KeccakRndForm::Lean)
            .into_iter()
            .filter(|i| !(i.bus_id == u64::from(BusId::ByteAlu) && reads(i, &left)))
            .collect();
        assert_eq!(without_d.len(), 1011 - 40);
        let (c_lo, c_hi) = (cols::cxz_left(0, 0), cols::cxz_left(0, 1));
        let r = (0..48)
            .find(|&r| trace.main_table.get(r, c_hi).canonical() < 255)
            .expect("a high byte below 255");
        let mut main = row_of(&trace, r);
        main[c_lo] = main[c_lo] - FE::from(256u64);
        main[c_hi] += FE::one();
        assert!(failing(&main).is_empty());
        assert!(bad_sends(&without_d, None, &main).is_empty());
        let lean = keccak_rnd::bus_interactions_for(KeccakRndForm::Lean);
        assert!(!bad_sends(&lean, None, &main).is_empty());
    }

    /// The AIR's θ entry is exactly `2·in − 2¹⁶·r − (lo + 256·hi)`, weighted
    /// by μ, for any field values: the identity [`super::theta_residue`] models
    /// and the exhaustive parity check runs over.
    #[test]
    fn the_theta_entry_is_the_identity_the_proof_uses() {
        let t = generate_keccak_rnd_trace(&ops());
        let mut seed = 0x1234_5678_9abc_def1u64;
        for x in 0..5 {
            for hw in 0..4 {
                for _ in 0..64 {
                    let mut main = row_of(&t, 3);
                    let [a, b, lo, hi, r] = [(); 5].map(|_| FE::from(next(&mut seed)));
                    main[cols::cxz(x, 3, 2 * hw)] = a;
                    main[cols::cxz(x, 3, 2 * hw + 1)] = b;
                    main[cols::cxz_left(x, 2 * hw)] = lo;
                    main[cols::cxz_left(x, 2 * hw + 1)] = hi;
                    main[cols::cxz_right_bit(x, hw)] = r;
                    let want = FE::from(2u64) * (a + FE::from(256u64) * b)
                        - FE::from(1u64 << 16) * r
                        - lo
                        - FE::from(256u64) * hi;
                    let got = eval_row(&main);
                    assert_eq!(got[THETA_BASE + 4 * x + hw], want, "x {x} hw {hw}");
                    assert_eq!(got[4 * x + hw], r * (FE::one() - r), "IS_BIT x {x} hw {hw}");
                    main[cols::MU] = FE::zero();
                    assert_eq!(
                        eval_row(&main)[THETA_BASE + 4 * x + hw],
                        FE::zero(),
                        "μ = 0"
                    );
                }
            }
        }
    }
}
