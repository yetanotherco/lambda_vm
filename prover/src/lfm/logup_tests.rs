//! The LogUp closure, and the join it inherits.
//!
//! ## The oracles
//!
//! Two, both production's own. `compute_commit_bus_offset` (`lib.rs`) for the
//! COMMIT-bus target, and `Verifier::multi_verify` for the balance itself — the
//! fixture is a real sender/receiver pair whose bus genuinely closes, and
//! production accepting it at target zero is what says so. Nothing here asserts
//! a balance this file computed.
//!
//! ## What this suite cannot see
//!
//! Whether `L` is bound to a table's aux TRACE. That binding is the circular
//! accumulator constraint plus the `acc[0] = 0` boundary, and it belongs to the
//! constraint leg; this suite checks only that the closure consumes the same
//! `L` that leg divides by `N`. It also cannot see a full epoch's table set —
//! the fixture is two tables, not the twenty-odd a continuation epoch carries,
//! so the SUM is exercised but its length is not.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark::proof::stark::MultiProof;
use stark::proof::view::StarkProofView;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

use super::builder::LfmBuilder;
use super::compiler::compile;
use super::executor::execute;
use super::hash::TestPermutation;
use super::logup::{COMMIT_BUS_ID, LogUpShape, emit_bus_closure, emit_commit_bus_target};
use super::validator::validate;
use super::word::{LfmWord, base_word, ext_word, word_as_ext};

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

fn options() -> stark::proof::options::ProofOptions {
    stark::proof::options::GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// The mirrored bus discriminant really is the VM's.
///
/// A copied protocol constant is the one thing a differential cannot catch,
/// because both sides of it move together — the same reason the join suite pins
/// `ROWS_PER_LEAF` against `crypto/stark`'s.
#[test]
fn bus_id_matches_production() {
    assert_eq!(
        COMMIT_BUS_ID,
        crate::tables::types::BusId::Commit as u64,
        "the COMMIT fingerprint's constant term is the VM's own bus id"
    );
}

// =============================================================================
// The COMMIT-bus target
// =============================================================================

/// Emit the target gadget alone and run it.
fn run_target(bytes: &[u8], start: u64, z: FEE, alpha: FEE) -> Option<FEE> {
    let shape = LogUpShape {
        num_contributing_tables: 0,
        num_output_bytes: bytes.len(),
    };
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena((3 + bytes.len()) as u32);
    let z_cell = b.hint_word(arena, 0).as_ext();
    let alpha_cell = b.hint_word(arena, 1).as_ext();
    let start_cell = b.hint_felt(arena, 2);
    let byte_cells: Vec<_> = (0..bytes.len() as u32)
        .map(|i| b.hint_felt(arena, 3 + i))
        .collect();
    let target =
        emit_commit_bus_target(&mut b, &shape, z_cell, alpha_cell, start_cell, &byte_cells);
    b.public(target.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the target program is admissible");

    let words: Vec<LfmWord> = std::iter::once(ext_word(&z))
        .chain(std::iter::once(ext_word(&alpha)))
        .chain(std::iter::once(base_word(FE::from(start))))
        .chain(bytes.iter().map(|v| base_word(FE::from(*v as u64))))
        .collect();
    execute(&program, &[words], &TestPermutation)
        .ok()
        .map(|e| word_as_ext(&e.public_words[0].1).expect("ext"))
}

/// ★ The machine's COMMIT-bus target equals production's, over a sweep of
/// lengths and carried start indices.
///
/// The lengths are not decorative. `start` advances by one per byte inside the
/// gadget, so a formula that reset it, or that folded the bytes in reverse,
/// agrees with production only at length one — and the empty case is a separate
/// short-circuit in production that a nonempty-only test would never reach.
#[test]
fn the_commit_bus_target_matches_production() {
    let z = FEE::new([FE::from(7u64), FE::from(11u64), FE::from(13u64)]);
    let alpha = FEE::new([FE::from(5u64), FE::from(3u64), FE::from(2u64)]);

    let mut checked = 0usize;
    for len in [0usize, 1, 2, 3, 7, 8, 33] {
        // Distinct byte values, so a gadget that mixed up index and value would
        // not accidentally agree.
        let bytes: Vec<u8> = (0..len).map(|i| (17 * i + 3) as u8).collect();
        for start in [0u64, 1, 254, 1_000_000] {
            let want = crate::compute_commit_bus_offset(&bytes, start, &z, &alpha)
                .expect("no collision on this fixture");
            let got = run_target(&bytes, start, z, alpha)
                .unwrap_or_else(|| panic!("len {len} start {start}: the target must execute"));
            assert_eq!(got, want, "len {len}, start {start}");
            if len > 0 {
                assert_ne!(
                    got,
                    FEE::zero(),
                    "len {len} start {start}: a zero target would make the \
                     comparison vacuous"
                );
            }
            checked += 1;
        }
    }
    println!("commit-bus target: {checked} (length, start) combinations vs production");
}

/// ★ A fingerprint COLLISION is rejected, not silently folded.
///
/// Production batch-inverts and returns `None` on a zero divisor. The machine
/// divides the interned ONE by the fingerprint, so a collision is `1/0` — an
/// error, hence unprovable. Had the term been written as a direct division with
/// a vanishing numerator instead, the `0/0 = 1` convention would have accepted
/// exactly the proof production rejects, which is the mistake the DEEP leg
/// documents and this test exists to keep from recurring.
#[test]
fn a_fingerprint_collision_is_unprovable() {
    // fingerprint_0 = z − (busId + start·α + byte·α²). With α = 1, start = 0
    // and byte = 0 that is z − busId, so z = busId collides exactly.
    let alpha = FEE::one();
    let z = FEE::from(COMMIT_BUS_ID);
    let bytes = [0u8];

    assert!(
        crate::compute_commit_bus_offset(&bytes, 0, &z, &alpha).is_none(),
        "the fixture must be a genuine collision for production too, or this \
         test is checking the machine against nothing"
    );
    assert!(
        run_target(&bytes, 0, z, alpha).is_none(),
        "a colliding fingerprint must make the run unexecutable"
    );

    // And the same shape one step away from the collision still works, so the
    // rejection is the collision and not the shape.
    let z_ok = z + FEE::one();
    let want = crate::compute_commit_bus_offset(&bytes, 0, &z_ok, &alpha).expect("no collision");
    assert_eq!(run_target(&bytes, 0, z_ok, alpha).expect("executes"), want);
    println!("collision rejected; the neighbouring non-colliding z still folds");
}

// =============================================================================
// The closure, over a bus that really balances
// =============================================================================

/// A sender/receiver pair over one bus, proved together.
///
/// Modelled on `tests::bitwise_tests`' pair: the sender emits one AND lookup,
/// the receiver answers it. Their contributions are equal and opposite, so the
/// bus closes at zero — and `multi_verify` accepting at target zero is what
/// establishes that, rather than any arithmetic here.
fn balanced_pair() -> (Vec<BoxedAir>, MultiProof<Gl, Ext3, ()>) {
    use crate::tables::types::{BusId, alu_op};
    use crate::test_utils::multi_prove_ram;
    use stark::constraints::builder::EmptyConstraints;
    use stark::lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    };
    use stark::trace::TraceTable;

    const X: u64 = 5;
    const Y: u64 = 3;
    const NUM_ROWS: usize = 4;

    type Air = AirWithBuses<Gl, Ext3, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
    let opts = options();

    // Columns: 0 = x, 1 = y, 2 = and, 3 = multiplicity/flag. Same layout both
    // sides, so one trace builder serves.
    // Both sides fingerprint the SAME tuple; only the sender/receiver sign
    // differs, which is what makes the two contributions cancel.
    let values = || {
        vec![
            BusValue::constant(alu_op::AND as u64),
            BusValue::Packed {
                start_column: 0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: 1,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: 2,
                packing: Packing::Direct,
            },
        ]
    };

    let sender = Air::new(
        4,
        AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::sender(
                BusId::ByteAlu,
                Multiplicity::Column(3),
                values(),
            )],
        },
        &opts,
        1,
        EmptyConstraints,
    )
    .with_name("SENDER");
    let receiver = Air::new(
        4,
        AuxiliaryTraceBuildData {
            interactions: vec![BusInteraction::receiver(
                BusId::ByteAlu,
                Multiplicity::Column(3),
                values(),
            )],
        },
        &opts,
        1,
        EmptyConstraints,
    )
    .with_name("RECEIVER");

    let make_trace = || {
        let mut data = vec![FE::zero(); NUM_ROWS * 4];
        data[0] = FE::from(X);
        data[1] = FE::from(Y);
        data[2] = FE::from(X & Y);
        data[3] = FE::one();
        TraceTable::<Gl, Ext3>::new_main(data, 4, 1)
    };
    let mut sender_trace = make_trace();
    let mut receiver_trace = make_trace();

    let pairs: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&receiver, &mut receiver_trace, &()),
    ];
    let proof = multi_prove_ram(pairs, &mut DefaultTranscript::<Ext3>::new(&[]))
        .expect("the balanced pair must prove");
    (vec![Box::new(sender), Box::new(receiver)], proof)
}

type BoxedAir = Box<dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>>;

/// ★ The machine's closure accepts a bus production says balances, and rejects
/// every single-word move away from it.
///
/// The oracle is `multi_verify` at target zero. It is checked FIRST: if the
/// fixture's bus did not actually close, the machine agreeing with it would say
/// nothing.
#[test]
fn the_closure_matches_a_bus_that_really_balances() {
    let (airs, proof) = balanced_pair();
    let air_refs: Vec<&dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>> =
        airs.iter().map(|a| &**a).collect();

    assert!(
        Verifier::multi_verify(
            &air_refs,
            &proof,
            &mut DefaultTranscript::<Ext3>::new(&[]),
            &FEE::zero(),
        ),
        "production must accept this pair at target zero, or the fixture is not \
         a balanced bus and nothing below means anything"
    );

    let contributions: Vec<FEE> = (0..proof.proofs.len())
        .map(|i| {
            StarkProofView::Owned(&proof.proofs[i])
                .bus_table_contribution()
                .expect("both tables carry a contribution")
        })
        .collect();
    assert_eq!(contributions.len(), 2);
    assert!(
        contributions.iter().all(|c| *c != FEE::zero()),
        "both contributions must be nonzero, else the sum is vacuously zero: {contributions:?}"
    );
    assert_eq!(
        contributions[0] + contributions[1],
        FEE::zero(),
        "the pair is equal and opposite"
    );

    let shape = LogUpShape {
        num_contributing_tables: 2,
        num_output_bytes: 0,
    };
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(2);
    let cells: Vec<_> = (0..2u32).map(|i| b.hint_word(arena, i).as_ext()).collect();
    let zero = b.ext_const(&FEE::zero());
    let total = emit_bus_closure(&mut b, &shape, &cells, zero);
    b.public(total.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the closure program is admissible");

    let honest: Vec<LfmWord> = contributions.iter().map(ext_word).collect();
    let exec = execute(&program, std::slice::from_ref(&honest), &TestPermutation)
        .expect("a balanced bus must close in the machine too");
    assert_eq!(
        word_as_ext(&exec.public_words[0].1).expect("ext"),
        FEE::zero()
    );

    // Falsification: move either contribution, in any lane.
    let mut vectors = 0usize;
    for table in 0..2usize {
        for lane in 0..3usize {
            let mut arenas = vec![honest.clone()];
            arenas[0][table][lane] += FE::one();
            assert!(
                execute(&program, &arenas, &TestPermutation).is_err(),
                "table {table} lane {lane}: an unbalanced bus must not close"
            );
            vectors += 1;
        }
    }
    println!("closure: balanced pair accepted, {vectors} single-lane moves rejected");
}

// =============================================================================
// The join: one `L`, two consumers
// =============================================================================

use super::constraint_tests::{RealSubProof, real_sub_proof};
use super::constraints::{
    OodOperands, emit_alpha_powers, emit_constraint_evals, emit_quotient, emit_table_offset,
    hint_ood_frame, ood_frame_words,
};

/// Where the per-row offset comes from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Offset {
    /// `L/N` derived in-machine from the closure's own `L`. What production
    /// does, and what [`emit_table_offset`] exists to enforce.
    Derived,
    /// `L/N` hinted as its own arena word, independent of the `L` the closure
    /// sums. The shape this leg exists to forbid — a test artifact, built to be
    /// attacked.
    HintedSeparately,
}

/// One sub-proof's composition check AND the LogUp closure, over the same
/// proof, with the offset wired either way.
///
/// Arenas 0-2 are the constraint leg's (frame, uniforms, parts); arena 3 is the
/// closure's expected balance, plus — in the split shape only — the separately
/// hinted offset.
fn composition_and_closure_source(
    sp: &RealSubProof,
    offset: Offset,
) -> super::builder::LfmProgramSource {
    let mut b = LfmBuilder::new();

    let frame_arena = b.declare_arena(ood_frame_words(&sp.artifact));
    let (steps, _) = hint_ood_frame(&mut b, &sp.artifact, frame_arena, 0);

    let num_uniforms = (sp.rap_challenges.len() + 3) as u32;
    let uniform_arena = b.declare_arena(num_uniforms);
    let mut next = 0u32;
    let mut take = |b: &mut LfmBuilder| {
        let c = b.hint_word(uniform_arena, next).as_ext();
        next += 1;
        c
    };
    let rap_challenges: Vec<_> = (0..sp.rap_challenges.len()).map(|_| take(&mut b)).collect();
    let alpha_powers = emit_alpha_powers(
        &mut b,
        rap_challenges[stark::lookup::LOGUP_CHALLENGE_ALPHA],
        sp.alpha_powers.len(),
    );
    let contribution = take(&mut b);
    let zeta = take(&mut b);
    let beta = take(&mut b);

    let parts_arena = b.declare_arena(sp.claimed_parts.len() as u32);
    let claimed_parts: Vec<_> = (0..sp.claimed_parts.len() as u32)
        .map(|i| b.hint_word(parts_arena, i).as_ext())
        .collect();

    let closure_arena = b.declare_arena(match offset {
        Offset::Derived => 1,
        Offset::HintedSeparately => 2,
    });
    let target = b.hint_word(closure_arena, 0).as_ext();

    let table_offset = match offset {
        Offset::Derived => emit_table_offset(&mut b, contribution, sp.quotient.log2_trace_length),
        Offset::HintedSeparately => b.hint_word(closure_arena, 1).as_ext(),
    };

    let ood = OodOperands {
        steps,
        main_width: sp.main_width,
        rap_challenges,
        alpha_powers,
        table_offset,
    };
    let (evals, _) = emit_constraint_evals(&mut b, &sp.artifact, &ood);
    let q = emit_quotient(
        &mut b,
        &sp.quotient,
        &ood,
        zeta,
        beta,
        &evals,
        &claimed_parts,
    );
    b.assert_eq_ext(q.claimed, q.composition);

    let shape = LogUpShape {
        num_contributing_tables: 1,
        num_output_bytes: 0,
    };
    let total = emit_bus_closure(&mut b, &shape, &[contribution], target);
    b.public(total.as_cell());
    b.finish()
}

/// Arenas for the program above. `delta` moves the `L` the CLOSURE sums (and
/// the target with it, so the closure itself still balances); the offset stays
/// truthful, which is what a forger would want.
fn join_arenas(sp: &RealSubProof, offset: Offset, delta: FEE) -> Vec<Vec<LfmWord>> {
    let mut arenas = sp.arenas();
    let forged = sp.contribution + delta;
    // Slot of `contribution` inside the uniform arena.
    let slot = sp.rap_challenges.len();
    arenas[1][slot] = ext_word(&forged);
    let mut closure = vec![ext_word(&forged)];
    if offset == Offset::HintedSeparately {
        closure.push(ext_word(&sp.table_offset));
    }
    arenas.push(closure);
    arenas
}

/// ★ The join, stated as the property it exists for: a prover cannot feed the
/// bus a contribution the constraint leg did not accept.
///
/// The honest run passes both halves. Moving `L` — while keeping the closure
/// self-consistent by moving its target too, which is exactly what a forger
/// would do — must break the CONSTRAINT half, because the offset is derived
/// from the very cell that moved.
///
/// The control is the same program with the offset hinted separately. It
/// accepts the forgery: the accumulator sees a truthful `L/N` and wraps, the
/// closure sees a fabricated `L` and balances, and the bus statement is about a
/// number attached to no trace. That is what the derivation denies, and it is
/// run here rather than argued.
#[test]
fn the_closure_cannot_sum_a_contribution_the_constraints_rejected() {
    let sp = real_sub_proof();
    assert_ne!(
        sp.contribution,
        FEE::zero(),
        "the fixture's table must carry a real bus contribution, or moving it \
         is not a tamper"
    );

    let joined = compile(composition_and_closure_source(&sp, Offset::Derived));
    validate(&joined).expect("the joined program is admissible");
    let split = compile(composition_and_closure_source(
        &sp,
        Offset::HintedSeparately,
    ));
    validate(&split).expect("the control is admissible");

    // Honest, both shapes.
    for (label, program, offset) in [
        ("joined", &joined, Offset::Derived),
        ("control", &split, Offset::HintedSeparately),
    ] {
        let exec = execute(
            program,
            &join_arenas(&sp, offset, FEE::zero()),
            &TestPermutation,
        )
        .unwrap_or_else(|e| panic!("{label}: the honest run must execute: {e:?}"));
        assert_eq!(
            word_as_ext(&exec.public_words[0].1).expect("ext"),
            sp.contribution,
            "{label}: the published total is the contribution that was checked"
        );
    }

    // Forge, several deltas and several lanes, so the vector class is not one
    // value in one coordinate.
    let deltas = [
        FEE::one(),
        FEE::new([FE::zero(), FE::one(), FE::zero()]),
        FEE::new([FE::zero(), FE::zero(), FE::from(7u64)]),
        FEE::new([FE::from(3u64), FE::from(5u64), FE::from(9u64)]),
    ];
    for delta in deltas {
        assert!(
            execute(
                &joined,
                &join_arenas(&sp, Offset::Derived, delta),
                &TestPermutation
            )
            .is_err(),
            "joined: a forged contribution must break the constraint half \
             (delta {delta:?})"
        );
        let forged = execute(
            &split,
            &join_arenas(&sp, Offset::HintedSeparately, delta),
            &TestPermutation,
        )
        .unwrap_or_else(|e| panic!("control: the split shape is what PERMITS this forgery: {e:?}"));
        assert_eq!(
            word_as_ext(&forged.public_words[0].1).expect("ext"),
            sp.contribution + delta,
            "control: the forgery publishes its own fabricated contribution"
        );
    }
    println!(
        "join: {} forged contributions rejected by the derivation, all accepted \
         by the split control",
        deltas.len()
    );
}

/// ★ The two-consumer rule, as an ABSOLUTE property of the emitted program.
///
/// Method rule 7: a relative test dies the moment its two sides unify, so this
/// asserts nothing about variants and everything about the program itself —
/// which cells are arena hints and which are computed. `L/N` and every alpha
/// power must be COMPUTED, because a hinted one is a claim about `L` or `α`
/// that no other constraint checks; `L` itself and the raw challenges must be
/// hints, or the test would pass vacuously against an emitter that simply
/// dropped them.
///
/// This survives any refactor of how the offset is produced. It only fails if
/// something starts reading these values out of an arena again, which is
/// exactly the regression it exists to catch.
#[test]
fn the_derived_uniforms_are_not_arena_words() {
    use super::instr::Instr;

    let sp = real_sub_proof();
    assert!(
        !sp.alpha_powers.is_empty(),
        "the fixture must exercise LogUp alpha powers"
    );

    let mut b = LfmBuilder::new();
    let uniform_arena = b.declare_arena((sp.rap_challenges.len() + 1) as u32);
    let rap: Vec<_> = (0..sp.rap_challenges.len() as u32)
        .map(|i| b.hint_word(uniform_arena, i).as_ext())
        .collect();
    let contribution = b
        .hint_word(uniform_arena, sp.rap_challenges.len() as u32)
        .as_ext();
    let alpha_powers = emit_alpha_powers(
        &mut b,
        rap[stark::lookup::LOGUP_CHALLENGE_ALPHA],
        sp.alpha_powers.len(),
    );
    let table_offset = emit_table_offset(&mut b, contribution, sp.quotient.log2_trace_length);
    let source = b.finish();

    let hinted: std::collections::HashSet<u64> = source
        .instrs
        .iter()
        .filter_map(|i| match i {
            Instr::Hint { out, .. } => Some(out.0),
            _ => None,
        })
        .collect();

    // Positive control: the things that SHOULD be arena words are.
    assert!(
        hinted.contains(&contribution.addr().0),
        "L itself is proof data and must be hinted, or this test is checking \
         an emitter that reads nothing"
    );
    for (i, c) in rap.iter().enumerate() {
        assert!(
            hinted.contains(&c.addr().0),
            "rap challenge {i} is hinted in this isolated slice"
        );
    }

    // The property: derived values are not arena words.
    assert!(
        !hinted.contains(&table_offset.addr().0),
        "L/N must be COMPUTED from L; a hinted offset lets a prover satisfy \
         every accumulator while the closure sums a different L"
    );
    for (i, p) in alpha_powers.iter().enumerate() {
        // alpha^1 IS the alpha cell, which is legitimately hinted here; every
        // other power must be computed (alpha^0 is an interned constant).
        if i == stark::lookup::LOGUP_CHALLENGE_ALPHA {
            continue;
        }
        assert!(
            !hinted.contains(&p.addr().0),
            "alpha power {i} must be COMPUTED from alpha; a hinted power is a \
             claim about alpha that nothing checks, and the LogUp fingerprints \
             are built out of exactly these"
        );
    }
    println!(
        "absolute check: L/N and {} alpha powers are computed, not hinted",
        alpha_powers.len()
    );
}
