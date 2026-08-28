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
//! `L` that leg divides by `N`.
//!
//! Most fixtures here are two or three tables rather than the twenty-odd a
//! continuation epoch carries, so they exercise the SUM but not its length. One
//! is not: [`a_zero_row_fixed_table_carries_some_zero_not_none`] proves and
//! verifies a real epoch and runs the closure over all twenty-four of its
//! contributions.
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
    let proof = multi_prove_ram(pairs, &mut crate::hash_pin::block_transcript(&[]))
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
            &mut crate::hash_pin::block_transcript(&[]),
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

// =============================================================================
// Degenerate parameter: per-CHUNK vs per-FAMILY accumulation
// =============================================================================

/// One sender and TWO receiver chunks of the same family, proved together.
///
/// A continuation epoch splits each table family into chunks, and `VmAirs::new`
/// builds one AIR per chunk (`lib.rs`: `(0..table_counts.cpu).map(|i| … CPU[i])`),
/// so a family of `k` chunks is `k` entries in the AIR vector and `k`
/// sub-proofs. The closure iterates those entries, so it accumulates PER CHUNK.
///
/// Every fixture up to here had one sub-proof per family, which makes
/// per-chunk and per-family the same sum — the degenerate case. Two chunks of
/// one family is the smallest shape that tells them apart: the sender's two
/// lookups are answered one per chunk, so dropping either chunk leaves a
/// nonzero remainder.
fn chunked_family() -> (Vec<BoxedAir>, MultiProof<Gl, Ext3, ()>) {
    use crate::tables::types::{BusId, alu_op};
    use crate::test_utils::multi_prove_ram;
    use stark::constraints::builder::EmptyConstraints;
    use stark::lookup::{
        AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
        NullBoundaryConstraintBuilder, Packing,
    };
    use stark::trace::TraceTable;

    /// The two lookups, answered by one receiver chunk each.
    const LOOKUPS: [(u64, u64); 2] = [(5, 3), (9, 6)];
    const NUM_ROWS: usize = 4;

    type Air = AirWithBuses<Gl, Ext3, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
    let opts = options();

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
    let build = |sender: bool, name: &str| {
        let interaction = if sender {
            BusInteraction::sender(BusId::ByteAlu, Multiplicity::Column(3), values())
        } else {
            BusInteraction::receiver(BusId::ByteAlu, Multiplicity::Column(3), values())
        };
        Air::new(
            4,
            AuxiliaryTraceBuildData {
                interactions: vec![interaction],
            },
            &opts,
            1,
            EmptyConstraints,
        )
        .with_name(name)
    };

    // The two receiver chunks are the SAME construction — one family, two
    // instances, exactly as `CPU[0]` and `CPU[1]` are.
    let sender = build(true, "SENDER");
    let recv0 = build(false, "RECEIVER[0]");
    let recv1 = build(false, "RECEIVER[1]");

    let trace_for = |rows: &[(u64, u64)]| {
        let mut data = vec![FE::zero(); NUM_ROWS * 4];
        for (r, (x, y)) in rows.iter().enumerate() {
            data[r * 4] = FE::from(*x);
            data[r * 4 + 1] = FE::from(*y);
            data[r * 4 + 2] = FE::from(x & y);
            data[r * 4 + 3] = FE::one();
        }
        TraceTable::<Gl, Ext3>::new_main(data, 4, 1)
    };
    let mut sender_trace = trace_for(&LOOKUPS);
    let mut recv0_trace = trace_for(&LOOKUPS[..1]);
    let mut recv1_trace = trace_for(&LOOKUPS[1..]);

    let pairs: Vec<(
        &dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>,
        _,
        _,
    )> = vec![
        (&sender, &mut sender_trace, &()),
        (&recv0, &mut recv0_trace, &()),
        (&recv1, &mut recv1_trace, &()),
    ];
    let proof = multi_prove_ram(pairs, &mut crate::hash_pin::block_transcript(&[]))
        .expect("the chunked family must prove");
    (
        vec![Box::new(sender), Box::new(recv0), Box::new(recv1)],
        proof,
    )
}

/// ★ The closure accumulates per CHUNK, and that is observable.
///
/// Both halves of the degenerate-parameter rule. The machine's three-term sum
/// closes a bus production accepts at target zero; and every two-term sum — a
/// per-family reading, which would collapse the two receiver chunks into one
/// contribution — is NONZERO, so the distinction is load-bearing on this
/// fixture rather than merely stated.
///
/// Without the second half this test would pass against an emitter that folded
/// a family's chunks into a single term, because on every earlier fixture, and
/// on any workload whose families happen to be one chunk each, the two readings
/// agree.
#[test]
fn the_closure_accumulates_per_chunk_not_per_family() {
    let (airs, proof) = chunked_family();
    let air_refs: Vec<&dyn AIR<Field = Gl, FieldExtension = Ext3, PublicInputs = ()>> =
        airs.iter().map(|a| &**a).collect();
    assert_eq!(air_refs.len(), 3, "one sender and two chunks of one family");

    assert!(
        Verifier::multi_verify(
            &air_refs,
            &proof,
            &mut crate::hash_pin::block_transcript(&[]),
            &FEE::zero(),
        ),
        "production must accept the chunked family at target zero, or the \
         fixture is not a balanced bus"
    );

    let contributions: Vec<FEE> = (0..proof.proofs.len())
        .map(|i| {
            StarkProofView::Owned(&proof.proofs[i])
                .bus_table_contribution()
                .expect("every table here has interactions")
        })
        .collect();
    assert!(
        contributions.iter().all(|c| *c != FEE::zero()),
        "each chunk must carry its OWN nonzero contribution — a zero one would \
         make dropping it invisible: {contributions:?}"
    );

    // Half one: the per-chunk sum closes, in the machine.
    let shape = LogUpShape {
        num_contributing_tables: 3,
        num_output_bytes: 0,
    };
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(3);
    let cells: Vec<_> = (0..3u32).map(|i| b.hint_word(arena, i).as_ext()).collect();
    let zero = b.ext_const(&FEE::zero());
    let total = emit_bus_closure(&mut b, &shape, &cells, zero);
    b.public(total.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("admissible");

    let words: Vec<LfmWord> = contributions.iter().map(ext_word).collect();
    let exec = execute(&program, std::slice::from_ref(&words), &TestPermutation)
        .expect("the per-chunk sum must close");
    assert_eq!(
        word_as_ext(&exec.public_words[0].1).expect("ext"),
        FEE::zero()
    );

    // Half two: every two-term reading DISAGREES. Dropping chunk 1 or chunk 2
    // is exactly what a per-family accumulator would do.
    for dropped in 0..3usize {
        let partial: FEE = contributions
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dropped)
            .map(|(_, c)| *c)
            .fold(FEE::zero(), |a, c| a + c);
        assert_ne!(
            partial,
            FEE::zero(),
            "dropping table {dropped} must break the balance, or the per-chunk \
             reading is not observable on this fixture"
        );
    }
    // …and the same statement in the MACHINE: a closure compiled for two
    // contributing tables — what a per-family emitter would build, one term per
    // family — fed a well-formed two-word arena, must not close.
    let family_shape = LogUpShape {
        num_contributing_tables: 2,
        num_output_bytes: 0,
    };
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(2);
    let cells: Vec<_> = (0..2u32).map(|i| b.hint_word(arena, i).as_ext()).collect();
    let zero = b.ext_const(&FEE::zero());
    emit_bus_closure(&mut b, &family_shape, &cells, zero);
    let per_family = compile(b.finish());
    for dropped in 1..3usize {
        let words: Vec<LfmWord> = contributions
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != dropped)
            .map(|(_, c)| ext_word(c))
            .collect();
        assert!(
            execute(&per_family, &[words], &TestPermutation).is_err(),
            "a per-family closure that folded chunk {dropped} away must not \
             close the bus"
        );
    }

    println!(
        "per-chunk accumulation witnessed: 3 chunks close, all 3 two-term \
         readings nonzero, and a per-family closure rejects both chunk drops"
    );
}

/// ★ `has_trace_interaction()` is SHAPE, and production checks the proof's
/// presence against it in BOTH directions.
///
/// `verifier.rs:1238` rejects an AIR with interactions whose proof carries no
/// bus public inputs, and `:1244` rejects the converse. So the count of
/// contributing tables is fixed by the AIR set, never read off the proof —
/// which is why [`LogUpShape::num_contributing_tables`] is a program constant.
/// A machine that sized its sum from the arena would let a prover drop a
/// table's contribution from the bus by omitting it.
#[test]
fn the_contributing_table_count_is_shape() {
    let (airs, proof) = chunked_family();
    for (i, air) in airs.iter().enumerate() {
        let view = StarkProofView::Owned(&proof.proofs[i]);
        assert!(
            air.has_trace_interaction(),
            "table {i} of this fixture declares interactions"
        );
        assert_eq!(
            air.has_trace_interaction(),
            view.has_bus_public_inputs(),
            "table {i}: production rejects any disagreement between the AIR's \
             declared interactions and the proof's bus public inputs, in both \
             directions — so the two can never disagree in a proof that verifies"
        );
    }

    // The shape is the AIR set's property. A program compiled for three
    // contributing tables cannot read a two-table arena: the schema mismatches.
    let shape = LogUpShape {
        num_contributing_tables: 3,
        num_output_bytes: 0,
    };
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(3);
    let cells: Vec<_> = (0..3u32).map(|i| b.hint_word(arena, i).as_ext()).collect();
    let zero = b.ext_const(&FEE::zero());
    emit_bus_closure(&mut b, &shape, &cells, zero);
    let program = compile(b.finish());
    assert!(
        execute(
            &program,
            &[vec![ext_word(&FEE::zero()); 2]],
            &TestPermutation
        )
        .is_err(),
        "a short arena must not satisfy a program compiled for three tables"
    );
    println!("contributing-table count is shape: a short arena is rejected");
}

// =============================================================================
// Degenerate parameter: a fixed table with NO rows, on a real epoch
// =============================================================================

/// How a fixed table's "no rows on the bus" claim is witnessed from its TRACE.
///
/// Needed because a zero-row table is not the same thing as a blank one. Two
/// padding conventions are in play, and taking either for the general case would
/// have mislabelled the other:
///
/// - `generate_keccak_rnd_trace` and ECSM's write nothing at all when there are
///   no operations, so their traces are literally zero.
/// - `generate_keccak_trace` pads with `state_ptr[lane] = 8·lane` (and KECCAK_RC
///   is a preprocessed constant table, ECDAS pads likewise). Those traces are
///   NOT zero, yet no row of them is on any bus, because every interaction's
///   multiplicity column is zero.
///
/// So the second form names the multiplicity columns. They come from each
/// table's own `bus_interactions()`, read there rather than through the AIR:
/// `&dyn AIR` does not expose the interaction list. KECCAK's eight interactions
/// and KECCAK_RND's fourteen are all `Multiplicity::Column(cols::MU)`, KECCAK_RC's
/// single one likewise, and ECDAS's three are `MU` twice plus `NEXT_OP` once.
/// ECSM's include `cols::k_bit(i)` as well, which is why the blank witness — the
/// stronger of the two — is the one used for it.
enum RowWitness {
    /// Every main cell is zero: the generator wrote nothing, so there is no row
    /// to participate in anything.
    Blank,
    /// Padding carries canonical values, so the trace is not blank. The witness
    /// is that every column any interaction uses as MULTIPLICITY is zero on
    /// every row, which gates every LogUp term off.
    GatedOff(&'static [usize]),
    /// This workload populates the table; no zero-row claim is made. Checked to
    /// be non-blank, so a misclassification here does not pass silently.
    Populated,
}

/// ★ MEASURED: a fixed table with no rows carries `Some(zero)`, never `None`.
///
/// ## Why this had to be measured
///
/// `FIXED_TABLE_COUNT` forces a sub-proof for all ten fixed tables whatever the
/// workload, so a real epoch always carries tables with no real rows. The
/// closure's [`LogUpShape::num_contributing_tables`] is a program CONSTANT, so
/// if such a table reported `None` the count would be workload-dependent and the
/// constant wrong. The LogUp leg closed with this labelled INFERENCE: production
/// rejects any AIR/proof disagreement (`verifier.rs:1238`), and
/// `has_trace_interaction()` is shape, so `None` would make every real epoch
/// unverifiable — therefore it must be `Some`. True, but an argument, and the
/// experiment is cheap. This is the experiment.
///
/// Note where the damage would have been: a zero `L` is arithmetically inert, so
/// dropping one would not move the SUM. What `None` would break is the arena
/// SCHEMA — a program compiled for `n` contributions fed `n − 1` words — which is
/// why the answer matters to the count and not to the balance.
///
/// ## What is measured, and against what
///
/// One REAL continuation epoch — epoch 0 of the LFM fixture guest, built by
/// `Traces::from_image_and_logs` and proved over the production epoch AIR set
/// (`VmAirs` + the epoch-local L2G table) under the real epoch statement, then
/// ACCEPTED by `Verifier::multi_verify_views` against production's own
/// `compute_expected_commit_bus_balance_view`. The acceptance is load-bearing
/// twice over: it is what makes this "what a verifying epoch proof carries"
/// rather than "what some prover run emitted", and it is what runs
/// `verifier.rs:1238`'s presence check over these very sub-proofs.
///
/// "No rows" is read off the TRACE, not inferred from the workload, and not read
/// back off the contribution being measured — see [`RowWitness`] for the two
/// forms it takes and why one would not do. `FIXED_TABLE_COUNT` keeps the
/// sub-proof either way: `generate_keccak_trace` pads a zero-operation table to
/// four rows rather than dropping it.
///
/// ## Which sub-proof is which table
///
/// Positional, because `VmAirs::new` builds these nine without `.with_name(…)` —
/// `AIR::name()` answers `"unknown"` for every one of them, so there is no name
/// on the proof side to match. The order is `lib.rs`'s own, and `air_refs()` and
/// `air_trace_pairs()` list it identically; that identity is what makes sub-proof
/// `i` this table's proof. It is not taken on trust: each position's sub-proof
/// must report the trace length that position's TRACE built.
///
/// ## What this test cannot see
///
/// The row-count cross-check cannot separate two tables of equal height, so
/// swapping (say) KECCAK and ECSM — both four rows here — would relabel two
/// results without failing. It catches the reorderings that change a height,
/// which is every one that could move a populated table into a zero-row slot.
///
/// Whether a fixed table whose interactions took a CONSTANT multiplicity would
/// answer differently. None does today — every multiplicity in the five zero-row
/// tables is a column, checked by reading their `bus_interactions()` — but such a
/// table would carry a nonzero `L` with no real rows, and this test would report
/// the changed contribution without explaining it. It measures an INTERMEDIATE
/// epoch, so HALT is out of scope (`VmAirs` omits it unless the epoch is final), and one workload, so it says nothing about
/// which tables are unused in general — only what a table with no rows carries.
#[test]
fn a_zero_row_fixed_table_carries_some_zero_not_none() {
    use crate::tables::trace_builder::{Traces, build_initial_image_paged};
    use crate::tables::{MaxRowsConfig, bitwise, local_to_global, register};
    use executor::elf::Elf;
    use executor::vm::execution::Executor;
    use math::field::traits::IsPrimeField;
    use stark::proof::view::MultiProofView;
    use stark::trace::TraceTable;

    let opts = super::proof_fixture::fixture_options();
    let elf_bytes = super::proof_fixture::read_inner_elf();
    let elf = Elf::load(&elf_bytes).expect("the fixture ELF must load");
    let epoch_size = 1usize << super::proof_fixture::FIXTURE_EPOCH_LOG2;

    // ---- epoch 0, built exactly as `prove_continuation` builds it ----
    let mut executor = Executor::new(&elf, vec![]).expect("executor");
    let image = build_initial_image_paged(&elf, &[]);
    let register_init = register::register_init_from_entry_point(elf.entry_point);
    let logs = executor
        .resume_with_limit(epoch_size)
        .expect("resume")
        .expect("the guest runs at least one epoch")
        .to_vec();
    let is_final = executor.pc() == 0;
    assert!(
        !is_final,
        "wanted an INTERMEDIATE epoch (every fixed table but HALT), but the \
         guest finished inside one epoch of {epoch_size} cycles"
    );

    let mut traces = Traces::from_image_and_logs(
        &elf,
        &image,
        &register_init,
        &logs,
        &MaxRowsConfig::default(),
        &[],
        is_final,
        true,
        #[cfg(feature = "disk-spill")]
        stark::storage_mode::StorageMode::Ram,
    )
    .expect("the epoch trace must build");

    let label = local_to_global::epoch_label(0);
    let mut provenance =
        local_to_global::genesis_provenance(image.iter().map(|(a, v)| (a, v as u64)));
    let boundary =
        local_to_global::epoch_boundary(&mut provenance, label, &traces.touched_memory_cells);
    // `prove_epoch`'s first act: the L2G table's range-check lookups must be
    // counted into BITWISE, or the epoch's own bus does not close.
    bitwise::update_multiplicities(
        &mut traces.bitwise,
        &local_to_global::collect_bitwise_from_l2g(&boundary),
    );

    // ---- the trace-side census, taken before proving borrows the traces ----
    let all_main_zero = |t: &TraceTable<Gl, Ext3>| -> bool {
        (0..t.main_table.height).all(|r| t.main_table.get_row(r).iter().all(|v| *v == FE::zero()))
    };
    let columns_zero = |t: &TraceTable<Gl, Ext3>, cols: &[usize]| -> bool {
        (0..t.main_table.height).all(|r| {
            let row = t.main_table.get_row(r);
            cols.iter().all(|c| row[*c] == FE::zero())
        })
    };
    // `(name, rows, has_no_bus_rows)`.
    //
    // ONE list, name and trace and witness together, deliberately: a version
    // that kept the names in a separate constant and zipped them onto the
    // traces passed with two names swapped — the swap moved only the label, so
    // the row-count cross-check below still compared the right trace against the
    // right sub-proof and saw nothing wrong. Merged, a reordering moves the
    // TRACE too, which that cross-check does catch.
    //
    // ★ The list is every always-on table an INTERMEDIATE epoch carries —
    // `FIXED_TABLE_COUNT` less HALT, which `VmAirs::air_refs` includes only on a
    // final epoch — **in `air_refs`' own order**, which the position-by-position
    // trace-length check below depends on. It carried NINE entries while the
    // constant was 11 and then 12: HINT and #903's BLAKE3 were never added, and
    // the shortfall was invisible because a fixture bug stopped this test
    // reaching its own assertion. A hand-written census that does not pin its
    // own length against the constant it is a census OF goes stale exactly that
    // way again, so the length is now asserted below.
    //
    // BLAKE3 has since left this list in the other direction: it is no longer
    // always-on but counted in `TableCounts::blake3`, and this epoch does not
    // use it, so it contributes no sub-proof to census.
    let census: Vec<(&str, usize, bool)> = {
        use crate::tables::{ecdas, hint, keccak, keccak_rc};
        let fixed: [(&str, &TraceTable<Gl, Ext3>, RowWitness); crate::FIXED_TABLE_COUNT - 1] = [
            ("BITWISE", &traces.bitwise, RowWitness::Populated),
            ("DECODE", &traces.decode, RowWitness::Populated),
            ("COMMIT", &traces.commit, RowWitness::Populated),
            (
                "KECCAK",
                &traces.keccak,
                RowWitness::GatedOff(&[keccak::cols::MU]),
            ),
            ("KECCAK_RND", &traces.keccak_rnd, RowWitness::Blank),
            (
                "KECCAK_RC",
                &traces.keccak_rc,
                RowWitness::GatedOff(&[keccak_rc::cols::MU]),
            ),
            ("ECSM", &traces.ecsm, RowWitness::Blank),
            (
                "ECDAS",
                &traces.ecdas,
                RowWitness::GatedOff(&[ecdas::cols::MU, ecdas::cols::NEXT_OP]),
            ),
            (
                "HINT",
                &traces.hint,
                RowWitness::GatedOff(&[hint::cols::MU]),
            ),
            ("REGISTER", &traces.register, RowWitness::Populated),
        ];
        fixed
            .into_iter()
            .map(|(name, t, witness)| {
                let no_bus_rows = match witness {
                    RowWitness::Blank => {
                        assert!(
                            all_main_zero(t),
                            "{name} was expected to have NO rows in this epoch \
                             (its generator writes nothing when there is no \
                             work), but its main trace is not all zero"
                        );
                        true
                    }
                    RowWitness::GatedOff(cols) => {
                        assert!(
                            columns_zero(t, cols),
                            "{name} was expected to have no rows on any bus, but \
                             one of its multiplicity columns {cols:?} is nonzero"
                        );
                        true
                    }
                    RowWitness::Populated => {
                        assert!(
                            !all_main_zero(t),
                            "{name} was classified as populated by this workload \
                             but its main trace is entirely zero — the \
                             classification, not the measurement, is wrong"
                        );
                        false
                    }
                };
                (name, t.num_rows(), no_bus_rows)
            })
            .collect()
    };

    // ---- prove it, over the production epoch AIR set ----
    let reg_fini = register::fini_from_trace(&traces.register);
    let table_counts = traces.table_counts();
    let public_output = traces.public_output_bytes.clone();
    let runtime_page_ranges = traces.runtime_page_ranges();

    let airs = crate::VmAirs::new(
        &elf,
        &opts,
        false,
        &[],
        &table_counts,
        None,
        is_final,
        None,
        None,
        Some((
            register::compute_precomputed_commitment_with_fini(&opts, &register_init, &reg_fini),
            register::NUM_PREPROCESSED_COLS_WITH_FINI,
        )),
    );
    let l2g_air = crate::continuation::l2g_memory_air(&opts, label);
    let mut l2g_trace = local_to_global::generate_local_to_global_trace(&boundary);

    // The real epoch statement, so the challenges are the ones a production
    // epoch proof is bound to.
    let seed = || {
        let mut t = crate::hash_pin::block_transcript(&[]);
        crate::statement::absorb_statement(
            &mut t,
            crate::statement::StatementKind::ContinuationEpoch { epoch_label: label },
            &elf_bytes,
            &public_output,
            &table_counts,
            0,
            &runtime_page_ranges,
            opts.fri_final_poly_log_degree,
        );
        t
    };

    let proof = {
        let mut pairs = airs.air_trace_pairs(&mut traces);
        pairs.push((&l2g_air, &mut l2g_trace, &()));
        crate::test_utils::multi_prove_ram(pairs, &mut seed()).expect("the epoch must prove")
    };

    let refs = {
        let mut r = airs.air_refs();
        r.push(&l2g_air);
        r
    };
    let view = MultiProofView::Owned(&proof);
    assert_eq!(
        census.len(),
        crate::FIXED_TABLE_COUNT - 1,
        "the census must name every always-on table an intermediate epoch \
         carries (all but HALT), or the sub-proof identity below is satisfied \
         by an undercount on both sides"
    );
    assert_eq!(
        view.len(),
        census.len() + table_counts.total() + 1,
        "an intermediate epoch is {} fixed tables, the chunked families, and \
         one L2G_MEMORY",
        crate::FIXED_TABLE_COUNT - 1
    );
    assert_eq!(refs.len(), view.len(), "one AIR per sub-proof");

    // ---- production must ACCEPT it, or nothing below is about a real proof ----
    let expected = crate::compute_expected_commit_bus_balance_view(
        &refs,
        view,
        &public_output,
        register_init[register::X254_INDEX] as u64,
        &mut seed(),
    )
    .expect("the COMMIT-bus target must exist");
    assert!(
        Verifier::multi_verify_views(&refs, view, &mut seed(), &expected),
        "production must ACCEPT this epoch proof — the measurement is about what \
         a VERIFYING proof carries, and this is also the run of \
         verifier.rs:1238's presence check"
    );

    // ---- THE MEASUREMENT ----
    println!(
        "\nreal continuation epoch (intermediate, {} sub-proofs), fixed tables:\n\
         \x20 {:<11} {:>9} {:>10} {:>5} {:>5} {:>4}  contribution",
        view.len(),
        "table",
        "rows",
        "proof_len",
        "iact",
        "bpi",
        "zero"
    );
    let mut zero_row = Vec::new();
    let mut with_rows = Vec::new();
    for (i, (name, rows, no_rows)) in census.iter().enumerate() {
        let sp = view.get(i);
        let interacts = refs[i].has_trace_interaction();
        let present = sp.has_bus_public_inputs();
        let contribution = sp.bus_table_contribution();
        assert_eq!(
            sp.trace_length(),
            *rows,
            "position {i} was labelled {name} but proved a trace of {} rows, not \
             the {rows} that table built — the census order no longer matches \
             air_refs()/air_trace_pairs()",
            sp.trace_length()
        );
        assert_eq!(
            interacts, present,
            "{name}: production rejects any disagreement between the AIR's \
             declared interactions and the proof's bus public inputs, in both \
             directions (verifier.rs:1238 and :1244)"
        );
        println!(
            "\x20 {:<11} {:>9} {:>10} {:>5} {:>5} {:>4}  {}",
            name,
            rows,
            sp.trace_length(),
            interacts,
            present,
            contribution.as_ref().is_some_and(|c| *c == FEE::zero()),
            match &contribution {
                None => "None".to_string(),
                Some(c) => format!(
                    "Some({:?})",
                    c.value()
                        .iter()
                        .map(|l| Gl::canonical(l.value()))
                        .collect::<Vec<_>>()
                ),
            }
        );
        if *no_rows {
            // THE ANSWER. A zero-row fixed table is still a contributing table.
            assert!(
                present,
                "{name} has no rows on any bus and must STILL carry bus public \
                 inputs — a None here would make num_contributing_tables \
                 workload-dependent, and the closure's program constant wrong"
            );
            assert_eq!(
                contribution,
                Some(FEE::zero()),
                "{name} has no rows on any bus, so every LogUp term is gated to \
                 zero and its L must be exactly zero"
            );
            zero_row.push(*name);
        } else {
            with_rows.push((*name, contribution));
        }
    }

    // Non-vacuity, both directions: there IS a zero-row fixed table in a real
    // epoch, and the observation distinguishes it from a populated one. Without
    // the second half, "every zero-row table reports Some(zero)" could hold
    // because every table reports Some(zero).
    assert!(
        !zero_row.is_empty(),
        "this epoch has no zero-row fixed table, so it cannot settle the \
         question — pick a guest that leaves one unused"
    );
    assert!(
        census
            .iter()
            .enumerate()
            .all(|(i, _)| view.get(i).has_bus_public_inputs()),
        "every fixed table of an epoch is a contributing table, populated or not"
    );
    assert!(
        with_rows.iter().any(|(_, c)| *c != Some(FEE::zero())),
        "no fixed table carries a NONZERO contribution, so Some(zero) is not a \
         distinguishing observation on this epoch: {with_rows:?}"
    );
    println!(
        "\x20 ANSWER: Some(zero), not None. {} zero-row fixed tables {:?}, each \
         Some(zero); {} populated, {} of them nonzero.",
        zero_row.len(),
        zero_row,
        with_rows.len(),
        with_rows
            .iter()
            .filter(|(_, c)| *c != Some(FEE::zero()))
            .count()
    );

    // ---- the converse, also measured: None would be REJECTED ----
    // The inference this experiment replaces ran the other way — a zero-row
    // table cannot report None, because production checks presence against
    // has_trace_interaction() before anything else (verifier.rs:1238), so a
    // None would make every real epoch unverifiable. That is now a run: strip
    // the bus public inputs off a zero-row sub-proof and watch this very proof
    // stop verifying. Only the `is_some` direction can be tested on an epoch —
    // all 25 sub-proofs declare interactions, so :1244's converse has no
    // subject here.
    for (i, (name, _, no_rows)) in census.iter().enumerate() {
        if !no_rows {
            continue;
        }
        let mut tampered = proof.clone();
        tampered.proofs[i].bus_public_inputs = None;
        assert!(
            !Verifier::multi_verify_views(
                &refs,
                MultiProofView::Owned(&tampered),
                &mut seed(),
                &expected,
            ),
            "{name} has no rows, but dropping its bus public inputs must still \
             be REJECTED — that rejection is why Some(zero) is forced rather \
             than merely observed"
        );
    }

    // ---- and the closure itself, over the REAL epoch's whole table set ----
    // The handoff's other open item: every earlier fixture is two or three
    // tables, so the SUM was exercised but its LENGTH was not.
    let contributions: Vec<FEE> = (0..view.len())
        .filter(|i| refs[*i].has_trace_interaction())
        .map(|i| {
            view.get(i)
                .bus_table_contribution()
                .expect("presence was just checked against the AIR")
        })
        .collect();
    let shape = LogUpShape {
        num_contributing_tables: contributions.len(),
        num_output_bytes: public_output.len(),
    };
    let (z, alpha) = crate::replay_transcript_phase_a_view(&refs, view, &mut seed());

    let n_tables = contributions.len() as u32;
    let n_bytes = public_output.len() as u32;
    let mut b = LfmBuilder::new();
    // Only the cells the gadget reads may be declared: an unread arena word is a
    // compile error, and an empty output makes the target a constant that reads
    // neither z, alpha, start nor any byte.
    let head = if n_bytes == 0 { 0 } else { 3 + n_bytes };
    let arena = b.declare_arena(head + n_tables);
    let target = if n_bytes == 0 {
        b.ext_const(&FEE::zero())
    } else {
        let z_cell = b.hint_word(arena, 0).as_ext();
        let alpha_cell = b.hint_word(arena, 1).as_ext();
        let start_cell = b.hint_felt(arena, 2);
        let byte_cells: Vec<_> = (0..n_bytes).map(|i| b.hint_felt(arena, 3 + i)).collect();
        emit_commit_bus_target(&mut b, &shape, z_cell, alpha_cell, start_cell, &byte_cells)
    };
    let contrib_cells: Vec<_> = (0..n_tables)
        .map(|i| b.hint_word(arena, head + i).as_ext())
        .collect();
    let total = emit_bus_closure(&mut b, &shape, &contrib_cells, target);
    b.public(total.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("the epoch closure program is admissible");

    let mut words: Vec<LfmWord> = Vec::new();
    if n_bytes > 0 {
        words.push(ext_word(&z));
        words.push(ext_word(&alpha));
        words.push(base_word(FE::from(
            register_init[register::X254_INDEX] as u64,
        )));
        words.extend(public_output.iter().map(|v| base_word(FE::from(*v as u64))));
    }
    words.extend(contributions.iter().map(ext_word));
    let exec = execute(&program, std::slice::from_ref(&words), &TestPermutation)
        .expect("a real epoch's LogUp bus must close in the machine");
    assert_eq!(
        word_as_ext(&exec.public_words[0].1).expect("ext"),
        expected,
        "the machine's published total must be production's own expected balance"
    );

    // Falsification: move any one contribution, in any lane. Every zero-row
    // table is in here too, so this is also the check that a Some(zero) term is
    // a real summand and not a no-op the emitter could drop.
    let mut vectors = 0usize;
    for table in 0..contributions.len() {
        for lane in 0..3usize {
            let mut arenas = vec![words.clone()];
            arenas[0][head as usize + table][lane] += FE::one();
            assert!(
                execute(&program, &arenas, &TestPermutation).is_err(),
                "table {table} lane {lane}: a moved contribution must not close"
            );
            vectors += 1;
        }
    }
    println!(
        "\x20 the closure also runs on the real epoch: {} contributing tables of \
         {} sub-proofs, {} output bytes, {vectors} single-lane moves rejected\n",
        contributions.len(),
        view.len(),
        public_output.len()
    );
}
