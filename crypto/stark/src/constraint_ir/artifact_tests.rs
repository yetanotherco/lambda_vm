//! Unit tests for [`ConstraintArtifact`]: the codec, the lift back to a
//! [`ConstraintProgram`], the self-consistency and shape checks, and the
//! pre-captured supply path.
//!
//! The per-table bit-exactness sweep over all 25 production AIRs lives in the
//! prover crate (`prover/src/tests/constraint_artifact_tests.rs`) — it needs the
//! production tables. What is here is what the production tables CANNOT cover:
//!
//! - **Nonzero `end_exemptions`.** Every production constraint applies to every
//!   row (`RowDomain::ALL`); nothing under `prover/src` uses
//!   `RowDomain::except_last`. So the production sweep would exercise the
//!   artifact's zerofier metadata only in its all-zero case, which proves
//!   nothing about the field that capture actually discards. The AIR below has
//!   exemptions on purpose.
//! - **Rejection.** A suite of AIRs that all validate proves the checks accept;
//!   it does not prove they can reject. The falsification tests here corrupt an
//!   artifact in each way `validate_self` claims to catch and assert it does.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsField;

use super::artifact::{ArtifactError, ArtifactMeta, ConstraintArtifact};
use crate::constraints::builder::{ConstraintBuilder, ConstraintSet, RowDomain};
use crate::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use crate::proof::options::GoldilocksCubicProofOptions;
use crate::traits::AIR;

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

const NUM_COLS: usize = 4;

/// A constraint set whose three constraints have DIFFERENT row domains, so the
/// artifact's `end_exemptions` is non-uniform and a bug that dropped, zeroed, or
/// permuted it would be visible. No production table does this today.
struct ExemptConstraints;

impl<F: IsField, E: IsField> ConstraintSet<F, E> for ExemptConstraints {
    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let m0 = b.main(0, 0);
        let m1 = b.main(0, 1);
        let m2 = b.main(0, 2);
        let n0 = b.main(1, 0);
        let n1 = b.main(1, 1);

        // c0: every row — a degree-2 product.
        b.emit_base(0, m0.clone() * m1.clone() - m2.clone());
        // c1: skips the last row (reads the next row).
        b.emit_base_rows(1, RowDomain::except_last(1), n0.clone() - m0.clone());
        // c2: skips the last two rows.
        b.emit_base_rows(2, RowDomain::except_last(2), n1 - m1);
    }
}

fn options() -> crate::proof::options::ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

/// The exemption-bearing AIR, with no bus interactions so its constraints are
/// exactly the three above (no LogUp suffix).
fn exempt_air() -> AirWithBuses<Gl, Ext3, NullBoundaryConstraintBuilder, (), ExemptConstraints> {
    AirWithBuses::new(
        NUM_COLS,
        AuxiliaryTraceBuildData {
            interactions: vec![],
        },
        &options(),
        1,
        ExemptConstraints,
    )
    .with_name("EXEMPT")
}

// =============================================================================
// The zerofier metadata capture discards
// =============================================================================

#[test]
fn artifact_carries_the_end_exemptions_capture_discards() {
    let air = exempt_air();
    let artifact = ConstraintArtifact::capture(&air);

    // The three constraints' row domains, which the ConstraintProgram alone has
    // no field for.
    let exemptions: Vec<u32> = artifact.meta.iter().map(|m| m.end_exemptions).collect();
    assert_eq!(
        exemptions,
        vec![0, 1, 2],
        "the artifact must preserve each constraint's row domain"
    );
    assert_eq!(artifact.constraints_meta(), air.constraints_meta().to_vec());

    // ... and it must survive the wire.
    let bytes = artifact.to_bytes().expect("serialize");
    let back = ConstraintArtifact::from_bytes(&bytes).expect("deserialize");
    assert_eq!(back, artifact, "artifact must round-trip exactly");
    assert_eq!(
        back.meta
            .iter()
            .map(|m| m.end_exemptions)
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
}

#[test]
fn zeroed_exemptions_are_rejected_against_the_air() {
    let air = exempt_air();
    let mut artifact = ConstraintArtifact::capture(&air);
    assert!(artifact.validate_against(&air).is_ok());

    // Drop the zerofier shapes — the exact damage that makes a serialized
    // program evaluate the right algebra against the wrong divisor.
    for m in &mut artifact.meta {
        m.end_exemptions = 0;
    }
    let err = artifact
        .validate_against(&air)
        .expect_err("an artifact with the row domains flattened must be rejected");
    assert!(
        matches!(
            err,
            ArtifactError::ShapeMismatch {
                field: "constraints_meta",
                ..
            }
        ),
        "expected a constraints_meta mismatch, got {err:?}"
    );
}

// =============================================================================
// Lift / codec
// =============================================================================

#[test]
fn lift_is_the_inverse_of_lower() {
    let air = exempt_air();
    let captured = air.constraint_program();
    let artifact = ConstraintArtifact::capture(&air);
    let lifted = artifact.program();

    assert_eq!(
        lifted.nodes, captured.nodes,
        "nodes must survive the round trip"
    );
    assert_eq!(
        lifted.dims, captured.dims,
        "dims must survive the round trip"
    );
    assert_eq!(lifted.roots, captured.roots);
    assert_eq!(lifted.num_base, captured.num_base);
    assert_eq!(lifted.base_consts, captured.base_consts);
    assert_eq!(lifted.ext_consts, captured.ext_consts);

    // And through the wire, not just in memory.
    let bytes = artifact.to_bytes().expect("serialize");
    let lifted2 = ConstraintArtifact::from_bytes(&bytes)
        .expect("deserialize")
        .program();
    assert_eq!(lifted2.nodes, captured.nodes);
    assert_eq!(lifted2.dims, captured.dims);
    assert_eq!(lifted2.base_consts, captured.base_consts);
}

#[test]
fn constants_survive_as_exact_field_values() {
    // Constants go out as raw limbs and come back through `from_raw`; a
    // canonicalization slip there would change a constraint's arithmetic
    // silently, so pin the values rather than only their count.
    let air = exempt_air();
    let artifact = ConstraintArtifact::capture(&air);
    let lifted = artifact.program();
    for (i, (a, b)) in lifted
        .base_consts
        .iter()
        .zip(air.constraint_program().base_consts.iter())
        .enumerate()
    {
        assert_eq!(a, b, "base_consts[{i}] changed value across the round trip");
        assert_eq!(
            a.value(),
            b.value(),
            "base_consts[{i}] changed representation across the round trip"
        );
    }
}

// =============================================================================
// Falsification: every check must be able to reject
// =============================================================================

#[test]
fn validate_self_rejects_a_forward_reference() {
    let air = exempt_air();
    let mut artifact = ConstraintArtifact::capture(&air);
    // Point the last node's operand at itself: no longer topologically ordered,
    // which an interpreter would read as an uninitialized value.
    let last = artifact.nodes.len() - 1;
    artifact.nodes[last].a = last as u32;
    let err = artifact
        .validate_self()
        .expect_err("a self-referential node must be rejected");
    assert!(matches!(err, ArtifactError::Malformed(_)), "got {err:?}");
}

#[test]
fn validate_self_rejects_an_out_of_range_root() {
    let air = exempt_air();
    let mut artifact = ConstraintArtifact::capture(&air);
    artifact.roots[0] = artifact.nodes.len() as u32;
    assert!(matches!(
        artifact.validate_self(),
        Err(ArtifactError::Malformed(_))
    ));
}

#[test]
fn validate_self_rejects_an_out_of_range_constant() {
    let air = exempt_air();
    let mut artifact = ConstraintArtifact::capture(&air);
    let n_base = artifact.base_consts.len() as u32;
    let node = artifact
        .nodes
        .iter_mut()
        .find(|n| n.op == super::device::OP_CONST_BASE)
        .expect("the program reads at least one base constant");
    node.a = n_base;
    assert!(matches!(
        artifact.validate_self(),
        Err(ArtifactError::Malformed(_))
    ));
}

#[test]
fn validate_self_rejects_a_non_prefix_base_kind() {
    let air = exempt_air();
    let mut artifact = ConstraintArtifact::capture(&air);
    // All three constraints are base-rooted here, so flipping one to Ext breaks
    // the "Base entries form a prefix of length num_base" invariant that
    // `num_base_from_meta` relies on.
    artifact.meta[0].kind = ArtifactMeta::KIND_EXT;
    assert!(matches!(
        artifact.validate_self(),
        Err(ArtifactError::Malformed(_))
    ));
}

#[test]
fn validate_self_rejects_permuted_metadata() {
    let air = exempt_air();
    let mut artifact = ConstraintArtifact::capture(&air);
    artifact.meta.swap(0, 2);
    assert!(matches!(
        artifact.validate_self(),
        Err(ArtifactError::Malformed(_))
    ));
}

#[test]
fn validate_against_rejects_a_shape_change() {
    let air = exempt_air();
    let mut artifact = ConstraintArtifact::capture(&air);
    assert!(artifact.validate_against(&air).is_ok());

    artifact.shape.main_width += 1;
    let err = artifact
        .validate_against(&air)
        .expect_err("a width change must be rejected");
    assert!(
        matches!(
            err,
            ArtifactError::ShapeMismatch {
                field: "main_width",
                ..
            }
        ),
        "got {err:?}"
    );
}

#[test]
fn from_bytes_rejects_a_corrupt_artifact() {
    // The codec must not hand back a structurally invalid artifact just because
    // the bytes deserialized: `from_bytes` runs `validate_self`.
    let air = exempt_air();
    let mut artifact = ConstraintArtifact::capture(&air);
    artifact.roots[0] = 9999;
    let bytes = artifact.to_bytes().expect("serialize");
    assert!(matches!(
        ConstraintArtifact::from_bytes(&bytes),
        Err(ArtifactError::Malformed(_))
    ));
}

// =============================================================================
// The pre-captured supply path (the scoped verify-path unban)
// =============================================================================

#[test]
fn precaptured_is_none_even_after_a_capture() {
    let air = exempt_air();
    assert!(
        air.precaptured_constraint_program().is_none(),
        "a freshly built AIR has no build-time program"
    );

    // Force a capture. This fills the AIR's OnceLock — but a captured program is
    // NOT a build-time artifact, and the guest-safe accessor must keep saying so.
    let _ = air.constraint_program();
    assert!(
        air.precaptured_constraint_program().is_none(),
        "a program the AIR captured at runtime must never be reported as pre-captured; \
         conflating the two would let a guest path believe capture had been avoided"
    );
}

#[test]
fn supplying_a_program_short_circuits_capture() {
    let program = ConstraintArtifact::capture(&exempt_air()).program();
    let air = exempt_air().with_precaptured(program);

    let supplied = air
        .precaptured_constraint_program()
        .expect("the supplied program must be visible");

    // Pointer identity is the actual proof that no capture ran: a capture would
    // have built a fresh program in the OnceLock and returned that instead.
    assert!(
        std::ptr::eq(air.constraint_program(), supplied),
        "constraint_program() must hand back the supplied program itself, not a fresh capture"
    );
}

#[test]
fn a_supplied_program_still_evaluates_correctly() {
    // Supplying a program must not change what the AIR computes.
    let program = ConstraintArtifact::capture(&exempt_air()).program();
    let air = exempt_air().with_precaptured(program);
    let artifact = ConstraintArtifact::capture(&air);
    assert!(artifact.validate_against(&air).is_ok());
    assert_eq!(
        artifact.program().nodes,
        exempt_air().constraint_program().nodes,
        "a supplied program must be the same program the AIR would have captured"
    );
}

#[test]
#[should_panic(expected = "roots")]
fn supplying_a_mismatched_program_panics() {
    // A program for a different constraint count must not be installable.
    let mut program = ConstraintArtifact::capture(&exempt_air()).program();
    program.roots.pop();
    let _ = exempt_air().with_precaptured(program);
}

// =============================================================================
// Degree bound
// =============================================================================

#[test]
fn composition_degree_multiplier_reproduces_the_bound() {
    let air = exempt_air();
    let artifact = ConstraintArtifact::capture(&air);
    let k = artifact.shape.composition_degree_multiplier as usize;
    assert!(k >= 1, "the multiplier must be positive");
    for log_n in [8usize, 12, 20] {
        let n = 1usize << log_n;
        assert_eq!(
            air.composition_poly_degree_bound(n),
            k * n,
            "the stored multiplier must reproduce the AIR's own bound at n=2^{log_n}"
        );
    }
}

/// A sanity floor on the field type used for constants, so a field swap does not
/// silently reinterpret the artifact's raw limbs.
#[test]
fn base_constants_are_goldilocks_limbs() {
    let air = exempt_air();
    let artifact = ConstraintArtifact::capture(&air);
    for &c in &artifact.base_consts {
        let fe = FieldElement::<Gl>::from_raw(c);
        assert_eq!(
            *fe.value(),
            c,
            "constant {c} is not a canonical Goldilocks limb"
        );
    }
}
