//! Build-time serialization of an AIR's transition constraints — "constraints
//! as data".
//!
//! [`DeviceProgram`](super::device::DeviceProgram) already flattens a captured
//! [`ConstraintProgram`] into POD arrays, but a program alone does not describe
//! an AIR's transition constraints: evaluating the roots is only part of the
//! job. A consumer also needs each constraint's ZEROFIER shape (which capture
//! discards) and the AIR's shape scalars. This module bundles all of it into one
//! serializable [`ConstraintArtifact`].
//!
//! # Why the bundle is four things, not one
//!
//! Derived from what [`crate::verifier`] actually calls on an AIR, not from the
//! trait surface:
//!
//! 1. **The program** — `nodes` / `base_consts` / `ext_consts` / `roots` /
//!    `num_base`, exactly [`DeviceProgram`]'s flat form. Replaces
//!    `AIR::compute_transition`.
//! 2. **Per-constraint metadata** — `{kind, end_exemptions}` per constraint.
//!    Capture DISCARDS this: [`ConstraintProgram`] records each constraint's
//!    root but not the row domain it applies to, and `end_exemptions` is what
//!    picks the constraint's zerofier
//!    (`AIR::transition_zerofier_evaluations_grouped` keys its dedup groups on
//!    exactly this field). A program without it evaluates the right algebra
//!    against the wrong divisor.
//! 3. **The AIR shape** — widths, step size, transition offsets, the next-row
//!    column set (which decides the pruned `g·z` OOD opening), max bus elements.
//! 4. **The composition degree multiplier** — see
//!    [`AirShape::composition_degree_multiplier`]. This one is easy to miss: it
//!    lives in neither [`AirContext`](crate::context::AirContext) nor
//!    [`ConstraintMeta`], only inside the `ConstraintSet` impl and the LogUp
//!    layout, yet the verifier needs it to size the composition polynomial.
//!
//! # What is deliberately NOT in the bundle
//!
//! - **`ProofOptions`.** `AirContext` bundles the proof options in with the
//!   shape scalars, but a captured program does not depend on them (pinned by
//!   the blowup-invariance test in the prover's artifact suite). Storing them
//!   would multiply the artifact count by the number of blowup factors for no
//!   information gain, and would wrongly imply the constraints are
//!   options-dependent. Options are supplied at AIR construction.
//! - **Boundary constraints.** `AIR::boundary_constraints` is a function of the
//!   public inputs, not a static property of the AIR, so it is not data in the
//!   sense this artifact means. Serializing it is a separate problem.
//! - **Derived scalars.** `has_aux_trace` and `num_auxiliary_rap_columns` are
//!   pure functions of `trace_layout`; `num_transition_constraints` is
//!   `roots.len()`. Storing a second copy only creates a way for the two to
//!   disagree.
//!
//! # Guest safety
//!
//! [`ConstraintArtifact::capture`] CAPTURES — it calls
//! `AIR::constraint_program`, which hash-conses. It is a build-time entry point
//! and must never run in a guest. Everything else here (deserialize,
//! [`ConstraintArtifact::program`], [`ConstraintArtifact::validate_against`]) is
//! pure data handling and is guest-safe: that asymmetry is the whole point of
//! the artifact.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as GoldilocksExtension;
use math::field::goldilocks::GoldilocksField;

use super::device::{DeviceNode, DeviceProgram};
use super::ir::{ConstraintProgram, Dim, Op};
use crate::constraints::builder::{ConstraintMeta, RootKind};
use crate::traits::AIR;

type Gl = GoldilocksField;
type Ext3 = GoldilocksExtension;

/// Trace lengths used to probe `AIR::composition_poly_degree_bound`. Two of
/// them, so [`ConstraintArtifact::capture`] can check the bound really is linear
/// in the trace length rather than assuming it.
const DEGREE_PROBE_LEN: usize = 1 << 10;
const DEGREE_PROBE_LEN_2: usize = 1 << 11;

// =============================================================================
// Metadata
// =============================================================================

/// [`ConstraintMeta`] as plain serializable data.
///
/// `kind` is encoded as a `u8` (see [`ArtifactMeta::KIND_BASE`] /
/// [`ArtifactMeta::KIND_EXT`]) rather than reusing [`RootKind`] so the wire
/// encoding is pinned independently of the in-memory enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ArtifactMeta {
    /// Constraint index. Stored rather than implied by position so
    /// [`ConstraintArtifact::validate_self`] can CHECK the dense-and-ordered
    /// invariant instead of silently depending on it.
    pub constraint_idx: u32,
    /// [`ArtifactMeta::KIND_BASE`] or [`ArtifactMeta::KIND_EXT`].
    pub kind: u8,
    /// Exempted rows at the end of the trace — the constraint's zerofier shape.
    pub end_exemptions: u32,
}

impl ArtifactMeta {
    /// Base-field rooted constraint.
    pub const KIND_BASE: u8 = 0;
    /// Extension-field (LogUp) rooted constraint.
    pub const KIND_EXT: u8 = 1;

    fn from_meta(m: &ConstraintMeta) -> Self {
        // An exhaustive match, so adding a RootKind variant is a build error
        // here rather than a silently wrong wire byte.
        let kind = match m.kind {
            RootKind::Base => Self::KIND_BASE,
            RootKind::Ext => Self::KIND_EXT,
        };
        Self {
            constraint_idx: m.constraint_idx as u32,
            kind,
            end_exemptions: m.end_exemptions as u32,
        }
    }

    /// Back to a [`ConstraintMeta`]. Panics on an unknown `kind` byte — a
    /// corrupt artifact must not silently become a base-field constraint.
    pub fn to_meta(self) -> ConstraintMeta {
        let kind = match self.kind {
            Self::KIND_BASE => RootKind::Base,
            Self::KIND_EXT => RootKind::Ext,
            other => panic!("unknown ArtifactMeta kind byte {other}"),
        };
        ConstraintMeta {
            constraint_idx: self.constraint_idx as usize,
            kind,
            end_exemptions: self.end_exemptions as usize,
        }
    }
}

// =============================================================================
// Shape
// =============================================================================

/// An AIR's transition-constraint shape: everything the verifier reads off the
/// AIR that is neither the program nor per-constraint metadata.
#[derive(Clone, Debug, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct AirShape {
    /// `AIR::step_size`.
    pub step_size: u32,
    /// `AIR::trace_layout().0` — main trace width.
    pub main_width: u32,
    /// `AIR::trace_layout().1` — aux trace width.
    pub aux_width: u32,
    /// `AirContext::transition_offsets` — the frame's row offsets.
    pub transition_offsets: Vec<u32>,
    /// `AIR::trace_ood_next_row_columns`, sorted and deduplicated: the
    /// full-width `[main | aux]` columns opened at `g·z`. Every other column is
    /// reconstructed as ZERO at the next row, so this set is soundness-critical.
    pub next_row_columns: Vec<u32>,
    /// `AIR::max_bus_elements` — decides the LogUp alpha-power count.
    pub max_bus_elements: u32,
    /// `AIR::has_trace_interaction`.
    pub has_trace_interaction: bool,
    /// `AIR::is_preprocessed`.
    pub is_preprocessed: bool,
    /// `AIR::num_precomputed_columns`.
    pub num_precomputed_columns: u32,
    /// `composition_poly_degree_bound(n) / n` — the trace-length-INDEPENDENT
    /// part of the composition degree bound.
    ///
    /// Stored as this observable rather than as the underlying `max_degree`
    /// because `max_degree` is not exposed on the `AIR` trait at all: it is
    /// `max(ConstraintSet::max_degree(), logup_max_degree(layout))`, private to
    /// the AIR's construction. The multiplier is what the verifier consumes, it
    /// is directly measurable through the public trait, and it needs no new
    /// trait method.
    pub composition_degree_multiplier: u32,
}

// =============================================================================
// The artifact
// =============================================================================

/// A build-time-serializable bundle of one AIR's transition constraints.
///
/// Produced by [`ConstraintArtifact::capture`] (build time, captures) and
/// consumed by [`ConstraintArtifact::program`] (guest-safe, pure data).
#[derive(Clone, Debug, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct ConstraintArtifact {
    /// Topologically ordered flat instruction list (id `i` references only
    /// `< i`) — [`DeviceProgram::nodes`].
    pub nodes: Vec<DeviceNode>,
    /// Base-field constant table, raw canonical limbs.
    pub base_consts: Vec<u64>,
    /// Extension-field constant table, raw canonical limbs.
    pub ext_consts: Vec<[u64; 3]>,
    /// Per-constraint root node ids.
    pub roots: Vec<u32>,
    /// Number of leading base-field-rooted constraints.
    pub num_base: u32,
    /// Idx-ordered, dense per-constraint metadata.
    pub meta: Vec<ArtifactMeta>,
    /// The AIR's shape scalars.
    pub shape: AirShape,
}

/// Why an artifact was rejected.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ArtifactError {
    /// The artifact is internally inconsistent.
    #[error("malformed constraint artifact: {0}")]
    Malformed(String),
    /// The artifact does not describe the AIR it was checked against.
    #[error(
        "constraint artifact does not match this AIR: {field} is {found} in the artifact but {expected} on the AIR"
    )]
    ShapeMismatch {
        /// The disagreeing field.
        field: &'static str,
        /// The artifact's value.
        found: String,
        /// The AIR's value.
        expected: String,
    },
    /// Serialization or deserialization failed.
    #[error("constraint artifact codec error: {0}")]
    Codec(String),
}

/// Compare one shape field, producing a [`ArtifactError::ShapeMismatch`].
fn check_field<T>(field: &'static str, found: T, expected: T) -> Result<(), ArtifactError>
where
    T: PartialEq + core::fmt::Debug,
{
    if found == expected {
        Ok(())
    } else {
        Err(ArtifactError::ShapeMismatch {
            field,
            found: format!("{found:?}"),
            expected: format!("{expected:?}"),
        })
    }
}

impl ConstraintArtifact {
    /// Capture an AIR's constraints into a serializable artifact.
    ///
    /// BUILD TIME ONLY: this calls `AIR::constraint_program`, which hash-conses
    /// the whole constraint body. Never call it from a verifier or a guest —
    /// that is precisely what the artifact exists to avoid.
    ///
    /// # Panics
    ///
    /// If `composition_poly_degree_bound` is not exactly linear in the trace
    /// length, since the artifact stores only the linear coefficient. Better a
    /// loud failure at build time than an artifact that silently misstates the
    /// composition bound.
    pub fn capture<A>(air: &A) -> Self
    where
        A: AIR<Field = Gl, FieldExtension = Ext3> + ?Sized,
    {
        let prog = air.constraint_program();
        let dev = DeviceProgram::lower(prog);

        let (main_width, aux_width) = air.trace_layout();

        let mut next_row_columns: Vec<u32> = air
            .trace_ood_next_row_columns()
            .into_iter()
            .map(|c| c as u32)
            .collect();
        next_row_columns.sort_unstable();
        next_row_columns.dedup();

        // The composition bound is `n * k`; recover `k` and check linearity
        // across two probe lengths rather than trusting one sample.
        let b1 = air.composition_poly_degree_bound(DEGREE_PROBE_LEN);
        let b2 = air.composition_poly_degree_bound(DEGREE_PROBE_LEN_2);
        assert_eq!(
            b1 % DEGREE_PROBE_LEN,
            0,
            "composition_poly_degree_bound({DEGREE_PROBE_LEN}) = {b1} is not a multiple of the \
             trace length; the artifact cannot store it as a linear multiplier"
        );
        let multiplier = b1 / DEGREE_PROBE_LEN;
        assert_eq!(
            b2,
            multiplier * DEGREE_PROBE_LEN_2,
            "composition_poly_degree_bound is not linear in the trace length ({b1} at \
             {DEGREE_PROBE_LEN}, {b2} at {DEGREE_PROBE_LEN_2}); the artifact's single \
             multiplier cannot represent it"
        );

        Self {
            nodes: dev.nodes,
            base_consts: dev.base_consts,
            ext_consts: dev.ext_consts,
            roots: dev.roots,
            num_base: dev.num_base,
            meta: air
                .constraints_meta()
                .iter()
                .map(ArtifactMeta::from_meta)
                .collect(),
            shape: AirShape {
                step_size: air.step_size() as u32,
                main_width: main_width as u32,
                aux_width: aux_width as u32,
                transition_offsets: air
                    .context()
                    .transition_offsets
                    .iter()
                    .map(|o| *o as u32)
                    .collect(),
                next_row_columns,
                max_bus_elements: air.max_bus_elements() as u32,
                has_trace_interaction: air.has_trace_interaction(),
                is_preprocessed: air.is_preprocessed(),
                num_precomputed_columns: air.num_precomputed_columns() as u32,
                composition_degree_multiplier: multiplier as u32,
            },
        }
    }

    /// The flat device form — the inverse direction of the arrays this artifact
    /// stores, at zero cost.
    pub fn device_program(&self) -> DeviceProgram {
        DeviceProgram {
            nodes: self.nodes.clone(),
            base_consts: self.base_consts.clone(),
            ext_consts: self.ext_consts.clone(),
            roots: self.roots.clone(),
            num_base: self.num_base,
        }
    }

    /// Lift back to a [`ConstraintProgram`] — the exact inverse of
    /// [`DeviceProgram::lower`], so the generic CPU interpreters
    /// ([`eval_program`](super::interp::eval_program) /
    /// [`eval_program_verifier`](super::interp::eval_program_verifier)) can run
    /// a deserialized artifact.
    ///
    /// Guest-safe: a linear walk over POD arrays, no capture and no hashing.
    ///
    /// # Panics
    ///
    /// On an unknown op tag or dim tag — a corrupt program must not evaluate to
    /// something plausible.
    pub fn program(&self) -> ConstraintProgram<Gl, Ext3> {
        use super::device::{
            DIM_BASE, DIM_EXT, OP_ADD, OP_ALPHA_POW, OP_CONST_BASE, OP_CONST_EXT, OP_EMBED, OP_MUL,
            OP_NEG, OP_RAP_CHALLENGE, OP_SUB, OP_TABLE_OFFSET, OP_VAR, unpack_var,
        };

        let mut nodes = Vec::with_capacity(self.nodes.len());
        let mut dims = Vec::with_capacity(self.nodes.len());

        for (i, n) in self.nodes.iter().enumerate() {
            let op = match n.op {
                OP_CONST_BASE => Op::ConstBase(n.a),
                OP_CONST_EXT => Op::ConstExt(n.a),
                OP_VAR => {
                    let (main, offset, row, col) = unpack_var(n.a, n.b);
                    Op::Var {
                        main,
                        offset,
                        row,
                        col,
                    }
                }
                OP_RAP_CHALLENGE => Op::RapChallenge { idx: n.a as u16 },
                OP_ALPHA_POW => Op::AlphaPow { idx: n.a as u16 },
                OP_TABLE_OFFSET => Op::TableOffset,
                OP_ADD => Op::Add(n.a, n.b),
                OP_SUB => Op::Sub(n.a, n.b),
                OP_MUL => Op::Mul(n.a, n.b),
                OP_NEG => Op::Neg(n.a),
                OP_EMBED => Op::Embed(n.a),
                other => panic!("unknown op tag {other} at node {i}"),
            };
            let dim = match n.dim {
                DIM_BASE => Dim::Base,
                DIM_EXT => Dim::Ext,
                other => panic!("unknown dim tag {other} at node {i}"),
            };
            nodes.push(op);
            dims.push(dim);
        }

        ConstraintProgram {
            nodes,
            dims,
            base_consts: self
                .base_consts
                .iter()
                .map(|c| FieldElement::<Gl>::from_raw(*c))
                .collect(),
            ext_consts: self
                .ext_consts
                .iter()
                .map(|limbs| {
                    FieldElement::<Ext3>::from_raw([
                        FieldElement::<Gl>::from_raw(limbs[0]),
                        FieldElement::<Gl>::from_raw(limbs[1]),
                        FieldElement::<Gl>::from_raw(limbs[2]),
                    ])
                })
                .collect(),
            roots: self.roots.clone(),
            num_base: self.num_base as usize,
        }
    }

    /// The per-constraint metadata as the engine's own type.
    pub fn constraints_meta(&self) -> Vec<ConstraintMeta> {
        self.meta.iter().map(|m| m.to_meta()).collect()
    }

    /// Internal consistency: the invariants a consumer would otherwise assume.
    ///
    /// Checks that node operands are topologically ordered and in range, that
    /// constant/root indices are in range, and that the metadata list is dense,
    /// idx-ordered and has its `Base` entries as a prefix of length `num_base`.
    pub fn validate_self(&self) -> Result<(), ArtifactError> {
        use super::device::{
            DIM_BASE, DIM_EXT, OP_ADD, OP_ALPHA_POW, OP_CONST_BASE, OP_CONST_EXT, OP_EMBED, OP_MUL,
            OP_NEG, OP_RAP_CHALLENGE, OP_SUB, OP_TABLE_OFFSET, OP_VAR,
        };
        let bad = |m: String| Err(ArtifactError::Malformed(m));

        for (i, n) in self.nodes.iter().enumerate() {
            if n.dim != DIM_BASE && n.dim != DIM_EXT {
                return bad(format!("node {i} has unknown dim tag {}", n.dim));
            }
            // Operand ids must reference strictly earlier nodes; constant and
            // uniform indices must be in range for their tables.
            let check_id = |x: u32| -> Result<(), ArtifactError> {
                if (x as usize) < i {
                    Ok(())
                } else {
                    Err(ArtifactError::Malformed(format!(
                        "node {i} references node {x}, which is not strictly earlier"
                    )))
                }
            };
            match n.op {
                OP_CONST_BASE => {
                    if n.a as usize >= self.base_consts.len() {
                        return bad(format!(
                            "node {i} reads base_consts[{}] of {}",
                            n.a,
                            self.base_consts.len()
                        ));
                    }
                }
                OP_CONST_EXT => {
                    if n.a as usize >= self.ext_consts.len() {
                        return bad(format!(
                            "node {i} reads ext_consts[{}] of {}",
                            n.a,
                            self.ext_consts.len()
                        ));
                    }
                }
                // Var/challenge/alpha/table-offset index per-proof inputs whose
                // lengths are not part of the artifact; range-checking them is
                // the caller's job at evaluation time.
                OP_VAR | OP_RAP_CHALLENGE | OP_ALPHA_POW | OP_TABLE_OFFSET => {}
                OP_ADD | OP_SUB | OP_MUL => {
                    check_id(n.a)?;
                    check_id(n.b)?;
                }
                OP_NEG | OP_EMBED => check_id(n.a)?,
                other => return bad(format!("node {i} has unknown op tag {other}")),
            }
        }

        if self.roots.len() != self.meta.len() {
            return bad(format!(
                "{} roots but {} metadata entries",
                self.roots.len(),
                self.meta.len()
            ));
        }
        for (c, &root) in self.roots.iter().enumerate() {
            if root as usize >= self.nodes.len() {
                return bad(format!(
                    "constraint {c} roots at node {root} of {}",
                    self.nodes.len()
                ));
            }
        }

        let num_base = self.num_base as usize;
        if num_base > self.meta.len() {
            return bad(format!(
                "num_base {num_base} exceeds the {} constraints",
                self.meta.len()
            ));
        }
        for (i, m) in self.meta.iter().enumerate() {
            if m.constraint_idx as usize != i {
                return bad(format!(
                    "metadata entry {i} claims constraint_idx {}; the list must be dense and \
                     idx-ordered",
                    m.constraint_idx
                ));
            }
            let expected = if i < num_base {
                ArtifactMeta::KIND_BASE
            } else {
                ArtifactMeta::KIND_EXT
            };
            if m.kind != expected {
                return bad(format!(
                    "constraint {i} has kind {} but num_base is {num_base}; Base entries must \
                     form a prefix of exactly that length",
                    m.kind
                ));
            }
        }

        Ok(())
    }

    /// Check that this artifact actually describes `air`.
    ///
    /// # What this proves, and what it does not
    ///
    /// It compares the SHAPE scalars and the per-constraint metadata — enough to
    /// reject an artifact captured from a different AIR, or a stale artifact
    /// from before a column was added or a constraint's exemptions changed.
    ///
    /// It does NOT prove the serialized program computes the same algebra as the
    /// AIR's compiled folder: verifying that requires evaluating both, which
    /// requires capture. An AIR edit that changes a constraint's arithmetic
    /// without changing any width or exemption passes this check. The build-time
    /// drift test is what covers that case, and it is not optional.
    pub fn validate_against<A>(&self, air: &A) -> Result<(), ArtifactError>
    where
        A: AIR<Field = Gl, FieldExtension = Ext3> + ?Sized,
    {
        self.validate_self()?;

        let (main_width, aux_width) = air.trace_layout();
        check_field("main_width", self.shape.main_width as usize, main_width)?;
        check_field("aux_width", self.shape.aux_width as usize, aux_width)?;
        check_field("step_size", self.shape.step_size as usize, air.step_size())?;
        check_field(
            "num_transition_constraints",
            self.roots.len(),
            air.context().num_transition_constraints,
        )?;
        check_field(
            "num_base",
            self.num_base as usize,
            air.num_base_transition_constraints(),
        )?;
        check_field(
            "max_bus_elements",
            self.shape.max_bus_elements as usize,
            air.max_bus_elements(),
        )?;
        check_field(
            "has_trace_interaction",
            self.shape.has_trace_interaction,
            air.has_trace_interaction(),
        )?;
        check_field(
            "is_preprocessed",
            self.shape.is_preprocessed,
            air.is_preprocessed(),
        )?;
        check_field(
            "num_precomputed_columns",
            self.shape.num_precomputed_columns as usize,
            air.num_precomputed_columns(),
        )?;

        let offsets: Vec<usize> = self
            .shape
            .transition_offsets
            .iter()
            .map(|o| *o as usize)
            .collect();
        check_field(
            "transition_offsets",
            offsets,
            air.context().transition_offsets.clone(),
        )?;

        let mut declared = air.trace_ood_next_row_columns();
        declared.sort_unstable();
        declared.dedup();
        let stored: Vec<usize> = self
            .shape
            .next_row_columns
            .iter()
            .map(|c| *c as usize)
            .collect();
        check_field("next_row_columns", stored, declared)?;

        check_field(
            "composition_degree_multiplier",
            self.shape.composition_degree_multiplier as usize,
            air.composition_poly_degree_bound(DEGREE_PROBE_LEN) / DEGREE_PROBE_LEN,
        )?;

        let air_meta: Vec<ConstraintMeta> = air.constraints_meta().to_vec();
        check_field("constraints_meta", self.constraints_meta(), air_meta)?;

        Ok(())
    }

    /// Serialize to the on-disk / in-guest byte form (rkyv, matching the
    /// proof format's own encoding).
    pub fn to_bytes(&self) -> Result<Vec<u8>, ArtifactError> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .map(|b| b.to_vec())
            .map_err(|e| ArtifactError::Codec(e.to_string()))
    }

    /// Deserialize, then check internal consistency. Guest-safe.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, ArtifactError> {
        let artifact = rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
            .map_err(|e| ArtifactError::Codec(e.to_string()))?;
        artifact.validate_self()?;
        Ok(artifact)
    }
}
