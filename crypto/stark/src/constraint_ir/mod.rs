//! Explicit-builder constraint capture (Plan B).
//!
//! Every transition constraint is captured once, at AIR-construction time,
//! into a flat single-field Goldilocks IR ([`ConstraintProgram`]) via an
//! explicit [`IrBuilder`] (rather than the recording "symbolic field" of Plan
//! A). Interpreting that IR on the CPU reproduces the constraint's real
//! `evaluate`/`compute` body bit-for-bit — including the LogUp framework
//! constraints (`crypto/stark/src/lookup.rs`).
//!
//! Both plans produce the SAME IR and use the SAME interpreter; they differ
//! only in the capture front-end. Here each constraint implements [`Capture`]
//! and translates its `evaluate` body into builder calls.
//!
//! Behind the `constraint-ir` Cargo feature, [`bridge`] swaps the interpreter
//! into the prover (`constraints/evaluator.rs`) and verifier (`verifier.rs`)
//! hot paths, in place of the boxed `Vec<Box<dyn TransitionConstraintEvaluator>>`
//! dispatch loop. The boxed path stays the default and the differential oracle.
//!
//! - [`ir`]: the IR data structures ([`ConstraintProgram`], [`Op`], [`Dim`]).
//! - [`builder`]: the [`IrBuilder`] and [`Expr`] capture API.
//! - [`interp`]: a CPU forward-pass interpreter over the IR.
//! - [`bridge`]: the generic-`Field`/`FieldExtension` → concrete-Goldilocks
//!   TypeId seam used to call the interpreter from the generic prover/verifier.
//!
//! [`ConstraintProgram`]: ir::ConstraintProgram
//! [`Op`]: ir::Op
//! [`Dim`]: ir::Dim

pub mod bridge;
pub mod builder;
pub mod interp;
pub mod ir;

pub use builder::{Expr, IrBuilder};
pub use interp::{eval_program, eval_program_base, eval_program_verifier};
pub use ir::{ConstraintProgram, Dim, Op};

/// A transition constraint that can record its algebra into an [`IrBuilder`].
///
/// Object-safe: `capture` is non-generic (it takes `&mut IrBuilder`), so a
/// constraint can be captured behind a `&dyn Capture`, mirroring the production
/// design where the capture method is not generic over the field tower.
pub trait Capture {
    /// Translate this constraint's algebra into builder nodes, finishing with a
    /// single `b.emit(constraint_idx, root)` call.
    fn capture(&self, b: &mut IrBuilder);
}
