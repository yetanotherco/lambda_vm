//! Explicit-builder constraint capture spike (Plan B).
//!
//! Proof-of-concept that lambda_vm's algebraic transition constraints can be
//! captured into a flat, single-field Goldilocks IR via an explicit
//! [`IrBuilder`] (rather than the recording "symbolic field" of Plan A), and
//! that interpreting that IR on the CPU reproduces the constraint's real
//! `evaluate` bit-for-bit.
//!
//! Both plans produce the SAME IR and use the SAME interpreter; they differ
//! only in the capture front-end. Here each constraint implements [`Capture`]
//! and translates its `evaluate` body into builder calls. This is CPU-only and
//! does not touch the prover hot loop, the LogUp framework, or GPU code.
//!
//! - [`ir`]: the IR data structures ([`ConstraintProgram`], [`Op`], [`Dim`]).
//! - [`builder`]: the [`IrBuilder`] and [`Expr`] capture API.
//! - [`interp`]: a CPU forward-pass interpreter over the IR.
//!
//! [`ConstraintProgram`]: ir::ConstraintProgram
//! [`Op`]: ir::Op
//! [`Dim`]: ir::Dim

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
