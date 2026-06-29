//! Symbolic-field constraint capture spike.
//!
//! A proof-of-concept that lambda_vm's algebraic transition constraints can be
//! captured into a flat, single-field Goldilocks IR by running each
//! constraint's existing generic `evaluate::<F, E>` over recording field types
//! ([`SymField`]/[`SymExt`]), and that interpreting that IR on the CPU
//! reproduces the constraint's real `evaluate` bit-for-bit.
//!
//! This is CPU-only and does not touch the prover hot loop, the LogUp
//! framework, or GPU code. See `thoughts/gpu-constraint-eval/plan-symbolic-field.md`.
//!
//! - [`ir`]: the IR data structures ([`ConstraintProgram`], [`Op`], [`Dim`]).
//! - [`sym_field`]: the recording fields and the thread-local arena.
//! - [`capture`]: capture a constraint into a [`ConstraintProgram`].
//! - [`interp`]: a CPU forward-pass interpreter over the IR.
//!
//! [`ConstraintProgram`]: ir::ConstraintProgram
//! [`Op`]: ir::Op
//! [`Dim`]: ir::Dim
//! [`SymField`]: sym_field::SymField
//! [`SymExt`]: sym_field::SymExt

pub mod capture;
pub mod interp;
pub mod ir;
pub mod sym_field;

pub use capture::capture_constraint;
pub use interp::eval_program_base;
pub use ir::{ConstraintProgram, Dim, Op};
pub use sym_field::{SymExt, SymField, SymId};
