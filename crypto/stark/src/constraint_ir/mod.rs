//! Field-generic flat IR for transition constraints.
//!
//! A transition constraint's algebra is captured, at AIR-construction time,
//! into a flat intermediate representation ([`ConstraintProgram`]) via an
//! explicit [`IrBuilder`]. Interpreting that IR on the CPU
//! ([`eval_program`] / [`eval_program_verifier`]) reproduces the constraint's
//! real evaluation bit-for-bit, and the same IR is the input to the future GPU
//! constraint-evaluation kernel.
//!
//! The whole module is generic over a field tower `<F: IsSubFieldOf<E>, E>`
//! (defaulting to the Goldilocks base field and its degree-3 extension), so a
//! capture front-end can target it for any field. Constants live in side tables
//! keyed by index, which keeps [`Op`] a plain `Copy + Eq + Hash` payload and the
//! builder's common-subexpression cache sound for every field.
//!
//! - [`ir`]: the IR data structures ([`ConstraintProgram`], [`Op`], [`Dim`]).
//! - [`builder`]: the [`IrBuilder`] and [`Expr`] capture API.
//! - [`interp`]: a CPU forward-pass interpreter over the IR.
//! - [`device`]: the concrete-Goldilocks flat lowering ([`DeviceProgram`]) for
//!   the GPU kernel, plus a CPU walker over that flat blob (the pre-GPU parity
//!   oracle).
//!
//! [`ConstraintProgram`]: ir::ConstraintProgram
//! [`Op`]: ir::Op
//! [`Dim`]: ir::Dim
//! [`DeviceProgram`]: device::DeviceProgram

pub mod builder;
pub mod device;
#[cfg(feature = "cuda")]
pub mod gpu_interp;
pub mod interp;
pub mod ir;

#[cfg(test)]
mod tests;

pub use builder::{Expr, IrBuilder};
pub use device::{DeviceNode, DeviceProgram, eval_device_program};
pub use interp::{eval_program, eval_program_base, eval_program_verifier};
pub use ir::{ConstraintProgram, Dim, Op};
