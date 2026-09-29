//! The 3MI Field VM: a single-instruction (`FMA` over the cubic extension)
//! machine for the field half of split recursion verification.
//!
//! Spec: `spec/chapters/field_vm.typ`, `spec/src/field_vm.toml` and
//! `spec/src/field_vm_decode.toml` (#971).

pub mod asm;
pub mod executor;
pub mod isa;

#[cfg(test)]
mod tests;
