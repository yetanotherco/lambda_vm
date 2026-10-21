//! The 3MI Field VM: a single-instruction (`FMA` over the cubic extension)
//! machine for the field half of split recursion verification.
//!
//! Spec: `spec/chapters/field_vm.typ`, `spec/src/field_vm.toml` and
//! `spec/src/field_vm_decode.toml` (#971).

pub mod air;
pub mod asm;
pub mod bridge;
pub mod decode;
pub mod executor;
pub mod isa;
pub mod mem;
pub mod mux;
pub mod prove;

#[cfg(test)]
pub(crate) mod hash_side;
#[cfg(test)]
mod lfm_compare;
#[cfg(test)]
pub(crate) mod lfm_translate;
#[cfg(test)]
mod prove_tests;
#[cfg(test)]
mod tests;
