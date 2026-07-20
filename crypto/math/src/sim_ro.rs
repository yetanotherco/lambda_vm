//! Shared ABI structs for the DEEP reduced-opening measurement ecalls
//! (`sim-ro-ecalls` / `sim-ro-query`). MEASUREMENT-ONLY, never proven.
//!
//! These are the `input_ptr` payloads for the two stub ecalls that let us
//! measure the optimistic cycle ceiling of a fused reduced-opening
//! accelerator chip (see `others/accelerator_noop_sim_spec.md`, Experiment 2).
//!
//! The structs live in `math` because it is the only crate that both the
//! guest-side marshaling (`crypto/stark` verifier, riscv64) and the host-side
//! executor handler depend on — so the layout has a single source of truth.
//! The guest builds the struct on its stack and passes `&input as u64`; the
//! executor reads each field out of guest memory with `load_doubleword` at
//! `input_ptr + offset_of!(Struct, field)`.
//!
//! # Layout contract (load-bearing)
//! - Every field is a `u64`, so `#[repr(C)]` gives field `i` at byte offset
//!   `8*i` and the whole struct is 8-aligned. The executor uses
//!   `core::mem::offset_of!` against these definitions, so field ORDER here is
//!   the ABI — reordering fields silently changes what the handler reads.
//! - Pointer fields are guest virtual addresses (`slice.as_ptr() as u64`).
//! - This ABI is specialised to the recursion guest's concrete field choice:
//!   base field = Goldilocks (`FieldElement` = 1 limb = 8 bytes, canonical, no
//!   Montgomery form) and extension = degree-3 Goldilocks (`FieldElement` =
//!   `[FpE; 3]` = 3 limbs = 24 bytes). A base slice element is 8 bytes apart;
//!   an extension slice element is 24 bytes apart. The host hard-codes those
//!   strides, so this ABI is ONLY valid for that instantiation.

/// Level A — `REDUCED_OPENING_ROW` input.
///
/// Per-query payload (constant across the row loop). The ecall additionally
/// takes `row_idx` (a1) and `out_ptr` (a2, a `[FieldElement<ext>; 2]` = 6 u64
/// scratch the host fills with `(base_row_sum, base_row_sum_sym)`).
///
/// Mirrors the column loop at `crypto/stark/src/verifier.rs` inside
/// `reconstruct_deep_composition_poly_evaluation_pair` (the
/// `for row_idx { base_row_sum += ... }` body). `base_at(col)` resolves a base
/// column into the concatenation `precomputed ‖ main`; aux columns
/// (`col >= num_base`) index the aux slice at `col - num_base`. Coefficients
/// come from the `[col][row]` grid via a per-column pointer table.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReducedOpeningRowInput {
    /// `lde_trace_precomputed_evaluations` (regular point) — base, 1 limb each.
    pub precomputed_ptr: u64,
    pub precomputed_len: u64,
    /// `lde_trace_main_evaluations` (regular point) — base, 1 limb each.
    pub main_ptr: u64,
    pub main_len: u64,
    /// `lde_trace_aux_evaluations` (regular point) — extension, 3 limbs each.
    pub aux_ptr: u64,
    pub aux_len: u64,
    /// Symmetric-point counterparts.
    pub precomputed_sym_ptr: u64,
    pub precomputed_sym_len: u64,
    pub main_sym_ptr: u64,
    pub main_sym_len: u64,
    pub aux_sym_ptr: u64,
    pub aux_sym_len: u64,
    /// Pointer to an array of `ood_width` column-data pointers
    /// (`trace_term_coeffs[col].as_ptr() as u64`), i.e. `coeff[col][row]` lives
    /// at `col_ptrs[col] + row*24`.
    pub coeff_col_ptrs_ptr: u64,
    /// `next_row_cols` slice (`usize`, 8 bytes each) — the g·z transition
    /// window iterated for pruned rows (`row_idx >= step_size`).
    pub next_row_cols_ptr: u64,
    pub next_row_cols_len: u64,
    /// Total OOD table width = `num_base + aux_len` = coeff grid column count.
    pub ood_width: u64,
    /// Rows `< step_size` iterate all columns; rows `>= step_size` iterate only
    /// `next_row_cols` (g·z pruning).
    pub step_size: u64,
}

/// Number of `u64` fields in [`ReducedOpeningRowInput`].
pub const REDUCED_OPENING_ROW_INPUT_FIELDS: usize = 17;

/// Level B — `REDUCED_OPENING_QUERY` input.
///
/// One payload per query; the ecall replaces the whole
/// `reconstruct_deep_composition_poly_evaluation_pair` body and takes
/// `out_ptr` (a1, a `[FieldElement<ext>; 2]` = 6 u64 the host fills with
/// `(deep_eval, deep_eval_sym)`).
///
/// Field-element scalars are passed BY POINTER (the host reads their limbs
/// in place): the guest verifier is a generic trait method that cannot assume
/// `Field::BaseType == u64`, so it can only take addresses, not inline limbs.
/// The host reads 1 limb at a base pointer, 3 at an extension pointer.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReducedOpeningQueryInput {
    /// `&evaluation_point` — base, 1 limb.
    pub evaluation_point_ptr: u64,
    /// `&evaluation_point_sym` — base, 1 limb.
    pub evaluation_point_sym_ptr: u64,
    /// `&primitive_root` (domain root `g`) — base, 1 limb.
    pub primitive_root_ptr: u64,
    /// `&challenges.z` — extension, 3 limbs.
    pub z_ptr: u64,
    /// `&query_invariant_terms.z_pow` — extension, 3 limbs.
    pub z_pow_ptr: u64,
    /// `&query_invariant_terms.h_sum_zpow` — extension, 3 limbs.
    pub h_sum_zpow_ptr: u64,
    /// `ood_evaluations_table_height` = `ood_row_sum.len()`.
    pub ood_height: u64,
    /// `ood_evaluations_table_width`.
    pub ood_width: u64,
    /// `number_of_parts` (composition poly parts / gammas used).
    pub number_of_parts: u64,
    /// g·z pruning threshold.
    pub step_size: u64,
    /// Regular-point base/aux eval slices.
    pub precomputed_ptr: u64,
    pub precomputed_len: u64,
    pub main_ptr: u64,
    pub main_len: u64,
    pub aux_ptr: u64,
    pub aux_len: u64,
    /// Symmetric-point base/aux eval slices.
    pub precomputed_sym_ptr: u64,
    pub precomputed_sym_len: u64,
    pub main_sym_ptr: u64,
    pub main_sym_len: u64,
    pub aux_sym_ptr: u64,
    pub aux_sym_len: u64,
    /// `lde_composition_poly_parts_evaluation` (len = `number_of_parts`) —
    /// extension, 3 limbs each; and its symmetric counterpart.
    pub composition_ptr: u64,
    pub composition_sym_ptr: u64,
    /// `trace_term_coeffs` per-column pointer table (see the Level A field).
    pub coeff_col_ptrs_ptr: u64,
    /// `challenges.gammas` (extension, 3 limbs each; first `number_of_parts`).
    pub gammas_ptr: u64,
    /// `query_invariant_terms.ood_row_sum` (extension, len = `ood_height`).
    pub ood_row_sum_ptr: u64,
    /// `next_row_cols` slice.
    pub next_row_cols_ptr: u64,
    pub next_row_cols_len: u64,
}

/// Number of `u64` fields in [`ReducedOpeningQueryInput`].
pub const REDUCED_OPENING_QUERY_INPUT_FIELDS: usize = 29;

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{align_of, size_of};

    #[test]
    fn row_input_is_packed_u64_array() {
        assert_eq!(align_of::<ReducedOpeningRowInput>(), 8);
        assert_eq!(
            size_of::<ReducedOpeningRowInput>(),
            REDUCED_OPENING_ROW_INPUT_FIELDS * 8
        );
        // Spot-check that repr(C) lays the fields out consecutively at 8*i.
        assert_eq!(
            core::mem::offset_of!(ReducedOpeningRowInput, precomputed_ptr),
            0
        );
        assert_eq!(
            core::mem::offset_of!(ReducedOpeningRowInput, coeff_col_ptrs_ptr),
            12 * 8
        );
        assert_eq!(
            core::mem::offset_of!(ReducedOpeningRowInput, step_size),
            16 * 8
        );
    }

    #[test]
    fn query_input_is_packed_u64_array() {
        assert_eq!(align_of::<ReducedOpeningQueryInput>(), 8);
        assert_eq!(
            size_of::<ReducedOpeningQueryInput>(),
            REDUCED_OPENING_QUERY_INPUT_FIELDS * 8
        );
        assert_eq!(
            core::mem::offset_of!(ReducedOpeningQueryInput, evaluation_point_ptr),
            0
        );
        assert_eq!(
            core::mem::offset_of!(ReducedOpeningQueryInput, next_row_cols_len),
            28 * 8
        );
    }
}
