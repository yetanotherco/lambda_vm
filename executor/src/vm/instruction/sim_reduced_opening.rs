//! Host-side handlers for the DEEP reduced-opening measurement ecalls.
//!
//! MEASUREMENT-ONLY. These stubs compute the CORRECT value of the recursion
//! verifier's DEEP reduced-opening hot loop host-side, in a single VM cycle,
//! and write it back to guest memory. Replacing the in-guest column loop with
//! a 1-cycle ecall lets us measure the optimistic cycle ceiling of a fused
//! reduced-opening accelerator chip (see `others/accelerator_noop_sim_spec.md`,
//! Experiment 2). They are TRUSTED passthroughs: the returned value is exact,
//! so the guest still accepts the proof. NEVER PROVE a build that calls them —
//! they have no chip table and would unbalance the Ecall LogUp bus.
//!
//! The computation mirrors, operation-for-operation, the loop in
//! `crypto/stark/src/verifier.rs`
//! (`reconstruct_deep_composition_poly_evaluation_pair`). Goldilocks
//! arithmetic does not always reduce to the canonical `[0, p)` representative
//! (add/sub can leave a value in `[p, 2^64)`), so field-value equality does not
//! imply bit equality. To keep the bytes written back byte-identical to what
//! the guest's own loop would produce, this handler reads the exact limb
//! representatives from guest memory and applies the SAME operators in the SAME
//! order as the verifier, via the SAME `math` field types.
//!
//! ABI: see [`math::sim_ro`]. This handler is specialised to the recursion
//! guest's concrete field choice (Goldilocks base = 1 limb, degree-3 extension
//! = 3 limbs); the strides below hard-code that.

use crate::vm::instruction::execution::ExecutionError;
use crate::vm::memory::{Memory, MemoryError};
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::sim_ro::{ReducedOpeningLayout, ReducedOpeningQueryInput, ReducedOpeningRowInput};

/// Syscall number for `REDUCED_OPENING_ROW` (Level A) — `u64::MAX - 30`.
/// Renumbered from MAX-20 to clear the reserved FEXT accelerator range
/// (MAX-19..MAX-21, LOAD/FMA/STORE) with a buffer after merging PR #818/#831;
/// must stay in lockstep with the guest wrapper const in `syscalls/src/syscalls.rs`.
pub const REDUCED_OPENING_ROW_SYSCALL_NUMBER: u64 = u64::MAX - 30;
/// Syscall number for `REDUCED_OPENING_QUERY` (Level B) — `u64::MAX - 31`.
/// Renumbered from MAX-21 to clear the FEXT accelerator range.
pub const REDUCED_OPENING_QUERY_SYSCALL_NUMBER: u64 = u64::MAX - 31;
/// Syscall number for `REGISTER_RO_LAYOUT` (ROUND-2 increment C) — `u64::MAX - 53`.
pub const REGISTER_RO_LAYOUT_SYSCALL_NUMBER: u64 = u64::MAX - 53;
/// Syscall number for `REDUCED_OPENING_ROW_INPLACE` (increment C) — `u64::MAX - 54`.
pub const REDUCED_OPENING_ROW_INPLACE_SYSCALL_NUMBER: u64 = u64::MAX - 54;

type F = FieldElement<GoldilocksField>;
type E = FieldElement<Degree3GoldilocksExtensionField>;

/// Bytes per base-field limb-group (Goldilocks `FieldElement` = 1 `u64`).
const BASE_STRIDE: u64 = 8;
/// Bytes per extension `FieldElement` (`[FpE; 3]` = 3 `u64`).
const EXT_STRIDE: u64 = 24;

/// Read a single `u64` field of an ABI struct at `base + offset`.
#[inline]
fn field(memory: &Memory, base: u64, offset: usize) -> Result<u64, MemoryError> {
    memory.load_doubleword(base.wrapping_add(offset as u64))
}

/// Read a base-field element (1 limb) at `addr`.
#[inline]
fn read_base(memory: &Memory, addr: u64) -> Result<F, MemoryError> {
    Ok(F::from_raw(memory.load_doubleword(addr)?))
}

/// Read an extension element (3 little-endian limbs) at `addr`.
#[inline]
fn read_ext(memory: &Memory, addr: u64) -> Result<E, MemoryError> {
    let l0 = memory.load_doubleword(addr)?;
    let l1 = memory.load_doubleword(addr.wrapping_add(8))?;
    let l2 = memory.load_doubleword(addr.wrapping_add(16))?;
    Ok(E::from_raw([
        F::from_raw(l0),
        F::from_raw(l1),
        F::from_raw(l2),
    ]))
}

/// Write an extension element (3 little-endian limbs) at `addr`.
#[inline]
fn write_ext(memory: &mut Memory, addr: u64, value: &E) -> Result<(), MemoryError> {
    let limbs = value.value();
    memory.store_doubleword(addr, *limbs[0].value())?;
    memory.store_doubleword(addr.wrapping_add(8), *limbs[1].value())?;
    memory.store_doubleword(addr.wrapping_add(16), *limbs[2].value())?;
    Ok(())
}

/// The per-query slice pointers + dims shared by both levels' row-sum loop.
/// All fields are guest addresses / counts pulled straight out of an ABI
/// struct.
struct RowInputs {
    precomputed_ptr: u64,
    precomputed_len: u64,
    main_ptr: u64,
    main_len: u64,
    aux_ptr: u64,
    aux_len: u64,
    precomputed_sym_ptr: u64,
    precomputed_sym_len: u64,
    main_sym_ptr: u64,
    main_sym_len: u64,
    aux_sym_ptr: u64,
    aux_sym_len: u64,
    coeff_col_ptrs_ptr: u64,
    next_row_cols_ptr: u64,
    next_row_cols_len: u64,
    ood_width: u64,
    step_size: u64,
}

impl RowInputs {
    fn num_precomputed(&self) -> u64 {
        self.precomputed_len
    }
    fn num_base(&self) -> u64 {
        self.precomputed_len + self.main_len
    }
    fn num_precomputed_sym(&self) -> u64 {
        self.precomputed_sym_len
    }
    fn num_base_sym(&self) -> u64 {
        self.precomputed_sym_len + self.main_sym_len
    }

    /// Mirrors the verifier's runtime guards (`num_base == num_base_sym`, the
    /// base+aux split matching the OOD width for both points). Real proofs
    /// always satisfy these before the ecall fires; a violation means garbage
    /// input, so reject rather than read out of bounds / measure nonsense.
    fn validate(&self) -> Result<(), ExecutionError> {
        if self.num_base() != self.num_base_sym()
            || self.num_base() + self.aux_len != self.ood_width
            || self.num_base_sym() + self.aux_sym_len != self.ood_width
        {
            return Err(ExecutionError::SimReducedOpeningInvalidDims);
        }
        Ok(())
    }

    /// `base_at(col)` — the `precomputed ‖ main` concatenation at the regular
    /// point.
    fn base_at(&self, memory: &Memory, col: u64) -> Result<F, MemoryError> {
        let np = self.num_precomputed();
        let addr = if col < np {
            self.precomputed_ptr
                .wrapping_add(col.wrapping_mul(BASE_STRIDE))
        } else {
            self.main_ptr
                .wrapping_add((col - np).wrapping_mul(BASE_STRIDE))
        };
        read_base(memory, addr)
    }

    /// `base_at_sym(col)` — the same at the symmetric point (its own split).
    fn base_at_sym(&self, memory: &Memory, col: u64) -> Result<F, MemoryError> {
        let np = self.num_precomputed_sym();
        let addr = if col < np {
            self.precomputed_sym_ptr
                .wrapping_add(col.wrapping_mul(BASE_STRIDE))
        } else {
            self.main_sym_ptr
                .wrapping_add((col - np).wrapping_mul(BASE_STRIDE))
        };
        read_base(memory, addr)
    }

    /// `trace_term_coeffs[col][row]` via the per-column pointer table.
    fn coeff(&self, memory: &Memory, col: u64, row: u64) -> Result<E, MemoryError> {
        let col_data =
            memory.load_doubleword(self.coeff_col_ptrs_ptr.wrapping_add(col.wrapping_mul(8)))?;
        read_ext(memory, col_data.wrapping_add(row.wrapping_mul(EXT_STRIDE)))
    }

    fn aux(&self, memory: &Memory, aux_idx: u64) -> Result<E, MemoryError> {
        read_ext(
            memory,
            self.aux_ptr.wrapping_add(aux_idx.wrapping_mul(EXT_STRIDE)),
        )
    }
    fn aux_sym(&self, memory: &Memory, aux_idx: u64) -> Result<E, MemoryError> {
        read_ext(
            memory,
            self.aux_sym_ptr
                .wrapping_add(aux_idx.wrapping_mul(EXT_STRIDE)),
        )
    }

    /// Accumulate one column into `(base_row_sum, base_row_sum_sym)`, matching
    /// the verifier's operand order exactly: base columns use the asymmetric
    /// `base * coeff` (`F * E`), aux columns use `coeff * aux` (`E * E`).
    fn accumulate(
        &self,
        memory: &Memory,
        base_row_sum: &mut E,
        base_row_sum_sym: &mut E,
        col: u64,
        row: u64,
        num_base: u64,
    ) -> Result<(), MemoryError> {
        let coeff = self.coeff(memory, col, row)?;
        if col < num_base {
            let base_val = self.base_at(memory, col)?;
            let base_val_sym = self.base_at_sym(memory, col)?;
            *base_row_sum += &base_val * &coeff;
            *base_row_sum_sym += &base_val_sym * &coeff;
        } else {
            let aux_idx = col - num_base;
            let aux_val = self.aux(memory, aux_idx)?;
            let aux_val_sym = self.aux_sym(memory, aux_idx)?;
            *base_row_sum += &coeff * &aux_val;
            *base_row_sum_sym += &coeff * &aux_val_sym;
        }
        Ok(())
    }

    /// Compute `(base_row_sum, base_row_sum_sym)` for one OOD row, honouring
    /// g·z pruning: rows `< step_size` sum all columns; later rows sum only the
    /// `next_row_cols` transition window.
    fn row_sums(&self, memory: &Memory, row_idx: u64) -> Result<(E, E), MemoryError> {
        let num_base = self.num_base();
        let mut base_row_sum = E::zero();
        let mut base_row_sum_sym = E::zero();
        if row_idx < self.step_size {
            for col in 0..self.ood_width {
                self.accumulate(
                    memory,
                    &mut base_row_sum,
                    &mut base_row_sum_sym,
                    col,
                    row_idx,
                    num_base,
                )?;
            }
        } else {
            for i in 0..self.next_row_cols_len {
                let col = memory
                    .load_doubleword(self.next_row_cols_ptr.wrapping_add(i.wrapping_mul(8)))?;
                self.accumulate(
                    memory,
                    &mut base_row_sum,
                    &mut base_row_sum_sym,
                    col,
                    row_idx,
                    num_base,
                )?;
            }
        }
        Ok((base_row_sum, base_row_sum_sym))
    }
}

/// Read the Level A input struct out of guest memory.
fn read_row_input(memory: &Memory, input_ptr: u64) -> Result<RowInputs, MemoryError> {
    use core::mem::offset_of;
    macro_rules! f {
        ($field:ident) => {
            field(
                memory,
                input_ptr,
                offset_of!(ReducedOpeningRowInput, $field),
            )?
        };
    }
    Ok(RowInputs {
        precomputed_ptr: f!(precomputed_ptr),
        precomputed_len: f!(precomputed_len),
        main_ptr: f!(main_ptr),
        main_len: f!(main_len),
        aux_ptr: f!(aux_ptr),
        aux_len: f!(aux_len),
        precomputed_sym_ptr: f!(precomputed_sym_ptr),
        precomputed_sym_len: f!(precomputed_sym_len),
        main_sym_ptr: f!(main_sym_ptr),
        main_sym_len: f!(main_sym_len),
        aux_sym_ptr: f!(aux_sym_ptr),
        aux_sym_len: f!(aux_sym_len),
        coeff_col_ptrs_ptr: f!(coeff_col_ptrs_ptr),
        next_row_cols_ptr: f!(next_row_cols_ptr),
        next_row_cols_len: f!(next_row_cols_len),
        ood_width: f!(ood_width),
        step_size: f!(step_size),
    })
}

/// Level A — `REDUCED_OPENING_ROW`. `a0 = &input`, `a1 = row_idx`,
/// `a2 = out_ptr` (a `[FieldElement<ext>; 2]` scratch = 6 u64). Writes
/// `(base_row_sum, base_row_sum_sym)` for `row_idx`.
pub fn reduced_opening_row(
    memory: &mut Memory,
    input_ptr: u64,
    row_idx: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    let inputs = read_row_input(memory, input_ptr)?;
    inputs.validate()?;
    let (base_row_sum, base_row_sum_sym) = inputs.row_sums(memory, row_idx)?;
    write_ext(memory, out_ptr, &base_row_sum)?;
    write_ext(memory, out_ptr.wrapping_add(EXT_STRIDE), &base_row_sum_sym)?;
    Ok(())
}

// ROUND-2 increment C — in-place reduced-opening ABI. The PROOF-CONSTANT layout
// (coeff col-ptr table, transition window, OOD dims, per-slice column counts) is
// registered ONCE per proof into this thread-local; each row ecall then supplies
// only `{row_idx, evals_ptr, out_ptr}`, where `evals_ptr` gives the six per-query
// eval-slice base pointers. This kills the per-query struct fill + col-ptr gather
// the Level A ABI repeats for every query, measuring the ceiling of a chip that
// reads its operands in place. `cli execute` runs single-threaded, one proof per
// process; REGISTER overwrites, so no stale-state hazard.
thread_local! {
    static RO_LAYOUT: core::cell::RefCell<Option<ReducedOpeningLayout>> =
        const { core::cell::RefCell::new(None) };
}

/// Read a `ReducedOpeningLayout` from guest memory and cache it in the
/// thread-local for the subsequent in-place row ecalls (`REGISTER_RO_LAYOUT`,
/// increment C). `a0 = &layout`.
pub fn register_ro_layout(memory: &Memory, layout_ptr: u64) -> Result<(), ExecutionError> {
    use core::mem::offset_of;
    macro_rules! f {
        ($field:ident) => {
            field(memory, layout_ptr, offset_of!(ReducedOpeningLayout, $field))?
        };
    }
    let layout = ReducedOpeningLayout {
        coeff_col_ptrs_ptr: f!(coeff_col_ptrs_ptr),
        next_row_cols_ptr: f!(next_row_cols_ptr),
        next_row_cols_len: f!(next_row_cols_len),
        ood_width: f!(ood_width),
        step_size: f!(step_size),
        precomputed_len: f!(precomputed_len),
        main_len: f!(main_len),
        aux_len: f!(aux_len),
        precomputed_sym_len: f!(precomputed_sym_len),
        main_sym_len: f!(main_sym_len),
        aux_sym_len: f!(aux_sym_len),
    };
    RO_LAYOUT.with(|cell| *cell.borrow_mut() = Some(layout));
    Ok(())
}

/// `REDUCED_OPENING_ROW_INPLACE` (increment C). `a0 = row_idx`, `a1 = evals_ptr`
/// (six per-query eval-slice base pointers: precomputed, main, aux, then their
/// symmetric counterparts), `a2 = out_ptr` (`[FieldElement<ext>; 2]` = 6 u64).
/// Rebuilds the Level A `RowInputs` from the registered proof-constant layout +
/// the six per-query base pointers, then reuses the exact same `row_sums`
/// computation — so the written `(base_row_sum, base_row_sum_sym)` is
/// byte-identical to `reduced_opening_row`.
pub fn reduced_opening_row_inplace(
    memory: &mut Memory,
    row_idx: u64,
    evals_ptr: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    let layout = RO_LAYOUT
        .with(|cell| *cell.borrow())
        .ok_or(ExecutionError::SimReducedOpeningNoLayout)?;
    // Six per-query eval-slice base pointers, in the ABI order documented on
    // `math::sim_ro::REDUCED_OPENING_INPLACE_EVALS`.
    let precomputed_ptr = memory.load_doubleword(evals_ptr)?;
    let main_ptr = memory.load_doubleword(evals_ptr.wrapping_add(8))?;
    let aux_ptr = memory.load_doubleword(evals_ptr.wrapping_add(16))?;
    let precomputed_sym_ptr = memory.load_doubleword(evals_ptr.wrapping_add(24))?;
    let main_sym_ptr = memory.load_doubleword(evals_ptr.wrapping_add(32))?;
    let aux_sym_ptr = memory.load_doubleword(evals_ptr.wrapping_add(40))?;

    let inputs = RowInputs {
        precomputed_ptr,
        precomputed_len: layout.precomputed_len,
        main_ptr,
        main_len: layout.main_len,
        aux_ptr,
        aux_len: layout.aux_len,
        precomputed_sym_ptr,
        precomputed_sym_len: layout.precomputed_sym_len,
        main_sym_ptr,
        main_sym_len: layout.main_sym_len,
        aux_sym_ptr,
        aux_sym_len: layout.aux_sym_len,
        coeff_col_ptrs_ptr: layout.coeff_col_ptrs_ptr,
        next_row_cols_ptr: layout.next_row_cols_ptr,
        next_row_cols_len: layout.next_row_cols_len,
        ood_width: layout.ood_width,
        step_size: layout.step_size,
    };
    inputs.validate()?;
    let (base_row_sum, base_row_sum_sym) = inputs.row_sums(memory, row_idx)?;
    write_ext(memory, out_ptr, &base_row_sum)?;
    write_ext(memory, out_ptr.wrapping_add(EXT_STRIDE), &base_row_sum_sym)?;
    Ok(())
}

/// Level B — `REDUCED_OPENING_QUERY`. `a0 = &input`, `a1 = out_ptr` (a
/// `[FieldElement<ext>; 2]` scratch = 6 u64). Reconstructs the whole
/// `(deep_eval, deep_eval_sym)` pair host-side, mirroring
/// `reconstruct_deep_composition_poly_evaluation_pair` verbatim.
pub fn reduced_opening_query(
    memory: &mut Memory,
    input_ptr: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    use core::mem::offset_of;
    macro_rules! q {
        ($field:ident) => {
            field(
                memory,
                input_ptr,
                offset_of!(ReducedOpeningQueryInput, $field),
            )?
        };
    }

    // Scalars are passed by pointer (generic guest can't inline limbs).
    let evaluation_point = read_base(memory, q!(evaluation_point_ptr))?;
    let evaluation_point_sym = read_base(memory, q!(evaluation_point_sym_ptr))?;
    let primitive_root = read_base(memory, q!(primitive_root_ptr))?;
    let z = read_ext(memory, q!(z_ptr))?;
    let z_pow = read_ext(memory, q!(z_pow_ptr))?;
    let h_sum_zpow = read_ext(memory, q!(h_sum_zpow_ptr))?;
    let ood_height = q!(ood_height);
    let number_of_parts = q!(number_of_parts);
    let composition_ptr = q!(composition_ptr);
    let composition_sym_ptr = q!(composition_sym_ptr);
    let gammas_ptr = q!(gammas_ptr);
    let ood_row_sum_ptr = q!(ood_row_sum_ptr);

    let row = RowInputs {
        precomputed_ptr: q!(precomputed_ptr),
        precomputed_len: q!(precomputed_len),
        main_ptr: q!(main_ptr),
        main_len: q!(main_len),
        aux_ptr: q!(aux_ptr),
        aux_len: q!(aux_len),
        precomputed_sym_ptr: q!(precomputed_sym_ptr),
        precomputed_sym_len: q!(precomputed_sym_len),
        main_sym_ptr: q!(main_sym_ptr),
        main_sym_len: q!(main_sym_len),
        aux_sym_ptr: q!(aux_sym_ptr),
        aux_sym_len: q!(aux_sym_len),
        coeff_col_ptrs_ptr: q!(coeff_col_ptrs_ptr),
        next_row_cols_ptr: q!(next_row_cols_ptr),
        next_row_cols_len: q!(next_row_cols_len),
        ood_width: q!(ood_width),
        step_size: q!(step_size),
    };
    row.validate()?;

    // Build both denominator sets (regular, then symmetric) and invert them
    // together in a single batch — verifier.rs lines 983-996.
    let ood_height_usize = ood_height as usize;
    let mut denoms: Vec<E> = Vec::with_capacity(2 * ood_height_usize);
    let mut current_z = z;
    for _ in 0..ood_height {
        denoms.push(&evaluation_point - &current_z);
        current_z = &primitive_root * &current_z;
    }
    let mut current_z = z;
    for _ in 0..ood_height {
        denoms.push(&evaluation_point_sym - &current_z);
        current_z = &primitive_root * &current_z;
    }
    FieldElement::inplace_batch_inverse(&mut denoms)
        .map_err(|_| ExecutionError::SimReducedOpeningInverse)?;
    let (denoms_trace, denoms_trace_sym) = denoms.split_at(ood_height_usize);

    let mut trace_term = E::zero();
    let mut trace_term_sym = E::zero();
    for row_idx in 0..ood_height {
        let ood_row_sum = read_ext(
            memory,
            ood_row_sum_ptr.wrapping_add(row_idx.wrapping_mul(EXT_STRIDE)),
        )?;
        let (base_row_sum, base_row_sum_sym) = row.row_sums(memory, row_idx)?;
        let i = row_idx as usize;
        trace_term += &denoms_trace[i] * &(&base_row_sum - &ood_row_sum);
        trace_term_sym += &denoms_trace_sym[i] * &(&base_row_sum_sym - &ood_row_sum);
    }

    // Composition-part terms — verifier.rs lines 1050-1064.
    let mut denom_composition_pair = [&evaluation_point - &z_pow, &evaluation_point_sym - &z_pow];
    FieldElement::inplace_batch_inverse(&mut denom_composition_pair)
        .map_err(|_| ExecutionError::SimReducedOpeningInverse)?;
    let [denom_composition, denom_composition_sym] = denom_composition_pair;

    let mut h_sum = E::zero();
    let mut h_sum_sym = E::zero();
    for j in 0..number_of_parts {
        let h_i_upsilon = read_ext(
            memory,
            composition_ptr.wrapping_add(j.wrapping_mul(EXT_STRIDE)),
        )?;
        let h_i_upsilon_sym = read_ext(
            memory,
            composition_sym_ptr.wrapping_add(j.wrapping_mul(EXT_STRIDE)),
        )?;
        let gamma = read_ext(memory, gammas_ptr.wrapping_add(j.wrapping_mul(EXT_STRIDE)))?;
        h_sum += &h_i_upsilon * &gamma;
        h_sum_sym += &h_i_upsilon_sym * &gamma;
    }
    let h_terms = (&h_sum - &h_sum_zpow) * denom_composition;
    let h_terms_sym = (&h_sum_sym - &h_sum_zpow) * denom_composition_sym;

    let deep_eval = trace_term + h_terms;
    let deep_eval_sym = trace_term_sym + h_terms_sym;
    write_ext(memory, out_ptr, &deep_eval)?;
    write_ext(memory, out_ptr.wrapping_add(EXT_STRIDE), &deep_eval_sym)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    //! The handler reads raw limbs from a mock guest `Memory` and computes with
    //! `math` field ops. Each test lays a random scenario into memory, runs the
    //! handler, and compares its bytes against an INDEPENDENT reference
    //! reimplementation of the corresponding `crypto/stark/src/verifier.rs`
    //! loop on the same `FieldElement` values. Agreement on the raw limbs
    //! (not just field value) proves the marshaling — limb order, base/aux
    //! strides, the `precomputed ‖ main` concat, the `[col][row]` coeff grid,
    //! and g·z pruning — is faithful. Values are built with `from_raw` over a
    //! PRNG so non-canonical Goldilocks representatives are exercised too.

    use super::*;

    /// Deterministic PRNG (SplitMix64-ish) so scenarios are reproducible.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // mix
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^ (z >> 31)
        }
        fn base(&mut self) -> F {
            F::from_raw(self.next())
        }
        fn ext(&mut self) -> E {
            E::from_raw([
                F::from_raw(self.next()),
                F::from_raw(self.next()),
                F::from_raw(self.next()),
            ])
        }
    }

    fn put_base(mem: &mut Memory, addr: u64, vals: &[F]) {
        for (i, v) in vals.iter().enumerate() {
            mem.store_doubleword(addr + i as u64 * BASE_STRIDE, *v.value())
                .unwrap();
        }
    }
    fn put_ext(mem: &mut Memory, addr: u64, vals: &[E]) {
        for (i, v) in vals.iter().enumerate() {
            for (j, limb) in v.value().iter().enumerate() {
                mem.store_doubleword(addr + i as u64 * EXT_STRIDE + j as u64 * 8, *limb.value())
                    .unwrap();
            }
        }
    }
    fn get_ext(mem: &Memory, addr: u64) -> E {
        read_ext(mem, addr).unwrap()
    }

    /// A generated scenario, holding both the `FieldElement` data (for the
    /// reference) and its serialized guest addresses (for the handler).
    struct Scenario {
        mem: Memory,
        // field data
        precomputed: Vec<F>,
        main: Vec<F>,
        aux: Vec<E>,
        precomputed_sym: Vec<F>,
        main_sym: Vec<F>,
        aux_sym: Vec<E>,
        coeffs: Vec<Vec<E>>, // [col][row]
        next_row_cols: Vec<usize>,
        num_precomputed: usize,
        num_base: usize,
        ood_width: usize,
        ood_height: usize,
        step_size: usize,
        // addresses
        precomputed_addr: u64,
        main_addr: u64,
        aux_addr: u64,
        precomputed_sym_addr: u64,
        main_sym_addr: u64,
        aux_sym_addr: u64,
        coeff_col_ptrs_addr: u64,
        next_row_cols_addr: u64,
    }

    /// `num_precomputed` precomputed + `num_main` main base columns, `num_aux`
    /// aux columns, `ood_height` rows. `next_row_cols` is the pruning window.
    fn build_scenario(
        seed: u64,
        num_precomputed: usize,
        num_main: usize,
        num_aux: usize,
        ood_height: usize,
        step_size: usize,
        next_row_cols: Vec<usize>,
    ) -> Scenario {
        let mut rng = Rng(seed);
        let num_base = num_precomputed + num_main;
        let ood_width = num_base + num_aux;

        let precomputed: Vec<F> = (0..num_precomputed).map(|_| rng.base()).collect();
        let main: Vec<F> = (0..num_main).map(|_| rng.base()).collect();
        let aux: Vec<E> = (0..num_aux).map(|_| rng.ext()).collect();
        let precomputed_sym: Vec<F> = (0..num_precomputed).map(|_| rng.base()).collect();
        let main_sym: Vec<F> = (0..num_main).map(|_| rng.base()).collect();
        let aux_sym: Vec<E> = (0..num_aux).map(|_| rng.ext()).collect();
        let coeffs: Vec<Vec<E>> = (0..ood_width)
            .map(|_| (0..ood_height).map(|_| rng.ext()).collect())
            .collect();

        // Lay everything out in guest memory at bumped, 8-aligned addresses.
        let mut mem = Memory::default();
        let mut next_addr = 0x10_000u64;
        let mut alloc = |n: u64| {
            let a = next_addr;
            next_addr += n.next_multiple_of(8);
            a
        };
        let precomputed_addr = alloc(precomputed.len() as u64 * BASE_STRIDE + 8);
        put_base(&mut mem, precomputed_addr, &precomputed);
        let main_addr = alloc(main.len() as u64 * BASE_STRIDE + 8);
        put_base(&mut mem, main_addr, &main);
        let aux_addr = alloc(aux.len() as u64 * EXT_STRIDE + 8);
        put_ext(&mut mem, aux_addr, &aux);
        let precomputed_sym_addr = alloc(precomputed_sym.len() as u64 * BASE_STRIDE + 8);
        put_base(&mut mem, precomputed_sym_addr, &precomputed_sym);
        let main_sym_addr = alloc(main_sym.len() as u64 * BASE_STRIDE + 8);
        put_base(&mut mem, main_sym_addr, &main_sym);
        let aux_sym_addr = alloc(aux_sym.len() as u64 * EXT_STRIDE + 8);
        put_ext(&mut mem, aux_sym_addr, &aux_sym);

        // Coeff grid: each column contiguous; a pointer table holds its base.
        let coeff_col_ptrs_addr = alloc(ood_width as u64 * 8 + 8);
        for (col, col_rows) in coeffs.iter().enumerate() {
            let col_addr = alloc(col_rows.len() as u64 * EXT_STRIDE + 8);
            put_ext(&mut mem, col_addr, col_rows);
            mem.store_doubleword(coeff_col_ptrs_addr + col as u64 * 8, col_addr)
                .unwrap();
        }

        let next_row_cols_addr = alloc(next_row_cols.len() as u64 * 8 + 8);
        for (i, &c) in next_row_cols.iter().enumerate() {
            mem.store_doubleword(next_row_cols_addr + i as u64 * 8, c as u64)
                .unwrap();
        }

        Scenario {
            mem,
            precomputed,
            main,
            aux,
            precomputed_sym,
            main_sym,
            aux_sym,
            coeffs,
            next_row_cols,
            num_precomputed,
            num_base,
            ood_width,
            ood_height,
            step_size,
            precomputed_addr,
            main_addr,
            aux_addr,
            precomputed_sym_addr,
            main_sym_addr,
            aux_sym_addr,
            coeff_col_ptrs_addr,
            next_row_cols_addr,
        }
    }

    impl Scenario {
        fn row_input(&self, input_addr: u64) -> Memory {
            // Write a ReducedOpeningRowInput at input_addr into a COPY of mem.
            let mut mem = self.mem.clone();
            use core::mem::offset_of;
            let w = |m: &mut Memory, off: usize, v: u64| {
                m.store_doubleword(input_addr + off as u64, v).unwrap()
            };
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, precomputed_ptr),
                self.precomputed_addr,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, precomputed_len),
                self.precomputed.len() as u64,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, main_ptr),
                self.main_addr,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, main_len),
                self.main.len() as u64,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, aux_ptr),
                self.aux_addr,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, aux_len),
                self.aux.len() as u64,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, precomputed_sym_ptr),
                self.precomputed_sym_addr,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, precomputed_sym_len),
                self.precomputed_sym.len() as u64,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, main_sym_ptr),
                self.main_sym_addr,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, main_sym_len),
                self.main_sym.len() as u64,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, aux_sym_ptr),
                self.aux_sym_addr,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, aux_sym_len),
                self.aux_sym.len() as u64,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, coeff_col_ptrs_ptr),
                self.coeff_col_ptrs_addr,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, next_row_cols_ptr),
                self.next_row_cols_addr,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, next_row_cols_len),
                self.next_row_cols.len() as u64,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, ood_width),
                self.ood_width as u64,
            );
            w(
                &mut mem,
                offset_of!(ReducedOpeningRowInput, step_size),
                self.step_size as u64,
            );
            mem
        }

        /// Independent reimplementation of the verifier's per-row column loop.
        /// Same iteration and accumulation order (`0..ood_width`, else the
        /// `next_row_cols` window) so raw limbs match, not just field value.
        fn ref_row_sums(&self, row_idx: usize) -> (E, E) {
            let cols: Vec<usize> = if row_idx < self.step_size {
                (0..self.ood_width).collect()
            } else {
                self.next_row_cols.clone()
            };
            let np = self.num_precomputed;
            let nps = self.precomputed_sym.len();
            let mut base_row_sum = E::zero();
            let mut base_row_sum_sym = E::zero();
            for col in cols {
                let coeff = &self.coeffs[col][row_idx];
                if col < self.num_base {
                    let bv = if col < np {
                        &self.precomputed[col]
                    } else {
                        &self.main[col - np]
                    };
                    let bvs = if col < nps {
                        &self.precomputed_sym[col]
                    } else {
                        &self.main_sym[col - nps]
                    };
                    base_row_sum += bv * coeff;
                    base_row_sum_sym += bvs * coeff;
                } else {
                    let a = col - self.num_base;
                    base_row_sum += coeff * &self.aux[a];
                    base_row_sum_sym += coeff * &self.aux_sym[a];
                }
            }
            (base_row_sum, base_row_sum_sym)
        }
    }

    #[test]
    fn level_a_row_matches_reference() {
        // num_precomputed=2, main=3 (num_base=5), aux=4 (ood_width=9), 6 rows,
        // step_size=4. Rows 0..4 iterate all columns; rows 4,5 prune to the
        // window [0,2,5,7] (base 0,2 + aux 5,7).
        let scn = build_scenario(0xC0FFEE, 2, 3, 4, 6, 4, vec![0, 2, 5, 7]);
        let input_addr = 0x9_000u64;
        let out_ptr = 0x8_000u64;

        for row_idx in 0..scn.ood_height {
            let mut mem = scn.row_input(input_addr);
            reduced_opening_row(&mut mem, input_addr, row_idx as u64, out_ptr).unwrap();
            let got = (get_ext(&mem, out_ptr), get_ext(&mem, out_ptr + EXT_STRIDE));
            let want = scn.ref_row_sums(row_idx);
            // Compare RAW limbs, not just field value: byte-identity is the
            // contract (the guest writes these bytes back into its own buffer).
            assert_eq!(got.0.value(), want.0.value(), "base_row_sum row {row_idx}");
            assert_eq!(
                got.1.value(),
                want.1.value(),
                "base_row_sum_sym row {row_idx}"
            );
        }
    }

    #[test]
    fn level_a_pruned_only_window_no_precomputed() {
        // Edge: no precomputed columns (num_precomputed=0), all-aux pruned row.
        let scn = build_scenario(0x1234, 0, 4, 3, 5, 2, vec![1, 4, 6]);
        let input_addr = 0x9_000u64;
        let out_ptr = 0x8_000u64;
        // Row 3 is pruned (>= step_size=2): window [1 (base), 4,6 (aux)].
        let mut mem = scn.row_input(input_addr);
        reduced_opening_row(&mut mem, input_addr, 3, out_ptr).unwrap();
        let got = (get_ext(&mem, out_ptr), get_ext(&mem, out_ptr + EXT_STRIDE));
        let want = scn.ref_row_sums(3);
        assert_eq!(got.0.value(), want.0.value());
        assert_eq!(got.1.value(), want.1.value());
    }

    // --- ROUND-2 increment C: in-place ABI ----------------------------------

    /// The in-place ABI (REGISTER_RO_LAYOUT once + per-row
    /// REDUCED_OPENING_ROW_INPLACE with only the six eval-slice base pointers)
    /// produces byte-identical row sums to Level A's fat-struct
    /// `reduced_opening_row` — every row, including g·z-pruned rows.
    #[test]
    fn row_inplace_matches_level_a() {
        for (seed, np, main, aux, height, step, window) in [
            (
                0xC0FFEEu64,
                2usize,
                3usize,
                4usize,
                6usize,
                4usize,
                vec![0, 2, 5, 7],
            ),
            (0x1234, 0, 4, 3, 5, 2, vec![1, 4, 6]),
        ] {
            let scn = build_scenario(seed, np, main, aux, height, step, window);
            use core::mem::offset_of;
            let (layout_addr, evals_addr, out_ptr) = (0x6_000u64, 0x7_000u64, 0x8_000u64);

            // Register the proof-constant layout.
            let mut mem = scn.mem.clone();
            let wl = |m: &mut Memory, off: usize, v: u64| {
                m.store_doubleword(layout_addr + off as u64, v).unwrap()
            };
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, coeff_col_ptrs_ptr),
                scn.coeff_col_ptrs_addr,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, next_row_cols_ptr),
                scn.next_row_cols_addr,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, next_row_cols_len),
                scn.next_row_cols.len() as u64,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, ood_width),
                scn.ood_width as u64,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, step_size),
                scn.step_size as u64,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, precomputed_len),
                scn.precomputed.len() as u64,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, main_len),
                scn.main.len() as u64,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, aux_len),
                scn.aux.len() as u64,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, precomputed_sym_len),
                scn.precomputed_sym.len() as u64,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, main_sym_len),
                scn.main_sym.len() as u64,
            );
            wl(
                &mut mem,
                offset_of!(ReducedOpeningLayout, aux_sym_len),
                scn.aux_sym.len() as u64,
            );
            register_ro_layout(&mem, layout_addr).unwrap();

            // Per-query six eval-slice base pointers.
            for (i, ptr) in [
                scn.precomputed_addr,
                scn.main_addr,
                scn.aux_addr,
                scn.precomputed_sym_addr,
                scn.main_sym_addr,
                scn.aux_sym_addr,
            ]
            .into_iter()
            .enumerate()
            {
                mem.store_doubleword(evals_addr + (i as u64) * 8, ptr)
                    .unwrap();
            }

            for row_idx in 0..scn.ood_height {
                reduced_opening_row_inplace(&mut mem, row_idx as u64, evals_addr, out_ptr).unwrap();
                let got = (get_ext(&mem, out_ptr), get_ext(&mem, out_ptr + EXT_STRIDE));
                let want = scn.ref_row_sums(row_idx);
                assert_eq!(
                    got.0.value(),
                    want.0.value(),
                    "seed {seed:#x} row {row_idx}"
                );
                assert_eq!(
                    got.1.value(),
                    want.1.value(),
                    "seed {seed:#x} sym row {row_idx}"
                );
            }
        }
    }

    /// The in-place row ecall before any REGISTER_RO_LAYOUT is a clean error, not
    /// a panic or garbage read.
    #[test]
    fn row_inplace_without_layout_errors() {
        // Fresh thread => thread-local layout is None.
        std::thread::spawn(|| {
            let mut mem = Memory::default();
            assert!(matches!(
                reduced_opening_row_inplace(&mut mem, 0, 0x7_000, 0x8_000),
                Err(ExecutionError::SimReducedOpeningNoLayout)
            ));
        })
        .join()
        .unwrap();
    }

    // --- Level B ------------------------------------------------------------

    struct QueryScenario {
        row: Scenario,
        evaluation_point: F,
        evaluation_point_sym: F,
        primitive_root: F,
        z: E,
        z_pow: E,
        h_sum_zpow: E,
        gammas: Vec<E>,
        composition: Vec<E>,
        composition_sym: Vec<E>,
        ood_row_sum: Vec<E>,
        number_of_parts: usize,
        // addresses
        evaluation_point_addr: u64,
        evaluation_point_sym_addr: u64,
        primitive_root_addr: u64,
        z_addr: u64,
        z_pow_addr: u64,
        h_sum_zpow_addr: u64,
        gammas_addr: u64,
        composition_addr: u64,
        composition_sym_addr: u64,
        ood_row_sum_addr: u64,
    }

    fn build_query_scenario() -> QueryScenario {
        let mut row = build_scenario(0xBEEF, 2, 3, 4, 6, 4, vec![0, 2, 5, 7]);
        let mut rng = Rng(0xABCDEF);
        let number_of_parts = 3;
        let evaluation_point = rng.base();
        let evaluation_point_sym = rng.base();
        let primitive_root = rng.base();
        let z = rng.ext();
        let z_pow = rng.ext();
        let h_sum_zpow = rng.ext();
        let gammas: Vec<E> = (0..number_of_parts).map(|_| rng.ext()).collect();
        let composition: Vec<E> = (0..number_of_parts).map(|_| rng.ext()).collect();
        let composition_sym: Vec<E> = (0..number_of_parts).map(|_| rng.ext()).collect();
        let ood_row_sum: Vec<E> = (0..row.ood_height).map(|_| rng.ext()).collect();

        // Append these into the scenario's memory at fresh addresses.
        let mut next = 0x40_000u64;
        let mut alloc = |n: u64| {
            let a = next;
            next += n.next_multiple_of(8);
            a
        };
        let evaluation_point_addr = alloc(8);
        put_base(&mut row.mem, evaluation_point_addr, &[evaluation_point]);
        let evaluation_point_sym_addr = alloc(8);
        put_base(
            &mut row.mem,
            evaluation_point_sym_addr,
            &[evaluation_point_sym],
        );
        let primitive_root_addr = alloc(8);
        put_base(&mut row.mem, primitive_root_addr, &[primitive_root]);
        let z_addr = alloc(EXT_STRIDE);
        put_ext(&mut row.mem, z_addr, &[z]);
        let z_pow_addr = alloc(EXT_STRIDE);
        put_ext(&mut row.mem, z_pow_addr, &[z_pow]);
        let h_sum_zpow_addr = alloc(EXT_STRIDE);
        put_ext(&mut row.mem, h_sum_zpow_addr, &[h_sum_zpow]);
        let gammas_addr = alloc(gammas.len() as u64 * EXT_STRIDE + 8);
        put_ext(&mut row.mem, gammas_addr, &gammas);
        let composition_addr = alloc(composition.len() as u64 * EXT_STRIDE + 8);
        put_ext(&mut row.mem, composition_addr, &composition);
        let composition_sym_addr = alloc(composition_sym.len() as u64 * EXT_STRIDE + 8);
        put_ext(&mut row.mem, composition_sym_addr, &composition_sym);
        let ood_row_sum_addr = alloc(ood_row_sum.len() as u64 * EXT_STRIDE + 8);
        put_ext(&mut row.mem, ood_row_sum_addr, &ood_row_sum);

        QueryScenario {
            row,
            evaluation_point,
            evaluation_point_sym,
            primitive_root,
            z,
            z_pow,
            h_sum_zpow,
            gammas,
            composition,
            composition_sym,
            ood_row_sum,
            number_of_parts,
            evaluation_point_addr,
            evaluation_point_sym_addr,
            primitive_root_addr,
            z_addr,
            z_pow_addr,
            h_sum_zpow_addr,
            gammas_addr,
            composition_addr,
            composition_sym_addr,
            ood_row_sum_addr,
        }
    }

    impl QueryScenario {
        fn query_input_mem(&self, input_addr: u64) -> Memory {
            let mut mem = self.row.mem.clone();
            use core::mem::offset_of;
            let scn = &self.row;
            macro_rules! set {
                ($field:ident, $v:expr) => {
                    mem.store_doubleword(
                        input_addr + offset_of!(ReducedOpeningQueryInput, $field) as u64,
                        $v,
                    )
                    .unwrap()
                };
            }
            set!(evaluation_point_ptr, self.evaluation_point_addr);
            set!(evaluation_point_sym_ptr, self.evaluation_point_sym_addr);
            set!(primitive_root_ptr, self.primitive_root_addr);
            set!(z_ptr, self.z_addr);
            set!(z_pow_ptr, self.z_pow_addr);
            set!(h_sum_zpow_ptr, self.h_sum_zpow_addr);
            set!(ood_height, scn.ood_height as u64);
            set!(ood_width, scn.ood_width as u64);
            set!(number_of_parts, self.number_of_parts as u64);
            set!(step_size, scn.step_size as u64);
            set!(precomputed_ptr, scn.precomputed_addr);
            set!(precomputed_len, scn.precomputed.len() as u64);
            set!(main_ptr, scn.main_addr);
            set!(main_len, scn.main.len() as u64);
            set!(aux_ptr, scn.aux_addr);
            set!(aux_len, scn.aux.len() as u64);
            set!(precomputed_sym_ptr, scn.precomputed_sym_addr);
            set!(precomputed_sym_len, scn.precomputed_sym.len() as u64);
            set!(main_sym_ptr, scn.main_sym_addr);
            set!(main_sym_len, scn.main_sym.len() as u64);
            set!(aux_sym_ptr, scn.aux_sym_addr);
            set!(aux_sym_len, scn.aux_sym.len() as u64);
            set!(composition_ptr, self.composition_addr);
            set!(composition_sym_ptr, self.composition_sym_addr);
            set!(coeff_col_ptrs_ptr, scn.coeff_col_ptrs_addr);
            set!(gammas_ptr, self.gammas_addr);
            set!(ood_row_sum_ptr, self.ood_row_sum_addr);
            set!(next_row_cols_ptr, scn.next_row_cols_addr);
            set!(next_row_cols_len, scn.next_row_cols.len() as u64);
            mem
        }

        /// Independent reimplementation of the whole verifier pair function.
        fn reference(&self) -> (E, E) {
            let scn = &self.row;
            let mut denoms: Vec<E> = Vec::new();
            let mut current_z = self.z;
            for _ in 0..scn.ood_height {
                denoms.push(&self.evaluation_point - &current_z);
                current_z = &self.primitive_root * &current_z;
            }
            let mut current_z = self.z;
            for _ in 0..scn.ood_height {
                denoms.push(&self.evaluation_point_sym - &current_z);
                current_z = &self.primitive_root * &current_z;
            }
            FieldElement::inplace_batch_inverse(&mut denoms).unwrap();
            let (denoms_trace, denoms_trace_sym) = denoms.split_at(scn.ood_height);

            let mut trace_term = E::zero();
            let mut trace_term_sym = E::zero();
            for row_idx in 0..scn.ood_height {
                let (brs, brs_sym) = scn.ref_row_sums(row_idx);
                let ood = &self.ood_row_sum[row_idx];
                trace_term += &denoms_trace[row_idx] * &(&brs - ood);
                trace_term_sym += &denoms_trace_sym[row_idx] * &(&brs_sym - ood);
            }

            let mut denom_pair = [
                &self.evaluation_point - &self.z_pow,
                &self.evaluation_point_sym - &self.z_pow,
            ];
            FieldElement::inplace_batch_inverse(&mut denom_pair).unwrap();
            let [denom_composition, denom_composition_sym] = denom_pair;

            let mut h_sum = E::zero();
            let mut h_sum_sym = E::zero();
            for j in 0..self.number_of_parts {
                h_sum += &self.composition[j] * &self.gammas[j];
                h_sum_sym += &self.composition_sym[j] * &self.gammas[j];
            }
            let h_terms = (&h_sum - &self.h_sum_zpow) * denom_composition;
            let h_terms_sym = (&h_sum_sym - &self.h_sum_zpow) * denom_composition_sym;
            (trace_term + h_terms, trace_term_sym + h_terms_sym)
        }
    }

    #[test]
    fn level_b_query_matches_reference() {
        let scn = build_query_scenario();
        let input_addr = 0x60_000u64;
        let out_ptr = 0x50_000u64;
        let mut mem = scn.query_input_mem(input_addr);
        reduced_opening_query(&mut mem, input_addr, out_ptr).unwrap();
        let got = (get_ext(&mem, out_ptr), get_ext(&mem, out_ptr + EXT_STRIDE));
        let want = scn.reference();
        assert_eq!(got.0.value(), want.0.value(), "deep_eval");
        assert_eq!(got.1.value(), want.1.value(), "deep_eval_sym");
    }

    #[test]
    fn limb_layout_round_trips_noncanonical() {
        // A base limb == p and an extension limb == p (non-canonical zeros)
        // must be read back and re-emitted bit-for-bit.
        const P: u64 = 0xFFFF_FFFF_0000_0001; // Goldilocks prime.
        let mut mem = Memory::default();
        put_ext(
            &mut mem,
            0x100,
            &[E::from_raw([
                F::from_raw(P),
                F::from_raw(1),
                F::from_raw(P),
            ])],
        );
        let e = get_ext(&mem, 0x100);
        assert_eq!(*e.value()[0].value(), P);
        assert_eq!(*e.value()[1].value(), 1);
        assert_eq!(*e.value()[2].value(), P);
    }

    // --- Full ECALL dispatch (register mapping + a7 routing) ----------------

    use crate::vm::instruction::decoding::Instruction;
    use crate::vm::registers::Registers;

    /// Drives `EcallEbreak` end-to-end so the a7 routing and the a0/a1/a2
    /// register→argument mapping in `execution.rs` are exercised (the handler
    /// tests above call the functions directly and can't catch a swapped reg).
    #[test]
    fn dispatch_routes_row_ecall_with_correct_registers() {
        let scn = build_scenario(0xD15, 2, 3, 4, 6, 4, vec![0, 2, 5, 7]);
        let input_addr = 0x9_000u64;
        let out_ptr = 0x8_000u64;
        let row_idx = 4u64; // a pruned row, to exercise next_row_cols too
        let mut mem = scn.row_input(input_addr);

        let mut regs = Registers::default();
        regs.write(17, REDUCED_OPENING_ROW_SYSCALL_NUMBER).unwrap(); // a7
        regs.write(10, input_addr).unwrap(); // a0
        regs.write(11, row_idx).unwrap(); // a1
        regs.write(12, out_ptr).unwrap(); // a2
        let mut pc = 0u64;
        Instruction::EcallEbreak
            .run(&mut pc, &mut regs, &mut mem)
            .unwrap();

        let got = (get_ext(&mem, out_ptr), get_ext(&mem, out_ptr + EXT_STRIDE));
        let want = scn.ref_row_sums(row_idx as usize);
        assert_eq!(got.0.value(), want.0.value());
        assert_eq!(got.1.value(), want.1.value());
    }

    #[test]
    fn dispatch_routes_query_ecall_with_correct_registers() {
        let scn = build_query_scenario();
        let input_addr = 0x60_000u64;
        let out_ptr = 0x50_000u64;
        let mut mem = scn.query_input_mem(input_addr);

        let mut regs = Registers::default();
        regs.write(17, REDUCED_OPENING_QUERY_SYSCALL_NUMBER)
            .unwrap(); // a7
        regs.write(10, input_addr).unwrap(); // a0
        regs.write(11, out_ptr).unwrap(); // a1
        let mut pc = 0u64;
        Instruction::EcallEbreak
            .run(&mut pc, &mut regs, &mut mem)
            .unwrap();

        let got = (get_ext(&mem, out_ptr), get_ext(&mem, out_ptr + EXT_STRIDE));
        let want = scn.reference();
        assert_eq!(got.0.value(), want.0.value());
        assert_eq!(got.1.value(), want.1.value());
    }
}
