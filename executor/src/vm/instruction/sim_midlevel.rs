//! Host-side handlers for the MID-LEVEL accelerator measurement ecalls
//! (sim/27). MEASUREMENT-ONLY, never proven.
//!
//! Each stub computes the CORRECT value of a recursion-verifier hot bucket
//! host-side, in a single VM cycle, and writes it back to guest memory. Like
//! the DEEP reduced-opening stubs (`sim_reduced_opening`) these let us measure
//! the optimistic guest-cycle ceiling of a future in-circuit accelerator chip
//! WITHOUT building its AIR yet. They are SOUND-SHAPED passthroughs: the host
//! computes each answer ONLY from the inputs the guest passes (never by peeking
//! at expected results or proof internals), so a tampered blob still cascades to
//! a mismatch and the guest still REJECTS. They drive no chip table, so a build
//! that emits them is EXECUTE-ONLY — NEVER prove it (the Ecall LogUp bus would
//! unbalance, like `Print`).
//!
//! The stubs and their buckets:
//!   * `SIM_POLY_EVAL`      (MAX-60) — FRI terminal-codeword FFT evaluation.
//!   * `SIM_POW`            (MAX-61) — Fp3 / Goldilocks `pow` (zerofier etc.).
//!   * `SIM_FOLD_CHAIN`     (MAX-62) — the per-query FRI fold butterfly chain.
//!   * `SIM_CONSTRAINT_EVAL`(MAX-63) — per-table OOD constraint evaluation.
//!   * `SIM_DOMAIN_POINTS`  (MAX-64) — batched primary FRI query eval points.
//!
//! All but `SIM_CONSTRAINT_EVAL` are true host offloads (the answer is computed
//! here from the guest's inputs, so the guest cycle count drops).
//! `SIM_CONSTRAINT_EVAL` is DIFFERENT: it is an op-count PROXY only, handled
//! inline in `execution.rs` (no
//! function here). The constraint-eval ceiling is NOT offloadable in this model
//! — the per-table `compute_transition` is AIR-specific compiled Rust living in
//! `crypto/stark`, and this executor depends only on `math`/`ecsm` (it cannot,
//! and architecturally should not, run the verification stack). So the guest
//! still runs the real evaluation and only tallies its transition-constraint
//! count, which prices the future data-driven expression chip in program length.
//!
//! ABI structs live in [`math::sim_midlevel`] (shared with the guest-side
//! marshaling in `crypto/stark` + `crypto/math`), specialised to the recursion
//! guest's concrete field choice: base = Goldilocks (1 limb = 8 bytes), ext =
//! degree-3 Goldilocks (`[FpE; 3]` = 3 limbs = 24 bytes).

use crate::vm::instruction::execution::ExecutionError;
use crate::vm::memory::{Memory, MemoryError};
use core::cell::RefCell;
use math::fft::bit_reversing::reverse_index;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::IsFFTField;
use math::polynomial::Polynomial;
use math::sim_midlevel::{
    ConstraintEvalInput, DomainPointsInput, FoldChainInput, PolyEvalInput, RegisterCommitInput,
};

/// Syscall number for `SIM_POLY_EVAL` — `u64::MAX - 60`.
pub const SIM_POLY_EVAL_SYSCALL_NUMBER: u64 = u64::MAX - 60;
/// Syscall number for `SIM_POW` — `u64::MAX - 61`.
pub const SIM_POW_SYSCALL_NUMBER: u64 = u64::MAX - 61;
/// Syscall number for `SIM_FOLD_CHAIN` — `u64::MAX - 62`.
pub const SIM_FOLD_CHAIN_SYSCALL_NUMBER: u64 = u64::MAX - 62;
/// Syscall number for `SIM_CONSTRAINT_EVAL` — `u64::MAX - 63`.
pub const SIM_CONSTRAINT_EVAL_SYSCALL_NUMBER: u64 = u64::MAX - 63;
/// Syscall number for `SIM_DOMAIN_POINTS` — `u64::MAX - 64`.
pub const SIM_DOMAIN_POINTS_SYSCALL_NUMBER: u64 = u64::MAX - 64;
/// Syscall number for `SIM_REGISTER_COMMIT` — `u64::MAX - 65`.
pub const SIM_REGISTER_COMMIT_SYSCALL_NUMBER: u64 = u64::MAX - 65;
/// Syscall number for `SIM_VERIFY_PATH_BATCH` — `u64::MAX - 66`.
pub const SIM_VERIFY_PATH_BATCH_SYSCALL_NUMBER: u64 = u64::MAX - 66;

type F = FieldElement<GoldilocksField>;
type E = FieldElement<Degree3GoldilocksExtensionField>;

/// Bytes per extension `FieldElement` (`[FpE; 3]` = 3 `u64`).
const EXT_STRIDE: u64 = 24;

/// Read a single `u64` ABI-struct field at `base + offset`.
#[inline]
fn field(memory: &Memory, base: u64, offset: usize) -> Result<u64, MemoryError> {
    memory.load_doubleword(base.wrapping_add(offset as u64))
}

/// Read an extension element (3 little-endian limbs) at `addr`.
#[inline]
fn read_ext(memory: &Memory, addr: u64) -> Result<E, MemoryError> {
    Ok(E::from_raw([
        F::from_raw(memory.load_doubleword(addr)?),
        F::from_raw(memory.load_doubleword(addr.wrapping_add(8))?),
        F::from_raw(memory.load_doubleword(addr.wrapping_add(16))?),
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

// =============================================================================
// SIM_POW (MAX-61) — Fp3 / Goldilocks exponentiation.
//
// Routes the recursion verifier's hot `pow` calls (Fp3 `z^N` in the OOD zerofier
// denominators, Goldilocks `g^(N-1)` in the end-exemption roots, plus the
// smaller `pow`s) through the host. `num_limbs` selects the field: 1 =
// Goldilocks, 3 = Fp3. The host reads the RAW limbs the guest stored and applies
// the SAME square-and-multiply the guest software `pow` would (via the same
// `math` field types), so the bytes written back are bit-identical to the
// in-guest result. SOUND-SHAPED: the answer is `base^exp` of the exact
// (untrusted) base/exponent the guest passes; a tampered blob changes `z` or the
// trace root and the wrong power cascades to a rejected proof.
// =============================================================================

/// Handle `SIM_POW`. `base_ptr` points at `num_limbs` little-endian raw limbs;
/// the host writes `base^exponent` (same width) to `out_ptr`.
pub fn sim_pow(
    memory: &mut Memory,
    base_ptr: u64,
    num_limbs: u64,
    exponent: u64,
    out_ptr: u64,
) -> Result<(), ExecutionError> {
    match num_limbs {
        1 => {
            let base = F::from_raw(memory.load_doubleword(base_ptr)?);
            let result = base.pow(exponent);
            memory.store_doubleword(out_ptr, *result.value())?;
        }
        3 => {
            let base = read_ext(memory, base_ptr)?;
            let result = base.pow(exponent);
            write_ext(memory, out_ptr, &result)?;
        }
        other => return Err(ExecutionError::SimPowBadWidth(other)),
    }
    Ok(())
}

// =============================================================================
// SIM_POLY_EVAL (MAX-60) — FRI terminal-codeword evaluation.
//
// The verifier reconstructs the FRI terminal codeword from the final-poly
// coefficients via a coset FFT (`terminal_codeword_from_coeffs`), then reads it
// only at the ~queried positions. This stub replaces the O(n log n) FFT-of-all
// with a Horner evaluation at ONLY the positions the queries actually hit.
//
// `terminal_codeword_from_coeffs` computes `natural[i] = P(offset · ω^i)` (ω =
// the size-`codeword_len` primitive root of unity, `P` = the final polynomial),
// then bit-reverse-permutes to FRI order. So the value at FRI-order position `p`
// is `P(offset · ω^{reverse_index(p, codeword_len)})`. The host evaluates that
// directly with Horner and writes it into the caller's full-length codeword
// buffer at slot `p` — leaving un-queried slots untouched (the verify path never
// reads them). SOUND-SHAPED: values come only from the (untrusted) coeffs and
// the honest positions/offset; tampered coeffs give a wrong polynomial whose
// values miss the folded `v` and the proof rejects.
// =============================================================================

/// Handle `SIM_POLY_EVAL`. `input_ptr` points at a [`PolyEvalInput`].
pub fn sim_poly_eval(memory: &mut Memory, input_ptr: u64) -> Result<(), ExecutionError> {
    use core::mem::offset_of;
    let coeffs_ptr = field(memory, input_ptr, offset_of!(PolyEvalInput, coeffs_ptr))?;
    let coeffs_len = field(memory, input_ptr, offset_of!(PolyEvalInput, coeffs_len))?;
    let terminal_offset_ptr = field(
        memory,
        input_ptr,
        offset_of!(PolyEvalInput, terminal_offset_ptr),
    )?;
    let terminal_offset = F::from_raw(memory.load_doubleword(terminal_offset_ptr)?);
    let codeword_len = field(memory, input_ptr, offset_of!(PolyEvalInput, codeword_len))?;
    let positions_ptr = field(memory, input_ptr, offset_of!(PolyEvalInput, positions_ptr))?;
    let positions_len = field(memory, input_ptr, offset_of!(PolyEvalInput, positions_len))?;
    let out_ptr = field(memory, input_ptr, offset_of!(PolyEvalInput, out_ptr))?;

    if !codeword_len.is_power_of_two() || codeword_len == 0 {
        return Err(ExecutionError::SimPolyEvalBadDomain(codeword_len));
    }

    // Read the final-poly coefficients (extension elements, 3 limbs each).
    let coeffs: Vec<E> = (0..coeffs_len)
        .map(|k| read_ext(memory, coeffs_ptr.wrapping_add(k.wrapping_mul(EXT_STRIDE))))
        .collect::<Result<_, _>>()?;
    let poly = Polynomial::new(&coeffs);

    // ω: the size-`codeword_len` primitive root of unity in the base field —
    // exactly the root `evaluate_offset_fft` uses (see the doc comment above).
    let order = codeword_len.trailing_zeros() as u64;
    let omega = GoldilocksField::get_primitive_root_of_unity(order)
        .map_err(|_| ExecutionError::SimPolyEvalBadDomain(codeword_len))?;

    for j in 0..positions_len {
        // FRI-order position → natural-order index → coset evaluation point.
        let p = memory.load_doubleword(positions_ptr.wrapping_add(j.wrapping_mul(8)))?;
        let natural_idx = reverse_index(p as usize, codeword_len);
        let x = &terminal_offset * omega.pow(natural_idx as u64);
        // Horner: evaluate the extension-coeff polynomial at the base point `x`
        // (embedded into the extension). Value-identical to the FFT codeword.
        let value = poly.evaluate(&x.to_extension::<Degree3GoldilocksExtensionField>());
        write_ext(
            memory,
            out_ptr.wrapping_add(p.wrapping_mul(EXT_STRIDE)),
            &value,
        )?;
    }
    Ok(())
}

// =============================================================================
// SIM_DOMAIN_POINTS (MAX-64) — batched primary FRI query evaluation points.
//
// step_3 needs υ_i = coset_offset · lde_primitive_root^{reverse_index(2·iota_i,
// lde_length)} for every query. In software each is one `pow` (a SIM_POW ecall
// under `sim-pow`); this batches all of them into one host call that computes
// each point with the SAME field ops the guest would and writes it back raw, so
// the bytes are bit-identical to the in-guest result. SOUND-SHAPED: every point
// is a pure function of the honest, public domain params + the guest's iotas; a
// tampered blob shifts them and the wrong points cascade to a rejected proof.
// =============================================================================

/// Handle `SIM_DOMAIN_POINTS`. `input_ptr` points at a [`DomainPointsInput`];
/// writes `iotas_len` base-field evaluation points (1 limb each) to `out_ptr`.
pub fn sim_domain_points(memory: &mut Memory, input_ptr: u64) -> Result<(), ExecutionError> {
    use core::mem::offset_of;
    let iotas_ptr = field(memory, input_ptr, offset_of!(DomainPointsInput, iotas_ptr))?;
    let iotas_len = field(memory, input_ptr, offset_of!(DomainPointsInput, iotas_len))?;
    let lde_length = field(memory, input_ptr, offset_of!(DomainPointsInput, lde_length))?;
    let root_ptr = field(
        memory,
        input_ptr,
        offset_of!(DomainPointsInput, lde_primitive_root_ptr),
    )?;
    let coset_ptr = field(
        memory,
        input_ptr,
        offset_of!(DomainPointsInput, coset_offset_ptr),
    )?;
    let out_ptr = field(memory, input_ptr, offset_of!(DomainPointsInput, out_ptr))?;

    if !lde_length.is_power_of_two() || lde_length == 0 {
        return Err(ExecutionError::SimPolyEvalBadDomain(lde_length));
    }
    let lde_root = F::from_raw(memory.load_doubleword(root_ptr)?);
    let coset_offset = F::from_raw(memory.load_doubleword(coset_ptr)?);
    for j in 0..iotas_len {
        let iota = memory.load_doubleword(iotas_ptr.wrapping_add(j.wrapping_mul(8)))?;
        // υ = coset_offset · lde_primitive_root^{reverse_index(2·iota, lde_length)}
        // — exactly `VerifierDomain::lde_coset_element(reverse_index(iota*2, ·))`.
        let natural_idx = reverse_index((iota as usize).wrapping_mul(2), lde_length);
        let point = &coset_offset * lde_root.pow(natural_idx as u64);
        memory.store_doubleword(out_ptr.wrapping_add(j.wrapping_mul(8)), *point.value())?;
    }
    Ok(())
}

// =============================================================================
// SIM_REGISTER_COMMIT (MAX-65) — REGISTER preprocessed commitment.
//
// Each continuation epoch the verifier recomputes the REGISTER preprocessed
// commitment (FFT-interpolate + LDE-evaluate + Merkle-commit over OFFSET / INIT /
// FINI) to bind the proof's FINI column to R_{i+1}. The build lives in
// `crypto/stark` + prover, which this executor cannot depend on, so it is served
// through a CLI-registered evaluator (like SIM_CONSTRAINT_EVAL) that runs the
// prover's REAL `compute_precomputed_commitment_with_fini` — identical code,
// identical bytes. SOUND-SHAPED: the commitment is a pure function of the
// (public) register arrays the guest passes; a forged commitment breaks the
// downstream preprocessed-AIR opening and the proof rejects.
// =============================================================================

/// Handle `SIM_REGISTER_COMMIT`. `input_ptr` points at a [`RegisterCommitInput`];
/// reads the `init`/`fini` `u32` arrays, calls the CLI-registered evaluator, and
/// writes the 32-byte commitment (4 `u64` limbs) to `out_ptr`. Returns the
/// number of INIT entries committed, for the CLI's counter.
pub fn sim_register_commit(memory: &mut Memory, input_ptr: u64) -> Result<u64, ExecutionError> {
    use core::mem::offset_of;
    let init_ptr = field(memory, input_ptr, offset_of!(RegisterCommitInput, init_ptr))?;
    let init_len = field(memory, input_ptr, offset_of!(RegisterCommitInput, init_len))?;
    let fini_ptr = field(memory, input_ptr, offset_of!(RegisterCommitInput, fini_ptr))?;
    let fini_len = field(memory, input_ptr, offset_of!(RegisterCommitInput, fini_len))?;
    let out_ptr = field(memory, input_ptr, offset_of!(RegisterCommitInput, out_ptr))?;

    let read_u32s = |ptr: u64, len: u64| -> Result<Vec<u32>, MemoryError> {
        (0..len)
            .map(|i| memory.load_word(ptr.wrapping_add(i.wrapping_mul(4))))
            .collect()
    };
    let init = read_u32s(init_ptr, init_len)?;
    let fini = read_u32s(fini_ptr, fini_len)?;

    // The CLI-registered evaluator runs the prover's real commitment build; if
    // absent (never registered) fall back to a zero commitment, which makes the
    // guest's downstream preprocessed-AIR opening reject.
    let commitment = REGISTER_COMMIT_EVALUATOR.with(|slot| match slot.borrow().as_ref() {
        Some(evaluator) => evaluator(&init, &fini),
        None => [0u8; 32],
    });
    for (i, chunk) in commitment.chunks_exact(8).enumerate() {
        let limb = u64::from_le_bytes(chunk.try_into().unwrap());
        memory.store_doubleword(out_ptr.wrapping_add((i as u64).wrapping_mul(8)), limb)?;
    }
    Ok(init_len)
}

// =============================================================================
// SIM_FOLD_CHAIN (MAX-62) — per-query FRI fold butterfly chain.
//
// Per FRI query the verifier walks the committed layers folding pairwise:
// `v_{i+1} = (v_i + v_i^sym) + (ω^{-2^i}·ζ_{i+1})·(v_i − v_i^sym)`, starting from
// the deep-composition values `p0`/`p0_sym` at υ/−υ. This stub computes the WHOLE
// arithmetic chain host-side and returns EVERY layer value (so the guest keeps
// doing the per-layer Merkle path verification, which is VERIFY_PATH's job) plus
// the terminal value (which the guest still compares against the terminal
// codeword). SOUND-SHAPED: the chain is a pure function of the (untrusted) proof
// openings + the honest betas/points; a tampered `evaluation_sym` changes both
// the Merkle leaf (path rejects) and the folded `v` (terminal check rejects).
// =============================================================================

/// Handle `SIM_FOLD_CHAIN`. `input_ptr` points at a [`FoldChainInput`]; writes
/// `num_layers + 1` extension values to `out_ptr` (the per-layer values used for
/// the Merkle checks, followed by the terminal value).
pub fn sim_fold_chain(memory: &mut Memory, input_ptr: u64) -> Result<(), ExecutionError> {
    use core::mem::offset_of;
    let p0 = read_ext(
        memory,
        field(memory, input_ptr, offset_of!(FoldChainInput, p0_ptr))?,
    )?;
    let p0_sym = read_ext(
        memory,
        field(memory, input_ptr, offset_of!(FoldChainInput, p0_sym_ptr))?,
    )?;
    let eval_point_inv_ptr = field(
        memory,
        input_ptr,
        offset_of!(FoldChainInput, eval_point_inv_ptr),
    )?;
    let eval_point_inv = F::from_raw(memory.load_doubleword(eval_point_inv_ptr)?);
    let zetas_ptr = field(memory, input_ptr, offset_of!(FoldChainInput, zetas_ptr))?;
    let layers_sym_ptr = field(
        memory,
        input_ptr,
        offset_of!(FoldChainInput, layers_sym_ptr),
    )?;
    let num_layers = field(memory, input_ptr, offset_of!(FoldChainInput, num_layers))?;
    let out_ptr = field(memory, input_ptr, offset_of!(FoldChainInput, out_ptr))?;

    // Initial butterfly: v = (p0 + p0_sym) + (υ^{-1}·ζ0)·(p0 − p0_sym). The
    // subfield product `base · ext` (`&F * &E`) is exactly `Fp3Fma::base_mul`'s
    // default, and `c·d + e` is `Fp3Fma::fma`'s default — byte-identical.
    let zeta0 = read_ext(memory, zetas_ptr)?;
    let c0 = &eval_point_inv * &zeta0;
    let mut v = &c0 * &(&p0 - &p0_sym) + &(&p0 + &p0_sym);
    write_ext(memory, out_ptr, &v)?;

    // Fold through each committed layer. `pt` = (υ^{-1})^{2^{i+1}} (the squares
    // the guest iterator yields); `ζ_{i+1}` is the next folding challenge.
    let mut pt = eval_point_inv.square();
    for i in 0..num_layers {
        let eval_sym = read_ext(
            memory,
            layers_sym_ptr.wrapping_add(i.wrapping_mul(EXT_STRIDE)),
        )?;
        let zeta = read_ext(
            memory,
            zetas_ptr.wrapping_add((i + 1).wrapping_mul(EXT_STRIDE)),
        )?;
        let c = &pt * &zeta;
        v = &c * &(&v - &eval_sym) + &(&v + &eval_sym);
        write_ext(
            memory,
            out_ptr.wrapping_add((i + 1).wrapping_mul(EXT_STRIDE)),
            &v,
        )?;
        pt = pt.square();
    }
    Ok(())
}

// =============================================================================
// SIM_CONSTRAINT_EVAL v2 (MAX-63) — per-table OOD constraint evaluation.
//
// Unlike the other three stubs (which compute host-side using only `math`), the
// constraint evaluation needs the recursion verifier's per-table constraint IR
// (`crypto/stark`), which this executor cannot depend on. The dependency cycle
// is broken by preloading: the CLI (which deps both prover and stark) captures
// each table's `ConstraintProgram` in the guest's exact compute_transition order
// and registers an EVALUATOR closure here. The handler reads the guest's OOD
// frame + LogUp challenges into a plain [`ConstraintEvalRequest`], hands it to
// the closure (which reconstructs the frame + runs `eval_program_verifier`), and
// writes the per-constraint evaluations back. SOUND-SHAPED: the answer is a pure
// function of the (untrusted) OOD frame + challenges the guest passes and the
// statically-known constraint programs (verifying-key material) — a tampered
// blob gives a wrong frame whose constraint values miss the composition check
// and the proof rejects.
// =============================================================================

/// Plain-data request handed to the registered constraint evaluator. All field
/// elements are raw Goldilocks limbs (base = 1, extension = 3), so this crate
/// needs no `crypto/stark` types; the CLI-side closure reconstructs them.
pub struct ConstraintEvalRequest {
    /// Global compute_transition sequence index (keys the preloaded program).
    pub seq_index: usize,
    /// OOD frame grid, row-major `height × width` extension elements.
    pub frame_data: Vec<[u64; 3]>,
    pub width: usize,
    pub height: usize,
    /// Number of leading main-trace columns (the frame's main/aux split point).
    pub num_main: usize,
    pub step_size: usize,
    pub rap_challenges: Vec<[u64; 3]>,
    pub alpha_powers: Vec<[u64; 3]>,
    pub table_offset: [u64; 3],
    pub num_constraints: usize,
}

/// The registered evaluator returns the per-constraint extension evaluations
/// (`num_constraints` × 3 limbs) plus the program's node count (priced as chip
/// program length). Boxed `dyn Fn` so the CLI can inject the stark-dependent
/// evaluation without this crate depending on stark.
type ConstraintEvaluator = Box<dyn Fn(&ConstraintEvalRequest) -> (Vec<[u64; 3]>, u64)>;

thread_local! {
    static CONSTRAINT_EVALUATOR: RefCell<Option<ConstraintEvaluator>> = const { RefCell::new(None) };
}

/// Register the constraint evaluator (called by the CLI before executing a
/// `sim-constraint-eval` build). MEASUREMENT-ONLY plumbing.
pub fn set_constraint_evaluator(evaluator: ConstraintEvaluator) {
    CONSTRAINT_EVALUATOR.with(|slot| *slot.borrow_mut() = Some(evaluator));
}

/// The registered REGISTER-commit evaluator maps the guest's `(init, fini)`
/// register arrays to the 32-byte preprocessed commitment. Boxed `dyn Fn` so the
/// CLI can inject the prover-dependent FFT+LDE+Merkle build without this crate
/// depending on stark/prover.
type RegisterCommitEvaluator = Box<dyn Fn(&[u32], &[u32]) -> [u8; 32]>;

thread_local! {
    static REGISTER_COMMIT_EVALUATOR: RefCell<Option<RegisterCommitEvaluator>> =
        const { RefCell::new(None) };
}

/// Register the REGISTER-commit evaluator (called by the CLI before executing a
/// `sim-register-commit` build). MEASUREMENT-ONLY plumbing.
pub fn set_register_commit_evaluator(evaluator: RegisterCommitEvaluator) {
    REGISTER_COMMIT_EVALUATOR.with(|slot| *slot.borrow_mut() = Some(evaluator));
}

/// Read `count` extension elements (3 limbs each) starting at `ptr`.
fn read_ext_array(memory: &Memory, ptr: u64, count: u64) -> Result<Vec<[u64; 3]>, MemoryError> {
    (0..count)
        .map(|i| {
            let a = ptr.wrapping_add(i.wrapping_mul(EXT_STRIDE));
            Ok([
                memory.load_doubleword(a)?,
                memory.load_doubleword(a.wrapping_add(8))?,
                memory.load_doubleword(a.wrapping_add(16))?,
            ])
        })
        .collect()
}

/// Handle `SIM_CONSTRAINT_EVAL` v2. `input_ptr` points at a [`ConstraintEvalInput`].
/// Returns `(num_constraints, node_count)` for the CLI's per-table counter.
pub fn sim_constraint_eval(
    memory: &mut Memory,
    input_ptr: u64,
) -> Result<(u64, u64), ExecutionError> {
    use core::mem::offset_of;
    let f = |off: usize| memory.load_doubleword(input_ptr.wrapping_add(off as u64));
    let seq_index = f(offset_of!(ConstraintEvalInput, seq_index))?;
    let frame_ptr = f(offset_of!(ConstraintEvalInput, frame_ptr))?;
    let width = f(offset_of!(ConstraintEvalInput, width))?;
    let height = f(offset_of!(ConstraintEvalInput, height))?;
    let num_main = f(offset_of!(ConstraintEvalInput, num_main))?;
    let step_size = f(offset_of!(ConstraintEvalInput, step_size))?;
    let rap_ptr = f(offset_of!(ConstraintEvalInput, rap_challenges_ptr))?;
    let rap_len = f(offset_of!(ConstraintEvalInput, rap_challenges_len))?;
    let alpha_ptr = f(offset_of!(ConstraintEvalInput, alpha_powers_ptr))?;
    let alpha_len = f(offset_of!(ConstraintEvalInput, alpha_powers_len))?;
    let table_offset_ptr = f(offset_of!(ConstraintEvalInput, table_offset_ptr))?;
    let num_constraints = f(offset_of!(ConstraintEvalInput, num_constraints))?;
    let out_ptr = f(offset_of!(ConstraintEvalInput, out_ptr))?;

    let request = ConstraintEvalRequest {
        seq_index: seq_index as usize,
        frame_data: read_ext_array(memory, frame_ptr, width.saturating_mul(height))?,
        width: width as usize,
        height: height as usize,
        num_main: num_main as usize,
        step_size: step_size as usize,
        rap_challenges: read_ext_array(memory, rap_ptr, rap_len)?,
        alpha_powers: read_ext_array(memory, alpha_ptr, alpha_len)?,
        table_offset: read_ext_array(memory, table_offset_ptr, 1)?[0],
        num_constraints: num_constraints as usize,
    };

    let (evals, node_count) = CONSTRAINT_EVALUATOR
        .with(|slot| slot.borrow().as_ref().map(|ev| ev(&request)))
        .ok_or(ExecutionError::SimConstraintNoProgram)?;

    if evals.len() != num_constraints as usize {
        return Err(ExecutionError::SimConstraintBadProgram);
    }
    for (i, e) in evals.iter().enumerate() {
        let a = out_ptr.wrapping_add((i as u64).wrapping_mul(EXT_STRIDE));
        memory.store_doubleword(a, e[0])?;
        memory.store_doubleword(a.wrapping_add(8), e[1])?;
        memory.store_doubleword(a.wrapping_add(16), e[2])?;
    }
    Ok((num_constraints, node_count))
}
