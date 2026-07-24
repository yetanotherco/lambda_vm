//! Shared ABI structs for the MID-LEVEL accelerator measurement ecalls (sim/27).
//! MEASUREMENT-ONLY, never proven.
//!
//! These are the `input_ptr` payloads for the pointer-passing stub ecalls
//! (`SIM_POLY_EVAL`, `SIM_FOLD_CHAIN`) that let us measure the optimistic
//! guest-cycle ceiling of future mid-level accelerator chips (FRI terminal
//! evaluation, the FRI fold butterfly chain). `SIM_POW` passes its (base,
//! width, exponent, out) directly in registers and needs no struct.
//!
//! Same layout contract as [`crate::sim_ro`]: every field is a `u64`, so
//! `#[repr(C)]` gives field `i` at byte offset `8*i` and the executor reads each
//! with `core::mem::offset_of!` — field ORDER here is the ABI. Pointer fields are
//! guest virtual addresses; extension elements are 3 little-endian limbs (24
//! bytes); base elements are 1 limb (8 bytes).

/// `SIM_POLY_EVAL` input — evaluate the FRI terminal polynomial at the queried
/// codeword positions.
///
/// The verifier reconstructs the terminal codeword as `natural[i] =
/// P(terminal_offset · ω^i)` (ω = the size-`codeword_len` root of unity, `P` the
/// final-poly), bit-reverse-permuted to FRI order. The host evaluates `P` (via
/// Horner) at exactly the `positions` the queries hit and writes each into slot
/// `p` of the caller's full-length codeword buffer at `out_ptr` (un-queried
/// slots stay untouched; the verify path never reads them).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PolyEvalInput {
    /// Final-polynomial coefficients (extension elements, 3 limbs each).
    pub coeffs_ptr: u64,
    pub coeffs_len: u64,
    /// Pointer to the terminal coset offset (a base-field element, 1 limb). A
    /// pointer, not a value, so the ABI stays field-generic on the guest (the
    /// verifier's `Field::BaseType` is not statically `u64`).
    pub terminal_offset_ptr: u64,
    /// Full terminal codeword length (power of two).
    pub codeword_len: u64,
    /// FRI-order positions the queries actually read (`u64` each).
    pub positions_ptr: u64,
    pub positions_len: u64,
    /// Full-length codeword buffer (`codeword_len` extension elements); the host
    /// writes only the queried slots.
    pub out_ptr: u64,
}

/// Number of `u64` fields in [`PolyEvalInput`].
pub const POLY_EVAL_INPUT_FIELDS: usize = 7;

/// `SIM_DOMAIN_POINTS` input — the batched primary FRI query evaluation points.
///
/// `step_3_verify_fri` needs the primary FRI query points
/// `υ_i = coset_offset · lde_primitive_root^{reverse_index(2·iota_i, lde_length)}`
/// for every query. In software that is one `pow` per query (a SIM_POW ecall each
/// when `sim-pow` is on); this batches all `iotas_len` of them into ONE host
/// call. The host writes `iotas_len` base-field elements (1 limb each) to
/// `out_ptr`, in iota order. SOUND-SHAPED: every point is a pure function of the
/// honest, public domain parameters and the (transcript-derived) iotas the guest
/// passes — a tampered blob shifts the iotas/roots and the wrong points cascade
/// to a rejected proof.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainPointsInput {
    /// FRI query index challenges (`u64` each).
    pub iotas_ptr: u64,
    pub iotas_len: u64,
    /// LDE domain length (power of two) that sizes the bit-reverse index.
    pub lde_length: u64,
    /// Pointer to the LDE primitive root (base-field element, 1 limb). A pointer,
    /// not a value, so the ABI stays field-generic on the guest.
    pub lde_primitive_root_ptr: u64,
    /// Pointer to the coset offset (base-field element, 1 limb).
    pub coset_offset_ptr: u64,
    /// Output: `iotas_len` base-field evaluation points (1 limb each), in order.
    pub out_ptr: u64,
}

/// Number of `u64` fields in [`DomainPointsInput`].
pub const DOMAIN_POINTS_INPUT_FIELDS: usize = 6;

/// `SIM_REGISTER_COMMIT` input — offload the REGISTER preprocessed commitment.
///
/// Each continuation epoch the verifier recomputes the REGISTER preprocessed
/// commitment (FFT-interpolate + LDE-evaluate + Merkle-commit over the OFFSET /
/// INIT / FINI columns) to bind the proof's FINI column to `R_{i+1}`. This
/// offloads the whole build to the host, which recomputes it from the guest's
/// `init` / `fini` register arrays with the SAME prover code, writing the
/// 32-byte commitment (4 limbs) to `out_ptr`. SOUND-SHAPED: the commitment is a
/// pure function of the (public) register arrays the guest passes — a forged
/// commitment breaks the downstream preprocessed-AIR opening binding and the
/// proof rejects. `init` / `fini` are `u32` arrays (4-byte stride).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterCommitInput {
    /// INIT register values (`u32` each, 4-byte stride).
    pub init_ptr: u64,
    pub init_len: u64,
    /// FINI register values (`u32` each, 4-byte stride).
    pub fini_ptr: u64,
    pub fini_len: u64,
    /// Output: the 32-byte commitment (4 little-endian `u64` limbs).
    pub out_ptr: u64,
}

/// Number of `u64` fields in [`RegisterCommitInput`].
pub const REGISTER_COMMIT_INPUT_FIELDS: usize = 5;

/// `SIM_FOLD_CHAIN` input — the whole per-query FRI fold butterfly chain.
///
/// Starting from the deep-composition values `p0`/`p0_sym` at υ/−υ, the verifier
/// folds through `num_layers` committed layers:
/// `v_{i+1} = (v_i + s_i) + ((υ^{-1})^{2^{i+1}}·ζ_{i+1})·(v_i − s_i)`, where the
/// initial `v_0` uses `υ^{-1}·ζ_0` and `s_i` are the layer symmetric openings.
/// The host writes ALL `num_layers + 1` values (`v_0 .. v_{num_layers-1}` used
/// for the per-layer Merkle checks the guest still runs, then the terminal
/// `v_{num_layers}`) to `out_ptr`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FoldChainInput {
    /// Deep-composition value at υ (extension, 3 limbs).
    pub p0_ptr: u64,
    /// Deep-composition value at −υ (extension, 3 limbs).
    pub p0_sym_ptr: u64,
    /// Pointer to the inverse FRI evaluation point υ^{-1} (base-field element, 1
    /// limb). A pointer, not a value, so the ABI stays field-generic on the guest.
    pub eval_point_inv_ptr: u64,
    /// Folding challenges ζ (extension array, length `num_layers + 1`).
    pub zetas_ptr: u64,
    /// Per-layer symmetric openings (extension array, length `num_layers`).
    pub layers_sym_ptr: u64,
    /// Number of committed FRI layers folded.
    pub num_layers: u64,
    /// Output buffer for `num_layers + 1` extension values.
    pub out_ptr: u64,
}

/// Number of `u64` fields in [`FoldChainInput`].
pub const FOLD_CHAIN_INPUT_FIELDS: usize = 7;

/// `SIM_VERIFY_PATH_BATCH` input — verify EVERY committed FRI layer's Merkle
/// opening for one query in a single ecall.
///
/// For each layer `i` (`0..num_layers`) the host hashes the ordered leaf pair
/// `(evaluation, evaluation_sym)` — ordered by the layer index's low bit, exactly
/// as `verify_fri_layer_openings` does — then folds that leaf up the `i`-th auth
/// path to the committed root, ANDing the per-layer accept into a single byte at
/// `out_ptr`. This subsumes the per-layer `HASH_FELTS` + `VERIFY_PATH` ecalls
/// (one batch call per query instead of one pair per layer). The layer values
/// come from [`FoldChainInput`]'s output. SOUND-SHAPED like `VERIFY_PATH`: the
/// host recomputes the true root and reports the real accept, so a tampered
/// opening yields a mismatched root -> `0` -> the guest rejects.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifyPathBatchInput {
    /// Number of committed FRI layers.
    pub num_layers: u64,
    /// The query's layer-0 FRI index (`iota`); halved per layer for both the leaf
    /// ordering (low bit) and the path start position (`index >> 1`).
    pub start_index: u64,
    /// Contiguous 32-byte committed layer roots; `root[i]` at `roots_ptr + i*32`.
    pub roots_ptr: u64,
    /// Per-layer evaluations at `υ^(2^i)` (extension array, 3 limbs / 24 bytes each).
    pub evals_ptr: u64,
    /// Per-layer symmetric evaluations at `-υ^(2^i)` (extension array, stride 24).
    pub evals_sym_ptr: u64,
    /// Array of `num_layers` `(path_ptr: u64, path_len: u64)` pairs (16 bytes each)
    /// giving each layer's contiguous 32-byte auth-path sibling nodes.
    pub path_descs_ptr: u64,
    /// Output: single accept byte (`1` = every layer verified, `0` = some path
    /// failed).
    pub out_ptr: u64,
}

/// Number of `u64` fields in [`VerifyPathBatchInput`].
pub const VERIFY_PATH_BATCH_INPUT_FIELDS: usize = 7;

/// `SIM_CONSTRAINT_EVAL` v2 input — offload one table's OOD constraint
/// evaluation (`compute_transition`) to the host.
///
/// The host reconstructs the OOD frame from `frame_ptr` (the row-major
/// `height × width` extension grid the verifier built, split into main/aux at
/// `num_main`) + the LogUp challenge slices, then runs `eval_program_verifier`
/// against the constraint program the CLI preloaded at `seq_index` (the global
/// compute_transition sequence number), writing `num_constraints` per-constraint
/// extension evaluations to `out_ptr`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConstraintEvalInput {
    /// Global compute_transition sequence index; keys the preloaded program.
    pub seq_index: u64,
    /// OOD frame grid: `height × width` row-major extension elements (3 limbs each).
    pub frame_ptr: u64,
    pub width: u64,
    pub height: u64,
    /// Leading main-trace column count (the frame's main/aux split point).
    pub num_main: u64,
    pub step_size: u64,
    /// LogUp RAP challenges (extension array).
    pub rap_challenges_ptr: u64,
    pub rap_challenges_len: u64,
    /// Precomputed LogUp alpha powers (extension array).
    pub alpha_powers_ptr: u64,
    pub alpha_powers_len: u64,
    /// Pointer to the LogUp table offset `L/N` (one extension element).
    pub table_offset_ptr: u64,
    /// Number of transition constraints = output buffer length.
    pub num_constraints: u64,
    /// Output buffer for `num_constraints` extension evaluations.
    pub out_ptr: u64,
}

/// Number of `u64` fields in [`ConstraintEvalInput`].
pub const CONSTRAINT_EVAL_INPUT_FIELDS: usize = 13;

#[cfg(test)]
mod poly_eval_check {
    use crate::fft::bit_reversing::{in_place_bit_reverse_permute, reverse_index};
    use crate::field::element::FieldElement;
    use crate::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Fp3;
    use crate::field::goldilocks::GoldilocksField as Gl;
    use crate::field::traits::IsFFTField;
    use crate::polynomial::Polynomial;

    type F = FieldElement<Gl>;
    type E = FieldElement<Fp3>;

    #[test]
    fn per_point_horner_matches_offset_fft() {
        let num_coeffs = 128usize;
        let blowup = 4usize;
        let codeword_len = num_coeffs * blowup; // 512
        // Deterministic pseudo-random coeffs + offset.
        let coeffs: Vec<E> = (0..num_coeffs)
            .map(|k| {
                E::from_raw([
                    F::from(1234567u64 * (k as u64 + 1) + 7),
                    F::from(98765u64 * (k as u64 + 3) + 11),
                    F::from(555u64 * (k as u64 + 5) + 13),
                ])
            })
            .collect();
        let offset = F::from(0x1234_5678_9abcu64);

        // Reference: full coset FFT then bit-reverse to FRI order.
        let poly = Polynomial::new(&coeffs);
        let mut natural =
            Polynomial::evaluate_offset_fft::<Gl>(&poly, blowup, Some(num_coeffs), &offset)
                .unwrap();
        in_place_bit_reverse_permute(&mut natural);
        let reference = natural;

        // Mine: per FRI-order position, Horner at offset·ω^{reverse_index(p)}.
        let order = codeword_len.trailing_zeros() as u64;
        let omega = Gl::get_primitive_root_of_unity(order).unwrap();
        for (p, ref_p) in reference.iter().enumerate() {
            let natural_idx = reverse_index(p, codeword_len as u64);
            let x = &offset * omega.pow(natural_idx as u64);
            let mine = poly.evaluate(&x.to_extension::<Fp3>());
            assert_eq!(mine, *ref_p, "mismatch at FRI position {p}");
        }
    }
}
