//! GPU dispatch for `compute_logup_batched_term_column`: takes a pair of
//! `BusInteraction`s and evaluates the full fingerprint + batch-invert +
//! term-assembly pipeline on the device.
//!
//! The serializer lives here (on the stark side) so it can see the
//! `BusValue` / `Multiplicity` types without creating a math ↔ stark
//! dependency cycle. The matching kernels are in
//! `crypto/math-cuda/kernels/logup.cu`.
//!
//! Only Goldilocks main trace + Fp3 extension is supported — everything
//! else returns `None` and the caller runs the CPU path.
//!
//! Canonicalization contract: all coefficients the GPU sees are already
//! canonical Goldilocks field elements in `[0, p)`. The kernels do not
//! sign-handle; they treat every `value` as a plain u64 coefficient.

use core::any::type_name;

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;
use math::field::traits::{IsField, IsSubFieldOf};
use math_cuda::logup::{
    self, FingerprintOp, LinearTerm as GpuLinearTerm, MultiplicityDesc,
};

use crate::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};

/// Goldilocks modulus p = 2^64 - 2^32 + 1. Used for the i64/u64 → canonical
/// u64 reduction so the kernel never sees a non-canonical coefficient.
const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

#[inline(always)]
fn canonical_u64(v: u64) -> u64 {
    if v >= GOLDILOCKS_P { v - GOLDILOCKS_P } else { v }
}

#[inline(always)]
fn canonical_i64(v: i64) -> u64 {
    if v >= 0 {
        canonical_u64(v as u64)
    } else {
        // -|v| mod p = p - (|v| mod p). |v| fits in u64 since i64 only
        // reaches -2^63 < 2^64, and (-v) as u64 handles that cleanly.
        let abs = v.unsigned_abs();
        let c = canonical_u64(abs);
        if c == 0 { 0 } else { GOLDILOCKS_P - c }
    }
}

/// Serializer output — one call's full bytecode bundle. Owned so the host
/// wrapper can upload contiguous slices without extra copies.
pub struct PairBytecode {
    pub ops_a: Vec<FingerprintOp>,
    pub ops_b: Vec<FingerprintOp>,
    pub linear_terms: Vec<GpuLinearTerm>,
    pub mult_a: MultiplicityDesc,
    pub mult_b: MultiplicityDesc,
    pub bus_id_a: u64,
    pub bus_id_b: u64,
    pub negate_a: bool,
    pub negate_b: bool,
    pub max_bus_elements: usize,
}

/// Translate one interaction's `values` into a list of `FingerprintOp`s,
/// appending any referenced LinearTerms to `pool`. `alpha_offset_start`
/// should be 1 (slot 0 is reserved for bus_id * alpha[0]).
fn encode_bus_values(
    values: &[BusValue],
    alpha_offset_start: usize,
    pool: &mut Vec<GpuLinearTerm>,
) -> Vec<FingerprintOp> {
    let mut ops = Vec::with_capacity(values.len());
    let mut alpha_offset = alpha_offset_start as u32;
    for bv in values {
        match bv {
            BusValue::Packed { start_column, packing } => {
                let kind = packing_to_op_kind(*packing);
                let consumed = packing.num_bus_elements() as u32;
                ops.push(FingerprintOp {
                    kind,
                    pad0: [0; 3],
                    alpha_offset,
                    start_col: *start_column as u32,
                    num_linear_terms: 0,
                    linear_term_offset: 0,
                    pad1: [0; 2],
                });
                alpha_offset += consumed;
            }
            BusValue::Linear(terms) => {
                let offset = pool.len() as u32;
                for t in terms {
                    pool.push(lower_linear_term(t));
                }
                ops.push(FingerprintOp {
                    kind: logup::OP_LINEAR,
                    pad0: [0; 3],
                    alpha_offset,
                    start_col: 0,
                    num_linear_terms: terms.len() as u32,
                    linear_term_offset: offset,
                    pad1: [0; 2],
                });
                alpha_offset += 1;
            }
        }
    }
    ops
}

fn packing_to_op_kind(p: Packing) -> u8 {
    match p {
        Packing::Direct => logup::OP_PACK_DIRECT,
        Packing::Word2L => logup::OP_PACK_WORD2L,
        Packing::Word4L => logup::OP_PACK_WORD4L,
        Packing::DWordWL => logup::OP_PACK_DWORDWL,
        Packing::DWordHHW => logup::OP_PACK_DWORDHHW,
        Packing::DWordWHH => logup::OP_PACK_DWORDWHH,
        Packing::DWordHL => logup::OP_PACK_DWORDHL,
        Packing::DWordBL => logup::OP_PACK_DWORDBL,
        Packing::QuadHL => logup::OP_PACK_QUADHL,
        Packing::QuadWL => logup::OP_PACK_QUADWL,
    }
}

fn lower_linear_term(t: &LinearTerm) -> GpuLinearTerm {
    match *t {
        LinearTerm::Column { coefficient, column } => GpuLinearTerm {
            kind: logup::LT_KIND_COLUMN,
            pad: [0; 3],
            column: column as u32,
            value: canonical_i64(coefficient),
        },
        LinearTerm::ColumnUnsigned { coefficient, column } => GpuLinearTerm {
            kind: logup::LT_KIND_COLUMN,
            pad: [0; 3],
            column: column as u32,
            value: canonical_u64(coefficient),
        },
        LinearTerm::Constant(value) => GpuLinearTerm {
            kind: logup::LT_KIND_CONSTANT,
            pad: [0; 3],
            column: 0,
            value: canonical_i64(value),
        },
    }
}

fn encode_multiplicity(
    m: &Multiplicity,
    pool: &mut Vec<GpuLinearTerm>,
) -> MultiplicityDesc {
    match m {
        Multiplicity::One => MultiplicityDesc {
            kind: logup::MULT_ONE,
            ..Default::default()
        },
        Multiplicity::Column(c) => MultiplicityDesc {
            kind: logup::MULT_COLUMN,
            cols: [*c as u32, 0, 0],
            ..Default::default()
        },
        Multiplicity::Sum(a, b) => MultiplicityDesc {
            kind: logup::MULT_SUM,
            cols: [*a as u32, *b as u32, 0],
            ..Default::default()
        },
        Multiplicity::Negated(c) => MultiplicityDesc {
            kind: logup::MULT_NEGATED,
            cols: [*c as u32, 0, 0],
            ..Default::default()
        },
        Multiplicity::Diff(a, b) => MultiplicityDesc {
            kind: logup::MULT_DIFF,
            cols: [*a as u32, *b as u32, 0],
            ..Default::default()
        },
        Multiplicity::Sum3(a, b, c) => MultiplicityDesc {
            kind: logup::MULT_SUM3,
            cols: [*a as u32, *b as u32, *c as u32],
            ..Default::default()
        },
        Multiplicity::Linear(terms) => {
            let offset = pool.len() as u32;
            for t in terms {
                pool.push(lower_linear_term(t));
            }
            MultiplicityDesc {
                kind: logup::MULT_LINEAR,
                cols: [0; 3],
                num_linear_terms: terms.len() as u32,
                linear_term_offset: offset,
                ..Default::default()
            }
        }
    }
}

/// Serialize a pair of interactions into the shared bytecode form.
pub fn build_pair_bytecode(
    interaction_a: &BusInteraction,
    interaction_b: &BusInteraction,
) -> PairBytecode {
    let mut linear_terms: Vec<GpuLinearTerm> = Vec::new();
    let ops_a = encode_bus_values(&interaction_a.values, 1, &mut linear_terms);
    let ops_b = encode_bus_values(&interaction_b.values, 1, &mut linear_terms);
    let mult_a = encode_multiplicity(&interaction_a.multiplicity, &mut linear_terms);
    let mult_b = encode_multiplicity(&interaction_b.multiplicity, &mut linear_terms);
    let max_bus_elements = interaction_a
        .num_bus_elements()
        .max(interaction_b.num_bus_elements());
    PairBytecode {
        ops_a,
        ops_b,
        linear_terms,
        mult_a,
        mult_b,
        bus_id_a: interaction_a.bus_id,
        bus_id_b: interaction_b.bus_id,
        negate_a: !interaction_a.is_sender,
        negate_b: !interaction_b.is_sender,
        max_bus_elements,
    }
}

/// Flatten `main_segment_cols` into column-major u64. SAFETY: the caller
/// must have verified that `F == GoldilocksField` so that each
/// `FieldElement<F>` is representationally a single u64.
unsafe fn flatten_main_cols<F>(main_segment_cols: &[Vec<FieldElement<F>>]) -> Vec<u64>
where
    F: IsField,
{
    if main_segment_cols.is_empty() {
        return Vec::new();
    }
    let n = main_segment_cols[0].len();
    let num_cols = main_segment_cols.len();
    let mut out = Vec::with_capacity(num_cols * n);
    for col in main_segment_cols {
        debug_assert_eq!(col.len(), n);
        for e in col {
            out.push(unsafe { *(e.value() as *const _ as *const u64) });
        }
    }
    out
}

/// Convert a Fp3 FieldElement to its raw `[u64; 3]` ext3 triple. The kernel
/// tolerates non-canonical inputs (it partial-reduces), so we skip the
/// extra canonicalization step and read raw u64 bits.
/// SAFETY: the caller must have verified `E == Degree3GoldilocksExtensionField`.
unsafe fn ext3_to_triple<E: IsField>(e: &FieldElement<E>) -> [u64; 3] {
    let ptr = e.value() as *const _ as *const [FieldElement<GoldilocksField>; 3];
    let triple = unsafe { &*ptr };
    [
        unsafe { *(triple[0].value() as *const _ as *const u64) },
        unsafe { *(triple[1].value() as *const _ as *const u64) },
        unsafe { *(triple[2].value() as *const _ as *const u64) },
    ]
}

/// Per-pair GPU-vs-CPU threshold on `trace_len`. Below this, the per-pair
/// overhead (main-cols H2D + kernel launches + 2n×3 D2H) dominates and the
/// rayon-parallel CPU path wins. Set conservatively; override via env var
/// for experiments.
const DEFAULT_GPU_LOGUP_THRESHOLD: usize = usize::MAX;

fn gpu_logup_threshold() -> usize {
    static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("LAMBDA_VM_GPU_LOGUP_THRESHOLD")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_GPU_LOGUP_THRESHOLD)
    })
}

static GPU_LOGUP_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn gpu_logup_calls() -> u64 {
    GPU_LOGUP_CALLS.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn reset_gpu_logup_calls() {
    GPU_LOGUP_CALLS.store(0, std::sync::atomic::Ordering::Relaxed);
}

#[inline]
fn gpu_supported<F, E>(trace_len: usize) -> bool {
    if type_name::<F>() != type_name::<GoldilocksField>() {
        return false;
    }
    if type_name::<E>() != type_name::<Degree3GoldilocksExtensionField>() {
        return false;
    }
    trace_len >= gpu_logup_threshold()
}

/// Batch-compute all committed pair term columns (and optionally the
/// absorbed virtual pair) on GPU for one table. Uploads main_cols exactly
/// once per table — this is the win vs. per-pair dispatch.
///
/// Returns `None` if the F/E type combination isn't supported; caller
/// falls back to the rayon CPU path entirely.
pub fn try_compute_table_term_columns<F, E>(
    interactions: &[BusInteraction],
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> Option<TableTermColumns<E>>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    if !gpu_supported::<F, E>(trace_len) {
        return None;
    }

    let (num_committed_pairs, absorbed_count) =
        crate::lookup::split_interactions(interactions.len());

    // Upload main_cols ONCE.
    let main_cols_u64 = unsafe { flatten_main_cols(main_segment_cols) };
    let device_main =
        math_cuda::logup::upload_main_cols(&main_cols_u64, main_segment_cols.len(), trace_len)
            .ok()?;

    let alpha = &challenges[crate::lookup::LOGUP_CHALLENGE_ALPHA];
    let z = &challenges[0];
    let z_triple = unsafe { ext3_to_triple(z) };

    let mut committed = Vec::with_capacity(num_committed_pairs);
    for i in 0..num_committed_pairs {
        let a = &interactions[i * 2];
        let b = &interactions[i * 2 + 1];
        let bytecode = build_pair_bytecode(a, b);
        let alpha_powers_fe =
            crate::lookup::compute_alpha_powers(alpha, bytecode.max_bus_elements);
        let mut alpha_powers_u64 = Vec::with_capacity(bytecode.max_bus_elements * 3);
        for ap in &alpha_powers_fe {
            let t = unsafe { ext3_to_triple(ap) };
            alpha_powers_u64.extend_from_slice(&t);
        }
        let result = math_cuda::logup::logup_pair_term_column_on_device(
            &device_main,
            bytecode.bus_id_a,
            bytecode.bus_id_b,
            &bytecode.ops_a,
            &bytecode.ops_b,
            &bytecode.linear_terms,
            &alpha_powers_u64,
            &z_triple,
            &bytecode.mult_a,
            &bytecode.mult_b,
            bytecode.negate_a,
            bytecode.negate_b,
        )
        .ok()?;
        GPU_LOGUP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        committed.push(triples_to_ext3_fieldelements::<E>(&result, trace_len));
    }

    // Virtual column (for absorbed interactions).
    let virtual_col = match absorbed_count {
        0 => None,
        2 => {
            let a = &interactions[interactions.len() - 2];
            let b = &interactions[interactions.len() - 1];
            let bytecode = build_pair_bytecode(a, b);
            let alpha_powers_fe =
                crate::lookup::compute_alpha_powers(alpha, bytecode.max_bus_elements);
            let mut alpha_powers_u64 = Vec::with_capacity(bytecode.max_bus_elements * 3);
            for ap in &alpha_powers_fe {
                let t = unsafe { ext3_to_triple(ap) };
                alpha_powers_u64.extend_from_slice(&t);
            }
            let result = math_cuda::logup::logup_pair_term_column_on_device(
                &device_main,
                bytecode.bus_id_a,
                bytecode.bus_id_b,
                &bytecode.ops_a,
                &bytecode.ops_b,
                &bytecode.linear_terms,
                &alpha_powers_u64,
                &z_triple,
                &bytecode.mult_a,
                &bytecode.mult_b,
                bytecode.negate_a,
                bytecode.negate_b,
            )
            .ok()?;
            GPU_LOGUP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(triples_to_ext3_fieldelements::<E>(&result, trace_len))
        }
        1 => {
            let a = &interactions[interactions.len() - 1];
            let mut pool: Vec<math_cuda::logup::LinearTerm> = Vec::new();
            let ops = encode_bus_values(&a.values, 1, &mut pool);
            let mult = encode_multiplicity(&a.multiplicity, &mut pool);
            let max_bus_elements = a.num_bus_elements();
            let alpha_powers_fe = crate::lookup::compute_alpha_powers(alpha, max_bus_elements);
            let mut alpha_powers_u64 = Vec::with_capacity(max_bus_elements * 3);
            for ap in &alpha_powers_fe {
                let t = unsafe { ext3_to_triple(ap) };
                alpha_powers_u64.extend_from_slice(&t);
            }
            let result = math_cuda::logup::logup_single_term_column_on_device(
                &device_main,
                a.bus_id,
                &ops,
                &pool,
                &alpha_powers_u64,
                &z_triple,
                &mult,
                !a.is_sender,
            )
            .ok()?;
            GPU_LOGUP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(triples_to_ext3_fieldelements::<E>(&result, trace_len))
        }
        _ => unreachable!("absorbed_count must be 0, 1, or 2"),
    };

    Some(TableTermColumns {
        committed,
        virtual_col,
    })
}

pub struct TableTermColumns<E: IsField> {
    pub committed: Vec<Vec<FieldElement<E>>>,
    pub virtual_col: Option<Vec<FieldElement<E>>>,
}

/// Try to run the pair on the GPU. Returns `Some(term_column)` on success
/// (3 * trace_len u64s flattened into FieldElement<E>) or `None` if the
/// type combination isn't supported — in which case the caller falls
/// back to the CPU path.
pub fn try_compute_pair_term_column<F, E>(
    interaction_a: &BusInteraction,
    interaction_b: &BusInteraction,
    main_segment_cols: &[Vec<FieldElement<F>>],
    trace_len: usize,
    challenges: &[FieldElement<E>],
) -> Option<Vec<FieldElement<E>>>
where
    F: IsField + IsSubFieldOf<E>,
    E: IsField,
{
    if !gpu_supported::<F, E>(trace_len) {
        return None;
    }

    // Compute alpha_powers (ext3 extension). Fallback on CPU side (cheap,
    // runs once per pair, O(max_bus_elements) multiplications).
    let alpha = &challenges[crate::lookup::LOGUP_CHALLENGE_ALPHA];
    let z = &challenges[0];

    let bytecode = build_pair_bytecode(interaction_a, interaction_b);
    let alpha_powers_fe = crate::lookup::compute_alpha_powers(alpha, bytecode.max_bus_elements);

    // Extract u64 views.
    let main_cols_u64 = unsafe { flatten_main_cols(main_segment_cols) };
    let mut alpha_powers_u64 = Vec::with_capacity(bytecode.max_bus_elements * 3);
    for ap in &alpha_powers_fe {
        let t = unsafe { ext3_to_triple(ap) };
        alpha_powers_u64.extend_from_slice(&t);
    }
    let z_triple = unsafe { ext3_to_triple(z) };

    let result = logup::logup_pair_term_column(
        &main_cols_u64,
        main_segment_cols.len(),
        trace_len,
        bytecode.bus_id_a,
        bytecode.bus_id_b,
        &bytecode.ops_a,
        &bytecode.ops_b,
        &bytecode.linear_terms,
        &alpha_powers_u64,
        &z_triple,
        &bytecode.mult_a,
        &bytecode.mult_b,
        bytecode.negate_a,
        bytecode.negate_b,
    )
    .ok()?;

    GPU_LOGUP_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(triples_to_ext3_fieldelements::<E>(&result, trace_len))
}

/// Reassemble `trace_len` ext3 triples back into `FieldElement<E>`.
/// SAFETY: caller must have verified E == Degree3GoldilocksExtensionField.
fn triples_to_ext3_fieldelements<E: IsField>(
    data: &[u64],
    trace_len: usize,
) -> Vec<FieldElement<E>> {
    assert_eq!(data.len(), trace_len * 3);
    let mut out = Vec::with_capacity(trace_len);
    for i in 0..trace_len {
        let triple: [FieldElement<GoldilocksField>; 3] = [
            FieldElement::<GoldilocksField>::from(data[i * 3]),
            FieldElement::<GoldilocksField>::from(data[i * 3 + 1]),
            FieldElement::<GoldilocksField>::from(data[i * 3 + 2]),
        ];
        // SAFETY: type_name check at the entry point guarantees E is
        // Degree3GoldilocksExtensionField, whose BaseType = [FpE; 3].
        let raw: <E as IsField>::BaseType = unsafe { core::mem::transmute_copy(&triple) };
        out.push(FieldElement::<E>::from_raw(raw));
    }
    out
}
