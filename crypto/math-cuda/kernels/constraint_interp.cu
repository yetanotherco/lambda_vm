// Transition-constraint interpreter kernel.
//
// Evaluates a captured `ConstraintProgram` (lowered to the flat device blob by
// `crypto/stark/src/constraint_ir/device.rs`) over every row of a
// device-resident LDE. It is a transliteration of the CPU walker
// `eval_device_program` (same module), with `FieldElement` arithmetic replaced
// by `goldilocks.cuh` / `ext3.cuh` — the two are asserted bit-for-bit equal by
// the pre-GPU parity test, so this kernel's output equals the compiled prover
// folder.
//
// Design (v2, dim-split + liveness slots):
//   * One thread per LDE row, grid-stride over all rows (fixed launch, any size).
//   * The lowering assigns every node a slot in one of two per-thread scratch
//     classes — base (`u64`) or ext (`Fe3`) — with liveness reuse, so scratch
//     is sized by the program's max-live-set, not its node count. Slots are
//     strided by thread for coalescing: base slot `s` for this thread is
//     `vb[s * num_threads + tid]`; ext slot `s` keeps its three components at
//     `ve[(s*3 + k) * num_threads + tid]`.
//   * Operands are encoded as `kind << 29 | payload` (see `OPK_*`): a slot in
//     either class, or a direct reference into the tiny uniform tables
//     (constants, RAP challenges, alpha powers, table offset) — uniform leaves
//     never touch scratch.
//   * Base-dim arithmetic runs in the base field (1 mul vs 9 for ext3), and
//     mixed base×ext ops use shortcuts (`mul_base`, componentwise add/sub)
//     that are bit-identical to the full ext op on the embedded operand:
//     embedding is a ring homomorphism, `gl::add(x,0) == x == gl::sub(x,0)`,
//     and `dot3` with zero products reduces to `gl::mul`. Where an identity
//     is NOT guaranteed bitwise (negating an embedded zero limb), the full
//     form is kept (`gl::sub(0, y)`, never `gl::neg(y)`).
//
// Output is the per-constraint eval matrix `d_evals[c*num_rows + row]` (Fe3;
// base-rooted constraints carry their value in `.a`). The composition kernel
// below fuses the `z*Σ(Cᵢ·βᵢ) + boundary` accumulation instead, avoiding the
// matrix entirely.
//
// Op tags, operand kinds and the `res`/root packing MUST stay in sync with
// `crypto/stark/src/constraint_ir/device.rs`.

#include "goldilocks.cuh"
#include "ext3.cuh"

using ext3::Fe3;

// -- op tags (mirror device.rs OP_*) --
#define OP_CONST_BASE 0u
#define OP_CONST_EXT 1u
#define OP_VAR 2u
#define OP_RAP_CHALLENGE 3u
#define OP_ALPHA_POW 4u
#define OP_TABLE_OFFSET 5u
#define OP_ADD 6u
#define OP_SUB 7u
#define OP_MUL 8u
#define OP_NEG 9u
#define OP_EMBED 10u

// -- operand kinds (mirror device.rs OPK_*): enc = kind << 29 | payload --
#define OPK_SHIFT 29u
#define OPK_PAYLOAD_MASK 0x1FFFFFFFu
#define OPK_BASE_SLOT 0u
#define OPK_EXT_SLOT 1u
#define OPK_BASE_CONST 2u
#define OPK_EXT_CONST 3u
#define OPK_RAP 4u
#define OPK_ALPHA 5u
#define OPK_OFFSET 6u

// -- res / root packing: bit 31 = ext slot class, low bits = slot index --
#define RES_EXT_BIT 0x80000000u
#define RES_SLOT_MASK 0x7FFFFFFFu

// A flat IR node. Packed into two u64 words for a pure-u64 upload (matching the
// crate's device-buffer convention):
//   word0 = op | (a << 32) ; word1 = b | (res << 32)
// This mirrors the `#[repr(C)] DeviceNode { op, a, b, res: u32 }` payload.
struct Node {
    uint32_t op, a, b, res;
};

__device__ __forceinline__ Node load_node(const uint64_t *d_nodes, uint64_t i) {
    uint64_t w0 = d_nodes[2 * i];
    uint64_t w1 = d_nodes[2 * i + 1];
    Node n;
    n.op = (uint32_t)(w0 & 0xFFFFFFFFull);
    n.a = (uint32_t)(w0 >> 32);
    n.b = (uint32_t)(w1 & 0xFFFFFFFFull);
    n.res = (uint32_t)(w1 >> 32);
    return n;
}

// The per-proof uniform tables an operand can reference directly.
struct Uniforms {
    const uint64_t *base_consts;
    const Fe3 *ext_consts;
    const Fe3 *rap;
    const Fe3 *alpha;
    Fe3 offset;
};

// Whether an encoded operand holds a base-field value (slot or constant).
__device__ __forceinline__ bool opk_is_base(uint32_t enc) {
    uint32_t kind = enc >> OPK_SHIFT;
    return kind == OPK_BASE_SLOT || kind == OPK_BASE_CONST;
}

// Load a base-field operand (kind must be a base kind).
__device__ __forceinline__ uint64_t load_base_operand(uint32_t enc, const uint64_t *vb,
                                                      uint64_t vstride, const Uniforms &u) {
    uint32_t payload = enc & OPK_PAYLOAD_MASK;
    return (enc >> OPK_SHIFT) == OPK_BASE_SLOT ? vb[(uint64_t)payload * vstride]
                                               : u.base_consts[payload];
}

// Load any operand as ext3, embedding base values as {x, 0, 0}.
__device__ __forceinline__ Fe3 load_ext_operand(uint32_t enc, const uint64_t *vb, const uint64_t *ve,
                                                uint64_t vstride, const Uniforms &u) {
    uint32_t kind = enc >> OPK_SHIFT;
    uint32_t payload = enc & OPK_PAYLOAD_MASK;
    switch (kind) {
    case OPK_BASE_SLOT:
        return ext3::make(vb[(uint64_t)payload * vstride], 0, 0);
    case OPK_EXT_SLOT: {
        const uint64_t *p = ve + (uint64_t)payload * 3 * vstride;
        return ext3::make(p[0], p[vstride], p[2 * vstride]);
    }
    case OPK_BASE_CONST:
        return ext3::make(u.base_consts[payload], 0, 0);
    case OPK_EXT_CONST:
        return u.ext_consts[payload];
    case OPK_RAP:
        return u.rap[payload];
    case OPK_ALPHA:
        return u.alpha[payload];
    default: // OPK_OFFSET
        return u.offset;
    }
}

__device__ __forceinline__ void store_base_slot(uint64_t *vb, uint64_t vstride, uint32_t slot,
                                                uint64_t v) {
    vb[(uint64_t)slot * vstride] = v;
}

__device__ __forceinline__ void store_ext_slot(uint64_t *ve, uint64_t vstride, uint32_t slot,
                                               const Fe3 &v) {
    uint64_t *p = ve + (uint64_t)slot * 3 * vstride;
    p[0] = v.a;
    p[vstride] = v.b;
    p[2 * vstride] = v.c;
}

// Read a root value as ext3 (base roots embed as {x, 0, 0}).
__device__ __forceinline__ Fe3 load_root(uint64_t root_enc, const uint64_t *vb, const uint64_t *ve,
                                         uint64_t vstride) {
    uint32_t enc = (uint32_t)root_enc;
    uint32_t slot = enc & RES_SLOT_MASK;
    if (enc & RES_EXT_BIT) {
        const uint64_t *p = ve + (uint64_t)slot * 3 * vstride;
        return ext3::make(p[0], p[vstride], p[2 * vstride]);
    }
    return ext3::make(vb[(uint64_t)slot * vstride], 0, 0);
}

// Resolve an `Op::Var` leaf against the device-resident LDE columns.
//   a = col (low 16 bits); b = main<<16 | offset<<8 | row  (see device.rs pack_var)
// Base (main) columns are column-major `d_main[col*main_stride + r]`; ext (aux)
// columns store component k at `d_aux[(col*3 + k)*aux_stride + r]` (GpuLdeExt3).
// The frame `offset` selects row `r = (row + offset*next_step) mod num_rows`.
__device__ __forceinline__ uint64_t var_row(uint32_t b, uint64_t row, uint64_t next_step,
                                            uint64_t num_rows) {
    uint32_t offset = (b >> 8) & 0xFFu;
    uint64_t r = row + (uint64_t)offset * next_step;
    if (r >= num_rows) {
        r -= num_rows; // wrap; offset*next_step < num_rows by construction
    }
    return r;
}

// Shared forward pass: evaluate every IR node of the program for one LDE row
// into the per-thread slot scratch. The single home of the op semantics — both
// kernels below run this exact walk, so an op change stays in lockstep with
// `constraint_ir/device.rs` in one place.
//
// Every mixed-op shortcut below must be bit-identical to the full ext3 op on
// the embedded operand; see the file header for the argument.
__device__ __forceinline__ void eval_program_row(
    uint64_t *vb, uint64_t *ve, uint64_t vstride, const uint64_t *d_nodes, uint64_t num_nodes,
    const Uniforms &u, uint64_t row, uint64_t next_step, uint64_t num_rows,
    const uint64_t *d_main, uint64_t main_stride, const uint64_t *d_aux, uint64_t aux_stride) {
    for (uint64_t i = 0; i < num_nodes; i++) {
        Node nd = load_node(d_nodes, i);
        uint32_t slot = nd.res & RES_SLOT_MASK;
        bool res_ext = (nd.res & RES_EXT_BIT) != 0;
        switch (nd.op) {
        case OP_VAR: {
            uint64_t r = var_row(nd.b, row, next_step, num_rows);
            uint32_t col = nd.a & 0xFFFFu;
            bool is_main = ((nd.b >> 16) & 1u) != 0u;
            if (is_main) {
                store_base_slot(vb, vstride, slot, d_main[(uint64_t)col * main_stride + r]);
            } else {
                uint64_t base = (uint64_t)col * 3;
                store_ext_slot(ve, vstride, slot,
                               ext3::make(d_aux[(base + 0) * aux_stride + r],
                                          d_aux[(base + 1) * aux_stride + r],
                                          d_aux[(base + 2) * aux_stride + r]));
            }
            break;
        }
        case OP_ADD: {
            if (!res_ext) {
                store_base_slot(vb, vstride, slot,
                                goldilocks::add(load_base_operand(nd.a, vb, vstride, u),
                                                load_base_operand(nd.b, vb, vstride, u)));
            } else if (opk_is_base(nd.a)) {
                // {x,0,0} + y = {add(x,y.a), y.b, y.c} (add(0,v) == v).
                uint64_t x = load_base_operand(nd.a, vb, vstride, u);
                Fe3 y = load_ext_operand(nd.b, vb, ve, vstride, u);
                store_ext_slot(ve, vstride, slot, ext3::make(goldilocks::add(x, y.a), y.b, y.c));
            } else if (opk_is_base(nd.b)) {
                Fe3 x = load_ext_operand(nd.a, vb, ve, vstride, u);
                uint64_t y = load_base_operand(nd.b, vb, vstride, u);
                store_ext_slot(ve, vstride, slot, ext3::make(goldilocks::add(x.a, y), x.b, x.c));
            } else {
                store_ext_slot(ve, vstride, slot,
                               ext3::add(load_ext_operand(nd.a, vb, ve, vstride, u),
                                         load_ext_operand(nd.b, vb, ve, vstride, u)));
            }
            break;
        }
        case OP_SUB: {
            if (!res_ext) {
                store_base_slot(vb, vstride, slot,
                                goldilocks::sub(load_base_operand(nd.a, vb, vstride, u),
                                                load_base_operand(nd.b, vb, vstride, u)));
            } else if (opk_is_base(nd.a)) {
                // {x,0,0} - y = {sub(x,y.a), sub(0,y.b), sub(0,y.c)}; sub(0,·)
                // is kept literal — it is NOT bitwise `neg` on non-canonical
                // limbs.
                uint64_t x = load_base_operand(nd.a, vb, vstride, u);
                Fe3 y = load_ext_operand(nd.b, vb, ve, vstride, u);
                store_ext_slot(ve, vstride, slot,
                               ext3::make(goldilocks::sub(x, y.a), goldilocks::sub(0, y.b),
                                          goldilocks::sub(0, y.c)));
            } else if (opk_is_base(nd.b)) {
                // x - {y,0,0} = {sub(x.a,y), x.b, x.c} (sub(v,0) == v).
                Fe3 x = load_ext_operand(nd.a, vb, ve, vstride, u);
                uint64_t y = load_base_operand(nd.b, vb, vstride, u);
                store_ext_slot(ve, vstride, slot, ext3::make(goldilocks::sub(x.a, y), x.b, x.c));
            } else {
                store_ext_slot(ve, vstride, slot,
                               ext3::sub(load_ext_operand(nd.a, vb, ve, vstride, u),
                                         load_ext_operand(nd.b, vb, ve, vstride, u)));
            }
            break;
        }
        case OP_MUL: {
            if (!res_ext) {
                store_base_slot(vb, vstride, slot,
                                goldilocks::mul(load_base_operand(nd.a, vb, vstride, u),
                                                load_base_operand(nd.b, vb, vstride, u)));
            } else if (opk_is_base(nd.a)) {
                // {x,0,0} * y = mul_base(y, x): dot3 with zero products
                // reduces to gl::mul exactly.
                uint64_t x = load_base_operand(nd.a, vb, vstride, u);
                Fe3 y = load_ext_operand(nd.b, vb, ve, vstride, u);
                store_ext_slot(ve, vstride, slot, ext3::mul_base(y, x));
            } else if (opk_is_base(nd.b)) {
                Fe3 x = load_ext_operand(nd.a, vb, ve, vstride, u);
                uint64_t y = load_base_operand(nd.b, vb, vstride, u);
                store_ext_slot(ve, vstride, slot, ext3::mul_base(x, y));
            } else {
                store_ext_slot(ve, vstride, slot,
                               ext3::mul(load_ext_operand(nd.a, vb, ve, vstride, u),
                                         load_ext_operand(nd.b, vb, ve, vstride, u)));
            }
            break;
        }
        case OP_NEG: {
            if (!res_ext) {
                store_base_slot(vb, vstride, slot,
                                goldilocks::neg(load_base_operand(nd.a, vb, vstride, u)));
            } else {
                store_ext_slot(ve, vstride, slot,
                               ext3::neg(load_ext_operand(nd.a, vb, ve, vstride, u)));
            }
            break;
        }
        case OP_EMBED: {
            store_ext_slot(ve, vstride, slot, load_ext_operand(nd.a, vb, ve, vstride, u));
            break;
        }
        // Uniform leaves materialize only when they are constraint roots.
        case OP_CONST_BASE:
            store_base_slot(vb, vstride, slot, u.base_consts[nd.a]);
            break;
        case OP_CONST_EXT:
            store_ext_slot(ve, vstride, slot, u.ext_consts[nd.a]);
            break;
        case OP_RAP_CHALLENGE:
            store_ext_slot(ve, vstride, slot, u.rap[nd.a]);
            break;
        case OP_ALPHA_POW:
            store_ext_slot(ve, vstride, slot, u.alpha[nd.a]);
            break;
        case OP_TABLE_OFFSET:
            store_ext_slot(ve, vstride, slot, u.offset);
            break;
        default:
            break;
        }
    }
}

extern "C" __global__ void constraint_interp_kernel(
    // output: per-constraint eval matrix, constraint-major [num_roots * num_rows]
    Fe3 *__restrict__ d_evals,
    // program (flat blob)
    const uint64_t *__restrict__ d_nodes, // 2 u64 per node
    uint64_t num_nodes,
    const uint64_t *__restrict__ d_base_consts,
    const Fe3 *__restrict__ d_ext_consts,
    const uint64_t *__restrict__ d_roots, // slot | ext_bit<<31, one per constraint
    uint64_t num_roots,
    // per-proof uniforms
    const Fe3 *__restrict__ d_rap_challenges,
    const Fe3 *__restrict__ d_alpha_powers,
    const Fe3 *__restrict__ d_table_offset, // single element
    // device-resident LDE
    const uint64_t *__restrict__ d_main,
    uint64_t main_stride,
    const uint64_t *__restrict__ d_aux,
    uint64_t aux_stride,
    uint64_t next_step,
    // sizing
    uint64_t num_rows,
    // scratch: per-thread slot files, [num_base_slots * num_threads] and
    // [num_ext_slots * 3 * num_threads]
    uint64_t *__restrict__ d_vals_base,
    uint64_t *__restrict__ d_vals_ext) {
    uint64_t task_offset = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * blockDim.x;

    uint64_t *vb = d_vals_base + task_offset;
    uint64_t *ve = d_vals_ext + task_offset;
    uint64_t vstride = num_threads;

    Uniforms u;
    u.base_consts = d_base_consts;
    u.ext_consts = d_ext_consts;
    u.rap = d_rap_challenges;
    u.alpha = d_alpha_powers;
    u.offset = *d_table_offset;

    for (uint64_t row = task_offset; row < num_rows; row += num_threads) {
        eval_program_row(vb, ve, vstride, d_nodes, num_nodes, u, row, next_step, num_rows, d_main,
                         main_stride, d_aux, aux_stride);

        // Emit each constraint root (base roots embed as {x, 0, 0}).
        for (uint64_t c = 0; c < num_roots; c++) {
            d_evals[c * num_rows + row] = load_root(d_roots[c], vb, ve, vstride);
        }
    }
}

// Fused composition-polynomial kernel: same node walk as
// `constraint_interp_kernel`, but instead of emitting the per-constraint matrix
// it accumulates the composition-poly evaluation H(row) on-device — no matrix
// materialization, no D2H. Mirrors the CPU accumulation in
// `crypto/stark/src/constraints/evaluator.rs` (uniform-zerofier case):
//
//   H(row) = z_inv[row % z_len] * Σ_c beta_trans[c] * C_c(row)          (transition)
//          + Σ_b z_b_inv[b*num_rows + row] * beta_bnd[b] * (trace_b - value_b)
//
// where a base-rooted C_c contributes via `mul_base` (bit-identical to the
// full mul on its embedding), z_inv is the cyclic base transition-zerofier
// inverse, and the boundary term reads the resident trace at column `b_col[b]`
// (main or aux).
extern "C" __global__ void constraint_composition_kernel(
    // output: one H(row) per LDE row
    Fe3 *__restrict__ d_h,
    // program (flat blob) — identical to the interpreter kernel
    const uint64_t *__restrict__ d_nodes,
    uint64_t num_nodes,
    const uint64_t *__restrict__ d_base_consts,
    const Fe3 *__restrict__ d_ext_consts,
    const uint64_t *__restrict__ d_roots,
    uint64_t num_roots,
    // per-proof uniforms
    const Fe3 *__restrict__ d_rap_challenges,
    const Fe3 *__restrict__ d_alpha_powers,
    const Fe3 *__restrict__ d_table_offset,
    // device-resident LDE
    const uint64_t *__restrict__ d_main,
    uint64_t main_stride,
    const uint64_t *__restrict__ d_aux,
    uint64_t aux_stride,
    uint64_t next_step,
    uint64_t num_rows,
    // transition accumulation
    const Fe3 *__restrict__ d_beta_trans, // [num_roots]
    const uint64_t *__restrict__ d_z_inv, // [z_len], cyclic
    uint64_t z_len,
    // boundary accumulation
    uint64_t num_boundary,
    const uint64_t *__restrict__ d_b_col,    // [num_boundary]
    const uint64_t *__restrict__ d_b_is_aux, // [num_boundary] (0/1)
    const Fe3 *__restrict__ d_b_value,       // [num_boundary]
    const Fe3 *__restrict__ d_b_beta,        // [num_boundary]
    const uint64_t *__restrict__ d_b_z_inv,  // [num_boundary * num_rows]
    // scratch: per-thread slot files
    uint64_t *__restrict__ d_vals_base,
    uint64_t *__restrict__ d_vals_ext) {
    uint64_t task_offset = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * blockDim.x;

    uint64_t *vb = d_vals_base + task_offset;
    uint64_t *ve = d_vals_ext + task_offset;
    uint64_t vstride = num_threads;

    Uniforms u;
    u.base_consts = d_base_consts;
    u.ext_consts = d_ext_consts;
    u.rap = d_rap_challenges;
    u.alpha = d_alpha_powers;
    u.offset = *d_table_offset;

    for (uint64_t row = task_offset; row < num_rows; row += num_threads) {
        eval_program_row(vb, ve, vstride, d_nodes, num_nodes, u, row, next_step, num_rows, d_main,
                         main_stride, d_aux, aux_stride);

        // Transition: z_inv * Σ_c beta_c * C_c. Base roots use mul_base —
        // bit-identical to mul(beta, {v,0,0}).
        Fe3 sum = ext3::zero();
        for (uint64_t c = 0; c < num_roots; c++) {
            uint32_t enc = (uint32_t)d_roots[c];
            uint32_t slot = enc & RES_SLOT_MASK;
            if (enc & RES_EXT_BIT) {
                const uint64_t *p = ve + (uint64_t)slot * 3 * vstride;
                Fe3 cval = ext3::make(p[0], p[vstride], p[2 * vstride]);
                sum = ext3::add(sum, ext3::mul(d_beta_trans[c], cval));
            } else {
                sum = ext3::add(sum, ext3::mul_base(d_beta_trans[c], vb[(uint64_t)slot * vstride]));
            }
        }
        Fe3 h = ext3::mul_base(sum, d_z_inv[row % z_len]);

        // Boundary: Σ_b z_b_inv[row] * beta_b * (trace[col_b] - value_b).
        for (uint64_t b = 0; b < num_boundary; b++) {
            uint64_t col = d_b_col[b];
            Fe3 tcell;
            if (d_b_is_aux[b] != 0) {
                uint64_t base = col * 3;
                tcell = ext3::make(d_aux[(base + 0) * aux_stride + row],
                                   d_aux[(base + 1) * aux_stride + row],
                                   d_aux[(base + 2) * aux_stride + row]);
            } else {
                tcell = ext3::make(d_main[col * main_stride + row], 0, 0);
            }
            Fe3 bp = ext3::sub(tcell, d_b_value[b]);
            // (z_b_inv * beta_b) * bp — matches the CPU op order.
            Fe3 zb = ext3::mul_base(d_b_beta[b], d_b_z_inv[b * num_rows + row]);
            h = ext3::add(h, ext3::mul(zb, bp));
        }

        d_h[row] = h;
    }
}

// ============================================================================
// Degree-2 quotient decomposition, pointwise on the LDE coset:
//   H0[i] = two_inv  * (h[i] + h[i+n])
//   H1[i] = inv_2x[i] * (h[i] - h[i+n])
// Reads the interleaved ext3 composition evals `h` (2n rows); writes the two
// halves in slab layout (3 base slabs per half, `slab_stride` u64 each; rows
// n.. stay zero as the LDE zero-pad).
extern "C" __global__ void decompose_d2_ext3(
    const uint64_t *__restrict__ h,
    const uint64_t *__restrict__ inv_2x,
    uint64_t two_inv,
    uint64_t n,
    uint64_t slab_stride,
    uint64_t *__restrict__ out) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        Fe3 x = ext3::make(h[i * 3], h[i * 3 + 1], h[i * 3 + 2]);
        Fe3 y = ext3::make(h[(i + n) * 3], h[(i + n) * 3 + 1], h[(i + n) * 3 + 2]);
        Fe3 h0 = ext3::mul_base(ext3::add(x, y), two_inv);
        Fe3 h1 = ext3::mul_base(ext3::sub(x, y), inv_2x[i]);
        out[0 * slab_stride + i] = h0.a;
        out[1 * slab_stride + i] = h0.b;
        out[2 * slab_stride + i] = h0.c;
        out[3 * slab_stride + i] = h1.a;
        out[4 * slab_stride + i] = h1.b;
        out[5 * slab_stride + i] = h1.c;
    }
}

// ============================================================================
// Degree-1 (num_parts==1) composition part: H IS the single part, already on
// the LDE coset, so there is no decompose and no re-extension. Only de-interleave
// the resident ext3 composition evals `h` (num_rows rows, interleaved
// `h[row*3 + k]`) into the 3-slab layout the commit / DEEP / FRI consumers
// expect (`out[k*num_rows + row]`).
extern "C" __global__ void comp_h_to_slabs_ext3(
    const uint64_t *__restrict__ h,
    uint64_t num_rows,
    uint64_t *__restrict__ out) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
         i += (uint64_t)gridDim.x * blockDim.x) {
        out[0 * num_rows + i] = h[i * 3];
        out[1 * num_rows + i] = h[i * 3 + 1];
        out[2 * num_rows + i] = h[i * 3 + 2];
    }
}
