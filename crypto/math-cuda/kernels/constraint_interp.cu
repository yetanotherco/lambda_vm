// Transition-constraint interpreter kernel.
//
// Evaluates a captured `ConstraintProgram` (lowered to the flat device blob by
// `crypto/stark/src/constraint_ir/device.rs`) over every row of a
// device-resident LDE, producing the per-constraint evaluations. It is a
// transliteration of the CPU walker `eval_device_program` (same module), with
// `FieldElement` arithmetic replaced by `goldilocks.cuh` / `ext3.cuh` — the two
// are asserted bit-for-bit equal by the pre-GPU parity test, so this kernel's
// output equals the compiled prover folder.
//
// Design (v2, dim-split — the "Phase 6" follow-on of the all-ext v1):
//   * One thread per LDE row, grid-stride over all rows (fixed launch, any size).
//   * Per-thread value scratch in GLOBAL memory, strided by thread for
//     coalescing — and PACKED by dimension: a DIM_BASE node owns 1 u64 slot, a
//     DIM_EXT node owns 3. Node `i`'s slots for this thread start at
//     `d_values[task_offset + d_val_offsets[i]*num_threads]` (lane `k` at
//     `+ (d_val_offsets[i]+k)*num_threads`). `d_val_offsets` (`num_nodes + 1`
//     prefix sums, built by `DeviceProgram::lower`) also gives any operand's
//     width as `offsets[id+1] - offsets[id]`.
//   * DIM_BASE nodes run plain Goldilocks arithmetic (1 base mul instead of the
//     9 the {x,0,0} embedding pays) and touch a third of the scratch bytes.
//     DIM_EXT nodes run ext3, widening a base operand to `{x,0,0}` on read —
//     exactly the CPU walker's `to_ext` auto-embed. `lower` normalizes dims so
//     DIM_BASE guarantees both operands are base; since the base primitives are
//     the same `goldilocks.cuh` ops the ext3 lanes use, outputs stay
//     bit-identical to the dim-split CPU walk `eval_device_program`.
//
// Output is the per-constraint eval matrix `d_evals[c*num_rows + row]` (Fe3;
// base-rooted constraints carry their value in `.a`). The fused
// `z*Σ(Cᵢ·βᵢ) + boundary` accumulation (avoiding the D2H of that matrix) is
// `constraint_composition_kernel` below — the production path.
//
// Op tags, dim tags and the `Var` packing MUST stay in sync with
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

// -- dim tags (mirror device.rs DIM_*) --
#define DIM_BASE 0u
#define DIM_EXT 1u

// A flat IR node. Packed into two u64 words for a pure-u64 upload (matching the
// crate's device-buffer convention):
//   word0 = op | (a << 32) ; word1 = b | (dim << 32)
// This mirrors the `#[repr(C)] DeviceNode { op, a, b, dim: u32 }` payload.
struct Node {
    uint32_t op, a, b, dim;
};

__device__ __forceinline__ Node load_node(const uint64_t *d_nodes, uint64_t i) {
    uint64_t w0 = d_nodes[2 * i];
    uint64_t w1 = d_nodes[2 * i + 1];
    Node n;
    n.op = (uint32_t)(w0 & 0xFFFFFFFFull);
    n.a = (uint32_t)(w0 >> 32);
    n.b = (uint32_t)(w1 & 0xFFFFFFFFull);
    n.dim = (uint32_t)(w1 >> 32);
    return n;
}

// Store a base value into a node's single scratch slot.
__device__ __forceinline__ void store_base(uint64_t *vals, uint64_t vstride, uint64_t off,
                                           uint64_t x) {
    vals[off * vstride] = x;
}

// Store an ext3 value into a node's three scratch slots.
__device__ __forceinline__ void store_ext(uint64_t *vals, uint64_t vstride, uint64_t off,
                                          const Fe3 &v) {
    vals[(off + 0) * vstride] = v.a;
    vals[(off + 1) * vstride] = v.b;
    vals[(off + 2) * vstride] = v.c;
}

// Load an operand node's value as ext3, widening a 1-slot base value to its
// embedding `{x,0,0}` — the device image of the CPU walker's `to_ext`.
__device__ __forceinline__ Fe3 load_operand(const uint64_t *vals, uint64_t vstride,
                                            const uint64_t *d_val_offsets, uint32_t id) {
    uint64_t off = d_val_offsets[id];
    uint64_t width = d_val_offsets[id + 1] - off;
    uint64_t a = vals[off * vstride];
    if (width == 1) {
        return ext3::make(a, 0, 0);
    }
    return ext3::make(a, vals[(off + 1) * vstride], vals[(off + 2) * vstride]);
}

// Shared forward pass: evaluate every IR node of the program for one LDE row
// into the per-thread packed value scratch (node `i`'s slots start at
// `vals[d_val_offsets[i] * vstride]`; id `i` references only nodes `< i`). The
// single home of the op semantics — both kernels below run this exact walk, so
// an op change stays in lockstep with `constraint_ir/device.rs` in one place.
//
// The DIM_BASE arithmetic arms rely on `DeviceProgram::lower`'s normalization:
// a DIM_BASE arithmetic node's operands are guaranteed base (1-slot), so the
// base ops read/write single u64 slots with plain Goldilocks arithmetic.
__device__ __forceinline__ void eval_program_row(
    uint64_t *vals, uint64_t vstride, const uint64_t *d_nodes, uint64_t num_nodes,
    const uint64_t *d_val_offsets, const uint64_t *d_base_consts, const Fe3 *d_ext_consts,
    const Fe3 *d_rap_challenges, const Fe3 *d_alpha_powers, Fe3 table_offset, uint64_t row,
    uint64_t next_step, uint64_t num_rows, const uint64_t *d_main, uint64_t main_stride,
    const uint64_t *d_aux, uint64_t aux_stride) {
    for (uint64_t i = 0; i < num_nodes; i++) {
        Node nd = load_node(d_nodes, i);
        uint64_t off = d_val_offsets[i];
        switch (nd.op) {
        case OP_CONST_BASE:
            store_base(vals, vstride, off, d_base_consts[nd.a]);
            break;
        case OP_CONST_EXT:
            store_ext(vals, vstride, off, d_ext_consts[nd.a]);
            break;
        case OP_VAR: {
            // a = col (low 16 bits); b = main<<16 | offset<<8 | row (see
            // device.rs pack_var). Base (main) columns are column-major
            // `d_main[col*main_stride + r]`; ext (aux) columns store component
            // k at `d_aux[(col*3 + k)*aux_stride + r]` (GpuLdeExt3). The frame
            // `offset` selects row `r = (row + offset*next_step) mod num_rows`.
            uint32_t col = nd.a & 0xFFFFu;
            bool is_main = ((nd.b >> 16) & 1u) != 0u;
            uint32_t frame_offset = (nd.b >> 8) & 0xFFu;

            uint64_t r = row + (uint64_t)frame_offset * next_step;
            if (r >= num_rows) {
                r -= num_rows; // wrap; offset*next_step < num_rows by construction
            }

            if (is_main) {
                store_base(vals, vstride, off, d_main[(uint64_t)col * main_stride + r]);
            } else {
                uint64_t base = (uint64_t)col * 3;
                store_ext(vals, vstride, off,
                          ext3::make(d_aux[(base + 0) * aux_stride + r],
                                     d_aux[(base + 1) * aux_stride + r],
                                     d_aux[(base + 2) * aux_stride + r]));
            }
            break;
        }
        case OP_RAP_CHALLENGE:
            store_ext(vals, vstride, off, d_rap_challenges[nd.a]);
            break;
        case OP_ALPHA_POW:
            store_ext(vals, vstride, off, d_alpha_powers[nd.a]);
            break;
        case OP_TABLE_OFFSET:
            store_ext(vals, vstride, off, table_offset);
            break;
        case OP_ADD:
            if (nd.dim == DIM_BASE) {
                store_base(vals, vstride, off,
                           goldilocks::add(vals[d_val_offsets[nd.a] * vstride],
                                           vals[d_val_offsets[nd.b] * vstride]));
            } else {
                store_ext(vals, vstride, off,
                          ext3::add(load_operand(vals, vstride, d_val_offsets, nd.a),
                                    load_operand(vals, vstride, d_val_offsets, nd.b)));
            }
            break;
        case OP_SUB:
            if (nd.dim == DIM_BASE) {
                store_base(vals, vstride, off,
                           goldilocks::sub(vals[d_val_offsets[nd.a] * vstride],
                                           vals[d_val_offsets[nd.b] * vstride]));
            } else {
                store_ext(vals, vstride, off,
                          ext3::sub(load_operand(vals, vstride, d_val_offsets, nd.a),
                                    load_operand(vals, vstride, d_val_offsets, nd.b)));
            }
            break;
        case OP_MUL:
            if (nd.dim == DIM_BASE) {
                store_base(vals, vstride, off,
                           goldilocks::mul(vals[d_val_offsets[nd.a] * vstride],
                                           vals[d_val_offsets[nd.b] * vstride]));
            } else {
                store_ext(vals, vstride, off,
                          ext3::mul(load_operand(vals, vstride, d_val_offsets, nd.a),
                                    load_operand(vals, vstride, d_val_offsets, nd.b)));
            }
            break;
        case OP_NEG:
            if (nd.dim == DIM_BASE) {
                store_base(vals, vstride, off,
                           goldilocks::neg(vals[d_val_offsets[nd.a] * vstride]));
            } else {
                store_ext(vals, vstride, off,
                          ext3::neg(load_operand(vals, vstride, d_val_offsets, nd.a)));
            }
            break;
        case OP_EMBED:
            // Widening load IS the embedding; Embed nodes are always DIM_EXT.
            store_ext(vals, vstride, off, load_operand(vals, vstride, d_val_offsets, nd.a));
            break;
        default:
            if (nd.dim == DIM_BASE) {
                store_base(vals, vstride, off, 0);
            } else {
                store_ext(vals, vstride, off, ext3::zero());
            }
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
    const uint64_t *__restrict__ d_val_offsets, // num_nodes + 1 scratch prefix sums
    const uint64_t *__restrict__ d_base_consts,
    const Fe3 *__restrict__ d_ext_consts,
    const uint64_t *__restrict__ d_roots,
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
    // scratch: per-thread packed value slots, [val_offsets[num_nodes] * num_threads]
    uint64_t *__restrict__ d_values) {
    uint64_t task_offset = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * blockDim.x;

    uint64_t *vals = d_values + task_offset;
    uint64_t vstride = num_threads;
    Fe3 table_offset = *d_table_offset;

    for (uint64_t row = task_offset; row < num_rows; row += num_threads) {
        eval_program_row(vals, vstride, d_nodes, num_nodes, d_val_offsets, d_base_consts,
                         d_ext_consts, d_rap_challenges, d_alpha_powers, table_offset, row,
                         next_step, num_rows, d_main, main_stride, d_aux, aux_stride);

        // Emit each constraint root (base roots widen to {x,0,0}).
        for (uint64_t c = 0; c < num_roots; c++) {
            uint32_t root = (uint32_t)d_roots[c];
            d_evals[c * num_rows + row] = load_operand(vals, vstride, d_val_offsets, root);
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
// where a base-rooted C_c is widened to its embedding {C,0,0} on read (exactly
// as the interpreter), z_inv is the cyclic base transition-zerofier inverse, and
// the boundary term reads the resident trace at column `b_col[b]` (main or aux).
extern "C" __global__ void constraint_composition_kernel(
    // output: one H(row) per LDE row
    Fe3 *__restrict__ d_h,
    // program (flat blob) — identical to the interpreter kernel
    const uint64_t *__restrict__ d_nodes,
    uint64_t num_nodes,
    const uint64_t *__restrict__ d_val_offsets, // num_nodes + 1 scratch prefix sums
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
    // scratch: per-thread packed value slots, [val_offsets[num_nodes] * num_threads]
    uint64_t *__restrict__ d_values) {
    uint64_t task_offset = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * blockDim.x;

    uint64_t *vals = d_values + task_offset;
    uint64_t vstride = num_threads;
    Fe3 table_offset = *d_table_offset;

    for (uint64_t row = task_offset; row < num_rows; row += num_threads) {
        eval_program_row(vals, vstride, d_nodes, num_nodes, d_val_offsets, d_base_consts,
                         d_ext_consts, d_rap_challenges, d_alpha_powers, table_offset, row,
                         next_step, num_rows, d_main, main_stride, d_aux, aux_stride);

        // Transition: z_inv * Σ_c beta_c * C_c.
        Fe3 sum = ext3::zero();
        for (uint64_t c = 0; c < num_roots; c++) {
            Fe3 cval = load_operand(vals, vstride, d_val_offsets, (uint32_t)d_roots[c]);
            sum = ext3::add(sum, ext3::mul(d_beta_trans[c], cval));
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
