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
// Design (v1, "stripped" — mirrors OpenVM's GLOBAL=true quotient kernel):
//   * One thread per LDE row, grid-stride over all rows (fixed launch, any size).
//   * Per-thread value array in GLOBAL memory, strided by thread for coalescing:
//     node `i` for this thread lives at `d_values[task_offset + i*num_threads]`.
//   * ALL values are carried as ext3 `Fe3` (a base value `x` is the embedding
//     `{x,0,0}`). Because embedding is a ring homomorphism and the device field
//     arithmetic is bit-identical to the CPU path, doing every op in ext3 yields
//     outputs bit-identical to the dim-split CPU walk — the per-node `dim` tag
//     is therefore unused here and reserved for the base/ext-split optimization
//     (Phase 6). `Op::Embed` is consequently the identity.
//
// Output is the per-constraint eval matrix `d_evals[c*num_rows + row]` (Fe3;
// base-rooted constraints carry their value in `.a`). Fusing the
// `z*Σ(Cᵢ·βᵢ) + boundary` accumulation into this kernel (to avoid a D2H of the
// matrix — the actual data-residency win) is the pipeline-integration follow-on.
//
// Op tags and the `Var` packing MUST stay in sync with
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

// Resolve an `Op::Var` leaf against the device-resident LDE columns.
//   a = col (low 16 bits); b = main<<16 | offset<<8 | row  (see device.rs pack_var)
// Base (main) columns are column-major `d_main[col*main_stride + r]`; ext (aux)
// columns store component k at `d_aux[(col*3 + k)*aux_stride + r]` (GpuLdeExt3).
// The frame `offset` selects row `r = (row + offset*next_step) mod num_rows`.
__device__ __forceinline__ Fe3 read_var(uint32_t a, uint32_t b, uint64_t row, uint64_t next_step,
                                        uint64_t num_rows, const uint64_t *d_main,
                                        uint64_t main_stride, const uint64_t *d_aux,
                                        uint64_t aux_stride) {
    uint32_t col = a & 0xFFFFu;
    bool is_main = ((b >> 16) & 1u) != 0u;
    uint32_t offset = (b >> 8) & 0xFFu;

    uint64_t r = row + (uint64_t)offset * next_step;
    if (r >= num_rows) {
        r -= num_rows; // wrap; offset*next_step < num_rows by construction
    }

    if (is_main) {
        uint64_t x = d_main[(uint64_t)col * main_stride + r];
        return ext3::make(x, 0, 0);
    }
    uint64_t base = (uint64_t)col * 3;
    uint64_t a0 = d_aux[(base + 0) * aux_stride + r];
    uint64_t a1 = d_aux[(base + 1) * aux_stride + r];
    uint64_t a2 = d_aux[(base + 2) * aux_stride + r];
    return ext3::make(a0, a1, a2);
}

extern "C" __global__ void constraint_interp_kernel(
    // output: per-constraint eval matrix, constraint-major [num_roots * num_rows]
    Fe3 *__restrict__ d_evals,
    // program (flat blob)
    const uint64_t *__restrict__ d_nodes, // 2 u64 per node
    uint64_t num_nodes,
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
    // scratch: per-thread value array, [num_nodes * num_threads]
    Fe3 *__restrict__ d_values) {
    uint64_t task_offset = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * blockDim.x;

    Fe3 *vals = d_values + task_offset;
    uint64_t vstride = num_threads;
    Fe3 table_offset = *d_table_offset;

    for (uint64_t row = task_offset; row < num_rows; row += num_threads) {
        // Forward pass: id `i` references only nodes `< i`.
        for (uint64_t i = 0; i < num_nodes; i++) {
            Node nd = load_node(d_nodes, i);
            Fe3 v;
            switch (nd.op) {
            case OP_CONST_BASE:
                v = ext3::make(d_base_consts[nd.a], 0, 0);
                break;
            case OP_CONST_EXT:
                v = d_ext_consts[nd.a];
                break;
            case OP_VAR:
                v = read_var(nd.a, nd.b, row, next_step, num_rows, d_main, main_stride, d_aux,
                             aux_stride);
                break;
            case OP_RAP_CHALLENGE:
                v = d_rap_challenges[nd.a];
                break;
            case OP_ALPHA_POW:
                v = d_alpha_powers[nd.a];
                break;
            case OP_TABLE_OFFSET:
                v = table_offset;
                break;
            case OP_ADD:
                v = ext3::add(vals[(uint64_t)nd.a * vstride], vals[(uint64_t)nd.b * vstride]);
                break;
            case OP_SUB:
                v = ext3::sub(vals[(uint64_t)nd.a * vstride], vals[(uint64_t)nd.b * vstride]);
                break;
            case OP_MUL:
                v = ext3::mul(vals[(uint64_t)nd.a * vstride], vals[(uint64_t)nd.b * vstride]);
                break;
            case OP_NEG:
                v = ext3::neg(vals[(uint64_t)nd.a * vstride]);
                break;
            case OP_EMBED:
                // All-ext representation: the base operand is already {x,0,0}.
                v = vals[(uint64_t)nd.a * vstride];
                break;
            default:
                v = ext3::zero();
                break;
            }
            vals[i * vstride] = v;
        }

        // Emit each constraint root.
        for (uint64_t c = 0; c < num_roots; c++) {
            uint64_t root = d_roots[c];
            d_evals[c * num_rows + row] = vals[root * vstride];
        }
    }
}
