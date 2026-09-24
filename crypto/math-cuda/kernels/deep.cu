// R4 deep composition polynomial evaluations.
//
// For each row i in 0..domain_size, accumulate:
//   result_i = sum over j of gamma_j * (H_j(x_i) - H_j(z^K)) * inv_h[i]               (H terms)
//            + sum over j,k of gamma'_{j,k} * (t_j(x_i) - t_j(z*w^k)) * inv_t[k,i]    (trace)
//
// evaluated in the folded (distributed) form
//   result_i = inv_h[i] * (sum_j gamma_j * H_j(x_i) - K_h)
//            + sum_k inv_t[k,i] * (sum_j gamma'_{j,k} * t_j(x_i) - ood_compressed[k])
// where the per-proof constants
//   K_h               = sum_j gamma_j * H_j(z^K)
//   ood_compressed[k] = sum_j gamma'_{j,k} * t_j(z*w^k)
// are folded on the host (`fold_ood_constants` in `src/deep.rs`). Same field
// element as the unfolded sum; it saves the per-(j,k) inv_t multiply, the
// per-row OOD subtractions, and turns main-column gamma products into
// ext3 x base.
//
// The kernel reads LDE column data at `i * row_stride`. Real R4 callers
// always pass `row_stride = 1` and `domain_size = lde_size` (evaluates
// every row); the stride parameter is exercised by the parity tests in
// `tests/deep.rs` so the kernel can also run a trace-coset evaluation.
// `j` ranges over num_parts for H-terms and num_total_cols (= num_main +
// num_aux) for trace terms. `k` ranges over num_eval_points.
//
// Buffer layouts (ALL on device):
//   main_lde    base, column-major: main_lde[c * lde_stride + r]
//   aux_lde     ext3 de-interleaved: aux_lde[(c*3 + k) * lde_stride + r]
//   h_lde       ext3 de-interleaved: h_lde[(p*3 + k) * lde_stride + r]
//   ood_folded  (1 + num_eval_points) * 3 (ext3 interleaved):
//               [K_h, ood_compressed[0], ..., ood_compressed[num_eval_points-1]]
//   gammas_h    num_parts * 3
//   gammas_tr   num_total_cols * num_eval_points * 3 (ext3 interleaved,
//               indexed as (col_idx * num_eval_points + k) * 3 + comp)
//   inv_h       domain_size * 3
//   inv_t       num_eval_points * domain_size * 3
//   deep_out    domain_size * 3 (ext3 interleaved; caller reinterprets)

#include "goldilocks.cuh"
#include "ext3.cuh"

// Largest block of eval points handled in one pass over the columns. Each
// point keeps one ext3 accumulator in registers; wider AIRs are processed in
// blocks of this size (re-reading the columns once per block).
#define DEEP_MAX_EVAL_BLOCK 4

__device__ __forceinline__ ext3::Fe3 load3(const uint64_t *p, uint64_t idx) {
    return ext3::make(p[idx], p[idx + 1], p[idx + 2]);
}

// Trace terms for eval points k_base..k_base+K, added into `result`. The
// j-outer / k-inner nesting loads each column value once and reuses it for
// all K points; `inv_t[k,i]` is applied once per point after the column sum.
template <int K>
__device__ __forceinline__ void deep_trace_block(
    ext3::Fe3 &result,
    const uint64_t *main_lde,
    const uint64_t *aux_lde,
    uint64_t lde_stride,
    uint64_t num_main,
    uint64_t num_aux,
    uint64_t num_eval_points,
    uint64_t k_base,
    uint64_t row,
    uint64_t i,
    uint64_t domain_size,
    const uint64_t *ood_folded,
    const uint64_t *gammas_tr,
    const uint64_t *inv_t) {
    ext3::Fe3 acc[K];
#pragma unroll
    for (int kk = 0; kk < K; ++kk) acc[kk] = ext3::zero();

    // Main columns: base-field value, so gamma * t_val is ext3 x base.
    for (uint64_t j = 0; j < num_main; ++j) {
        uint64_t t_val = main_lde[j * lde_stride + row];
        uint64_t idx = (j * num_eval_points + k_base) * 3;
#pragma unroll
        for (int kk = 0; kk < K; ++kk) {
            ext3::Fe3 gamma = load3(gammas_tr, idx + kk * 3);
            acc[kk] = ext3::add(acc[kk], ext3::mul_base(gamma, t_val));
        }
    }

    // Aux columns: ext3 value.
    for (uint64_t j = 0; j < num_aux; ++j) {
        ext3::Fe3 t_val = {
            aux_lde[(j * 3 + 0) * lde_stride + row],
            aux_lde[(j * 3 + 1) * lde_stride + row],
            aux_lde[(j * 3 + 2) * lde_stride + row],
        };
        uint64_t idx = ((num_main + j) * num_eval_points + k_base) * 3;
#pragma unroll
        for (int kk = 0; kk < K; ++kk) {
            ext3::Fe3 gamma = load3(gammas_tr, idx + kk * 3);
            acc[kk] = ext3::add(acc[kk], ext3::mul(gamma, t_val));
        }
    }

#pragma unroll
    for (int kk = 0; kk < K; ++kk) {
        uint64_t k = k_base + kk;
        ext3::Fe3 ood_k = load3(ood_folded, (1 + k) * 3);
        ext3::Fe3 inv_t_ki = load3(inv_t, (k * domain_size + i) * 3);
        result = ext3::add(result, ext3::mul(ext3::sub(acc[kk], ood_k), inv_t_ki));
    }
}

extern "C" __global__ void deep_composition_ext3_row(
    const uint64_t *main_lde,
    const uint64_t *aux_lde,
    const uint64_t *h_lde,
    uint64_t lde_stride,
    uint64_t num_main,
    uint64_t num_aux,
    uint64_t num_parts,
    uint64_t num_eval_points,
    uint64_t row_stride,
    uint64_t domain_size,
    const uint64_t *ood_folded,
    const uint64_t *gammas_h,
    const uint64_t *gammas_tr,
    const uint64_t *inv_h,
    const uint64_t *inv_t,
    uint64_t *deep_out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= domain_size) return;
    uint64_t row = i * row_stride;

    // H-terms: inv_h[i] * (sum_j gamma_j * H_j(x_i) - K_h)
    ext3::Fe3 h_acc = ext3::zero();
    for (uint64_t j = 0; j < num_parts; ++j) {
        ext3::Fe3 h_val = {
            h_lde[(j * 3 + 0) * lde_stride + row],
            h_lde[(j * 3 + 1) * lde_stride + row],
            h_lde[(j * 3 + 2) * lde_stride + row],
        };
        ext3::Fe3 gamma = load3(gammas_h, j * 3);
        h_acc = ext3::add(h_acc, ext3::mul(gamma, h_val));
    }
    ext3::Fe3 inv_h_i = load3(inv_h, i * 3);
    ext3::Fe3 result = ext3::mul(ext3::sub(h_acc, load3(ood_folded, 0)), inv_h_i);

    // Trace terms, in blocks of at most DEEP_MAX_EVAL_BLOCK eval points so the
    // accumulators stay in registers (production AIRs use 2 points: one block).
    uint64_t k = 0;
    for (; k + DEEP_MAX_EVAL_BLOCK <= num_eval_points; k += DEEP_MAX_EVAL_BLOCK) {
        deep_trace_block<DEEP_MAX_EVAL_BLOCK>(result, main_lde, aux_lde, lde_stride, num_main,
                                              num_aux, num_eval_points, k, row, i, domain_size,
                                              ood_folded, gammas_tr, inv_t);
    }
    switch (num_eval_points - k) {
        case 3:
            deep_trace_block<3>(result, main_lde, aux_lde, lde_stride, num_main, num_aux,
                                num_eval_points, k, row, i, domain_size, ood_folded, gammas_tr,
                                inv_t);
            break;
        case 2:
            deep_trace_block<2>(result, main_lde, aux_lde, lde_stride, num_main, num_aux,
                                num_eval_points, k, row, i, domain_size, ood_folded, gammas_tr,
                                inv_t);
            break;
        case 1:
            deep_trace_block<1>(result, main_lde, aux_lde, lde_stride, num_main, num_aux,
                                num_eval_points, k, row, i, domain_size, ood_folded, gammas_tr,
                                inv_t);
            break;
        default:
            break;
    }

    uint64_t out_idx = i * 3;
    deep_out[out_idx + 0] = result.a;
    deep_out[out_idx + 1] = result.b;
    deep_out[out_idx + 2] = result.c;
}

// Out-of-place bit-reverse permutation of an interleaved ext3 codeword:
// out[i] = in[bitrev_log_n(i)]. Puts the DEEP codeword in FRI order without
// leaving the device.
extern "C" __global__ void bit_reverse_ext3_interleaved(
    const uint64_t *__restrict__ in,
    uint64_t *__restrict__ out,
    uint64_t n,
    uint32_t log_n) {
    for (uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t j = __brevll(i) >> (64 - log_n);
        out[i * 3 + 0] = in[j * 3 + 0];
        out[i * 3 + 1] = in[j * 3 + 1];
        out[i * 3 + 2] = in[j * 3 + 2];
    }
}
