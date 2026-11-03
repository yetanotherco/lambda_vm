// R4 deep composition polynomial evaluations.
//
// For each row i in 0..domain_size, accumulate:
//   result_i = sum over j of gamma_j * (H_j(x_i) - H_j(z^K)) * inv_h[i]               (H terms)
//            + sum over j,k of gamma'_{j,k} * (t_j(x_i) - t_j(z*w^k)) * inv_t[k,i]    (trace)
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
//   h_ood       num_parts * 3  (ext3 interleaved)
//   trace_ood   num_total_cols * num_eval_points * 3 (ext3 interleaved,
//               indexed as (col_idx * num_eval_points + k) * 3 + comp)
//   gammas_h    num_parts * 3
//   gammas_tr   num_total_cols * num_eval_points * 3
//   inv_h       domain_size * 3
//   inv_t       num_eval_points * domain_size * 3
//   deep_out    domain_size * 3 (ext3 interleaved; caller reinterprets)

#include "goldilocks.cuh"
#include "ext3.cuh"

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
    const uint64_t *h_ood,
    const uint64_t *trace_ood,
    const uint64_t *gammas_h,
    const uint64_t *gammas_tr,
    const uint64_t *inv_h,
    const uint64_t *inv_t,
    uint64_t *deep_out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= domain_size) return;
    uint64_t row = i * row_stride;

    ext3::Fe3 result = ext3::zero();
    ext3::Fe3 inv_h_i = {inv_h[i * 3], inv_h[i * 3 + 1], inv_h[i * 3 + 2]};

    // H-terms
    for (uint64_t j = 0; j < num_parts; ++j) {
        ext3::Fe3 h_val = {
            h_lde[(j * 3 + 0) * lde_stride + row],
            h_lde[(j * 3 + 1) * lde_stride + row],
            h_lde[(j * 3 + 2) * lde_stride + row],
        };
        ext3::Fe3 h_ood_j = {h_ood[j * 3], h_ood[j * 3 + 1], h_ood[j * 3 + 2]};
        ext3::Fe3 num = ext3::sub(h_val, h_ood_j);
        ext3::Fe3 gamma = {gammas_h[j * 3], gammas_h[j * 3 + 1], gammas_h[j * 3 + 2]};
        ext3::Fe3 tmp = ext3::mul(gamma, num);
        tmp = ext3::mul(tmp, inv_h_i);
        result = ext3::add(result, tmp);
    }

    // Main trace terms: t_val (base) - t_ood (ext3)
    for (uint64_t j = 0; j < num_main; ++j) {
        uint64_t t_val = main_lde[j * lde_stride + row];
        for (uint64_t k = 0; k < num_eval_points; ++k) {
            uint64_t idx = (j * num_eval_points + k) * 3;
            ext3::Fe3 t_ood = {trace_ood[idx], trace_ood[idx + 1], trace_ood[idx + 2]};
            ext3::Fe3 num = {
                goldilocks::sub(t_val, t_ood.a),
                goldilocks::neg(t_ood.b),
                goldilocks::neg(t_ood.c),
            };
            ext3::Fe3 gamma = {gammas_tr[idx], gammas_tr[idx + 1], gammas_tr[idx + 2]};
            uint64_t inv_t_idx = (k * domain_size + i) * 3;
            ext3::Fe3 inv_t_ki = {inv_t[inv_t_idx], inv_t[inv_t_idx + 1], inv_t[inv_t_idx + 2]};
            ext3::Fe3 tmp = ext3::mul(gamma, num);
            tmp = ext3::mul(tmp, inv_t_ki);
            result = ext3::add(result, tmp);
        }
    }

    // Aux trace terms: t_val (ext3) - t_ood (ext3)
    for (uint64_t j = 0; j < num_aux; ++j) {
        ext3::Fe3 t_val = {
            aux_lde[(j * 3 + 0) * lde_stride + row],
            aux_lde[(j * 3 + 1) * lde_stride + row],
            aux_lde[(j * 3 + 2) * lde_stride + row],
        };
        uint64_t trace_j = num_main + j;
        for (uint64_t k = 0; k < num_eval_points; ++k) {
            uint64_t idx = (trace_j * num_eval_points + k) * 3;
            ext3::Fe3 t_ood = {trace_ood[idx], trace_ood[idx + 1], trace_ood[idx + 2]};
            ext3::Fe3 num = ext3::sub(t_val, t_ood);
            ext3::Fe3 gamma = {gammas_tr[idx], gammas_tr[idx + 1], gammas_tr[idx + 2]};
            uint64_t inv_t_idx = (k * domain_size + i) * 3;
            ext3::Fe3 inv_t_ki = {inv_t[inv_t_idx], inv_t[inv_t_idx + 1], inv_t[inv_t_idx + 2]};
            ext3::Fe3 tmp = ext3::mul(gamma, num);
            tmp = ext3::mul(tmp, inv_t_ki);
            result = ext3::add(result, tmp);
        }
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

// ---------------------------------------------------------------------------
// `deep_composition_ext3_row` with its denominators inverted in
// registers, per row, instead of read from the `inv_h` / `inv_t` buffers the
// six-kernel scan fills.
//
// Row `i` sits at the domain point `coset_points[i]`; its M = 1 + num_eval_points
// denominators are `x - z_scalars[m]` (DenomSign::XMinusZ: m = 0 is the H-term
// pole z^K, m = 1 + k the trace pole z·w^k — the layout the scan path's
// `compute_and_invert_denoms_ext3_dev` is called with). They are inverted
// together with ONE base-field inversion (`ext3_inv.cuh`), and the sum below is
// `deep_composition_ext3_row`'s, term for term. The inverses are the scan's as
// field elements, so the codeword is too.
//
// M is a template parameter so the inverses stay in registers; the host picks
// the instantiation and keeps the buffered kernel for any other M.
// `zero_flag` (nullable) as in `invert_denoms_rowwise_ext3_k*`.
// ---------------------------------------------------------------------------
#include "ext3_inv.cuh"

template <int M>
__device__ __forceinline__ void deep_fused_body(
    const uint64_t *__restrict__ main_lde, const uint64_t *__restrict__ aux_lde,
    const uint64_t *__restrict__ h_lde, uint64_t lde_stride, uint64_t num_main, uint64_t num_aux,
    uint64_t num_parts, uint64_t row_stride, uint64_t domain_size,
    const uint64_t *__restrict__ h_ood, const uint64_t *__restrict__ trace_ood,
    const uint64_t *__restrict__ gammas_h, const uint64_t *__restrict__ gammas_tr,
    const uint64_t *__restrict__ coset_points, const uint64_t *__restrict__ z_scalars,
    uint64_t *__restrict__ deep_out, uint32_t *zero_flag) {
    constexpr int K = M - 1;  // num_eval_points
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= domain_size) return;
    uint64_t row = i * row_stride;

    uint64_t x = coset_points[i];
    ext3::Fe3 inv[M];
#pragma unroll
    for (int m = 0; m < M; ++m) inv[m] = ext3_inv::x_minus_z(x, z_scalars + 3 * m);
    uint64_t product = ext3_inv::batch_inv<M>(inv);
    if (zero_flag != nullptr && goldilocks::canonical(product) == 0) atomicOr(zero_flag, 1u);

    ext3::Fe3 result = ext3::zero();

    // H-terms
    for (uint64_t j = 0; j < num_parts; ++j) {
        ext3::Fe3 h_val = {
            h_lde[(j * 3 + 0) * lde_stride + row],
            h_lde[(j * 3 + 1) * lde_stride + row],
            h_lde[(j * 3 + 2) * lde_stride + row],
        };
        ext3::Fe3 h_ood_j = {h_ood[j * 3], h_ood[j * 3 + 1], h_ood[j * 3 + 2]};
        ext3::Fe3 num = ext3::sub(h_val, h_ood_j);
        ext3::Fe3 gamma = {gammas_h[j * 3], gammas_h[j * 3 + 1], gammas_h[j * 3 + 2]};
        ext3::Fe3 tmp = ext3::mul(gamma, num);
        tmp = ext3::mul(tmp, inv[0]);
        result = ext3::add(result, tmp);
    }

    // Main trace terms: t_val (base) - t_ood (ext3)
    for (uint64_t j = 0; j < num_main; ++j) {
        uint64_t t_val = main_lde[j * lde_stride + row];
#pragma unroll
        for (int k = 0; k < K; ++k) {
            uint64_t idx = (j * K + k) * 3;
            ext3::Fe3 t_ood = {trace_ood[idx], trace_ood[idx + 1], trace_ood[idx + 2]};
            ext3::Fe3 num = {
                goldilocks::sub(t_val, t_ood.a),
                goldilocks::neg(t_ood.b),
                goldilocks::neg(t_ood.c),
            };
            ext3::Fe3 gamma = {gammas_tr[idx], gammas_tr[idx + 1], gammas_tr[idx + 2]};
            ext3::Fe3 tmp = ext3::mul(gamma, num);
            tmp = ext3::mul(tmp, inv[1 + k]);
            result = ext3::add(result, tmp);
        }
    }

    // Aux trace terms: t_val (ext3) - t_ood (ext3)
    for (uint64_t j = 0; j < num_aux; ++j) {
        ext3::Fe3 t_val = {
            aux_lde[(j * 3 + 0) * lde_stride + row],
            aux_lde[(j * 3 + 1) * lde_stride + row],
            aux_lde[(j * 3 + 2) * lde_stride + row],
        };
        uint64_t trace_j = num_main + j;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            uint64_t idx = (trace_j * K + k) * 3;
            ext3::Fe3 t_ood = {trace_ood[idx], trace_ood[idx + 1], trace_ood[idx + 2]};
            ext3::Fe3 num = ext3::sub(t_val, t_ood);
            ext3::Fe3 gamma = {gammas_tr[idx], gammas_tr[idx + 1], gammas_tr[idx + 2]};
            ext3::Fe3 tmp = ext3::mul(gamma, num);
            tmp = ext3::mul(tmp, inv[1 + k]);
            result = ext3::add(result, tmp);
        }
    }

    uint64_t out_idx = i * 3;
    deep_out[out_idx + 0] = result.a;
    deep_out[out_idx + 1] = result.b;
    deep_out[out_idx + 2] = result.c;
}

#define DEEP_FUSED(M)                                                                             \
    extern "C" __global__ void deep_composition_ext3_fused_m##M(                                  \
        const uint64_t *__restrict__ main_lde, const uint64_t *__restrict__ aux_lde,              \
        const uint64_t *__restrict__ h_lde, uint64_t lde_stride, uint64_t num_main,               \
        uint64_t num_aux, uint64_t num_parts, uint64_t row_stride, uint64_t domain_size,          \
        const uint64_t *__restrict__ h_ood, const uint64_t *__restrict__ trace_ood,               \
        const uint64_t *__restrict__ gammas_h, const uint64_t *__restrict__ gammas_tr,            \
        const uint64_t *__restrict__ coset_points, const uint64_t *__restrict__ z_scalars,        \
        uint64_t *__restrict__ deep_out, uint32_t *zero_flag) {                                   \
        deep_fused_body<M>(main_lde, aux_lde, h_lde, lde_stride, num_main, num_aux, num_parts,    \
                           row_stride, domain_size, h_ood, trace_ood, gammas_h, gammas_tr,        \
                           coset_points, z_scalars, deep_out, zero_flag);                         \
    }

// num_eval_points 0..=3; the tables prove with 2 (transition offsets {0, 1}).
DEEP_FUSED(1)
DEEP_FUSED(2)
DEEP_FUSED(3)
DEEP_FUSED(4)

// ---------------------------------------------------------------------------
// `deep_composition_ext3_fused_m*` with the row-independent part taken out of
// the row (`LAMBDA_VM_DEEP_HOISTED`; on by default). For each pole k,
//   Σ_j γ_jk·(t_j(x) − o_jk) = Σ_j γ_jk·t_j(x) − C_k,   C_k = Σ_j γ_jk·o_jk,
// and the H-terms alike (C_h = Σ_j γ_j·h_ood_j): a row costs one ext3×base
// product per (main column, pole), one ext3×ext3 per (aux column, pole) and per
// part, and 1 + K multiplies by the inverses, against two ext3×ext3 per term in
// `deep_fused_body`. `consts` holds C_h, then C_0 … C_{K−1} (ext3 interleaved),
// computed once on the host (`deep::hoisted_constants`). The same field element
// as the fused kernel at `row_stride = 1`, written canonical: the fused kernel's
// words are canonical but for a ≈ 2^-32 chance a word, so the two agree as raw
// words but there (measured on the card: 100 % at four shapes).
// ---------------------------------------------------------------------------
template <int M>
__device__ __forceinline__ void deep_hoisted_body(
    const uint64_t *__restrict__ main_lde, const uint64_t *__restrict__ aux_lde,
    const uint64_t *__restrict__ h_lde, uint64_t lde_stride, uint64_t num_main, uint64_t num_aux,
    uint64_t num_parts, uint64_t domain_size, const uint64_t *__restrict__ consts,
    const uint64_t *__restrict__ gammas_h, const uint64_t *__restrict__ gammas_tr,
    const uint64_t *__restrict__ coset_points, const uint64_t *__restrict__ z_scalars,
    uint64_t *__restrict__ deep_out, uint32_t *zero_flag) {
    constexpr int K = M - 1;
    constexpr int KA = K > 0 ? K : 1;
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= domain_size) return;

    uint64_t x = coset_points[i];
    ext3::Fe3 inv[M];
#pragma unroll
    for (int m = 0; m < M; ++m) inv[m] = ext3_inv::x_minus_z(x, z_scalars + 3 * m);
    uint64_t product = ext3_inv::batch_inv<M>(inv);
    if (zero_flag != nullptr && goldilocks::canonical(product) == 0) atomicOr(zero_flag, 1u);

    ext3::Fe3 acc_h = ext3::zero();
    for (uint64_t j = 0; j < num_parts; ++j) {
        ext3::Fe3 h_val = {
            h_lde[(j * 3 + 0) * lde_stride + i],
            h_lde[(j * 3 + 1) * lde_stride + i],
            h_lde[(j * 3 + 2) * lde_stride + i],
        };
        ext3::Fe3 gamma = {gammas_h[j * 3], gammas_h[j * 3 + 1], gammas_h[j * 3 + 2]};
        acc_h = ext3::add(acc_h, ext3::mul(gamma, h_val));
    }

    ext3::Fe3 acc[KA];
#pragma unroll
    for (int k = 0; k < KA; ++k) acc[k] = ext3::zero();
    for (uint64_t j = 0; j < num_main; ++j) {
        uint64_t t_val = main_lde[j * lde_stride + i];
#pragma unroll
        for (int k = 0; k < K; ++k) {
            uint64_t idx = (j * K + k) * 3;
            ext3::Fe3 gamma = {gammas_tr[idx], gammas_tr[idx + 1], gammas_tr[idx + 2]};
            acc[k] = ext3::add(acc[k], ext3::mul_base(gamma, t_val));
        }
    }
    for (uint64_t j = 0; j < num_aux; ++j) {
        ext3::Fe3 t_val = {
            aux_lde[(j * 3 + 0) * lde_stride + i],
            aux_lde[(j * 3 + 1) * lde_stride + i],
            aux_lde[(j * 3 + 2) * lde_stride + i],
        };
        uint64_t trace_j = num_main + j;
#pragma unroll
        for (int k = 0; k < K; ++k) {
            uint64_t idx = (trace_j * K + k) * 3;
            ext3::Fe3 gamma = {gammas_tr[idx], gammas_tr[idx + 1], gammas_tr[idx + 2]};
            acc[k] = ext3::add(acc[k], ext3::mul(gamma, t_val));
        }
    }

    ext3::Fe3 c_h = {consts[0], consts[1], consts[2]};
    ext3::Fe3 result = ext3::mul(ext3::sub(acc_h, c_h), inv[0]);
#pragma unroll
    for (int k = 0; k < K; ++k) {
        ext3::Fe3 c_k = {consts[3 + 3 * k], consts[4 + 3 * k], consts[5 + 3 * k]};
        result = ext3::add(result, ext3::mul(ext3::sub(acc[k], c_k), inv[1 + k]));
    }
    result = ext3::canonical(result);
    deep_out[i * 3 + 0] = result.a;
    deep_out[i * 3 + 1] = result.b;
    deep_out[i * 3 + 2] = result.c;
}

#define DEEP_HOISTED(M)                                                                           \
    extern "C" __global__ void deep_composition_ext3_hoisted_m##M(                                \
        const uint64_t *__restrict__ main_lde, const uint64_t *__restrict__ aux_lde,              \
        const uint64_t *__restrict__ h_lde, uint64_t lde_stride, uint64_t num_main,               \
        uint64_t num_aux, uint64_t num_parts, uint64_t domain_size,                               \
        const uint64_t *__restrict__ consts, const uint64_t *__restrict__ gammas_h,               \
        const uint64_t *__restrict__ gammas_tr, const uint64_t *__restrict__ coset_points,        \
        const uint64_t *__restrict__ z_scalars, uint64_t *__restrict__ deep_out,                  \
        uint32_t *zero_flag) {                                                                    \
        deep_hoisted_body<M>(main_lde, aux_lde, h_lde, lde_stride, num_main, num_aux, num_parts,  \
                             domain_size, consts, gammas_h, gammas_tr, coset_points, z_scalars,   \
                             deep_out, zero_flag);                                                \
    }

DEEP_HOISTED(1)
DEEP_HOISTED(2)
DEEP_HOISTED(3)
DEEP_HOISTED(4)
