// LogUp aux build: fingerprint kernel.
//
// One ext3 fingerprint per (interaction, row):
//   lc = bus_id + sum_e alpha_powers[alpha_idx(e)] * base_e
//   base_e = const_e + sum_t coef_t * main_col[col_t][row]   (Goldilocks base)
//   fp = z - lc
// Mirrors stark::logup_gpu::eval_fingerprint byte for byte.
//
// Layouts:
//   main: column-major, main[col * num_rows + row].
//   descriptor (CSR): interactions -> elements -> terms.
//   alpha_powers: ext3 interleaved, 3 limbs each.
//   out: ext3 interleaved, out[(k*num_rows + row)*3 + {0,1,2}].

#include "ext3.cuh"

using namespace ext3;

extern "C" __global__ void logup_fingerprint_ext3(
    const uint64_t *__restrict__ main,
    uint32_t num_rows,
    uint32_t num_interactions,
    const uint64_t *__restrict__ bus_ids,
    const uint32_t *__restrict__ elem_offsets,
    const uint32_t *__restrict__ elem_alpha_idx,
    const uint64_t *__restrict__ elem_const,
    const uint32_t *__restrict__ term_offsets,
    const uint64_t *__restrict__ term_coef,
    const uint32_t *__restrict__ term_col,
    const uint64_t *__restrict__ alpha_powers,
    uint64_t z0, uint64_t z1, uint64_t z2,
    uint64_t *__restrict__ out) {
  uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  uint64_t total = (uint64_t)num_interactions * (uint64_t)num_rows;
  if (tid >= total)
    return;

  uint32_t k = (uint32_t)(tid / num_rows);
  uint32_t row = (uint32_t)(tid % num_rows);

  Fe3 lc = make(bus_ids[k], 0, 0);

  uint32_t e_hi = elem_offsets[k + 1];
  for (uint32_t e = elem_offsets[k]; e < e_hi; ++e) {
    uint64_t base = elem_const[e];
    uint32_t t_hi = term_offsets[e + 1];
    for (uint32_t t = term_offsets[e]; t < t_hi; ++t) {
      uint64_t col_val = main[(uint64_t)term_col[t] * (uint64_t)num_rows + row];
      base = goldilocks::add(base, goldilocks::mul(term_coef[t], col_val));
    }
    uint32_t ai = elem_alpha_idx[e];
    Fe3 a = make(alpha_powers[ai * 3 + 0], alpha_powers[ai * 3 + 1],
                 alpha_powers[ai * 3 + 2]);
    lc = add(lc, mul_base(a, base));
  }

  Fe3 z = make(z0, z1, z2);
  Fe3 fp = sub(z, lc);
  uint64_t o = tid * 3;
  out[o + 0] = fp.a;
  out[o + 1] = fp.b;
  out[o + 2] = fp.c;
}

// Term combine: one ext3 per (output column, row):
//   term = sum_{k in col} signed_mult_k(row) * reciprocal_k[row]
// signed_mult_k = mult_const[k] + sum_t mult_coef_t * main_col[col_t][row]
// (receiver sign already folded into the coefficients by the builder).
// reciprocals: ext3 interleaved, [(k*num_rows + row)*3 + limb].
// out: ext3 interleaved, [(col*num_rows + row)*3 + limb].
extern "C" __global__ void logup_term_ext3(
    const uint64_t *__restrict__ main,
    uint32_t num_rows,
    const uint64_t *__restrict__ reciprocals,
    uint32_t num_out_cols,
    const uint32_t *__restrict__ out_col_offsets,
    const uint32_t *__restrict__ out_col_interactions,
    const uint64_t *__restrict__ mult_const,
    const uint32_t *__restrict__ mult_term_offsets,
    const uint64_t *__restrict__ mult_term_coef,
    const uint32_t *__restrict__ mult_term_col,
    uint64_t *__restrict__ out) {
  uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  uint64_t total = (uint64_t)num_out_cols * (uint64_t)num_rows;
  if (tid >= total)
    return;

  uint32_t col = (uint32_t)(tid / num_rows);
  uint32_t row = (uint32_t)(tid % num_rows);

  Fe3 term = zero();
  uint32_t ki_hi = out_col_offsets[col + 1];
  for (uint32_t ki = out_col_offsets[col]; ki < ki_hi; ++ki) {
    uint32_t k = out_col_interactions[ki];

    uint64_t m = mult_const[k];
    uint32_t t_hi = mult_term_offsets[k + 1];
    for (uint32_t t = mult_term_offsets[k]; t < t_hi; ++t) {
      uint64_t col_val = main[(uint64_t)mult_term_col[t] * (uint64_t)num_rows + row];
      m = goldilocks::add(m, goldilocks::mul(mult_term_coef[t], col_val));
    }

    uint64_t ro = ((uint64_t)k * num_rows + row) * 3;
    Fe3 r = make(reciprocals[ro], reciprocals[ro + 1], reciprocals[ro + 2]);
    term = add(term, mul_base(r, m));
  }

  uint64_t o = tid * 3;
  out[o + 0] = term.a;
  out[o + 1] = term.b;
  out[o + 2] = term.c;
}

// ===========================================================================
// Accumulated column (K4): running sum of the term columns, on device.
//   row_sum[i] = sum over all term columns of term[col][i]
//   S = inclusive prefix scan of row_sum ;  L = S[n-1] ;  offset = L / N
//   acc[i] = S[i] - (i+1) * offset          (matches build_accumulated_column)
// Additive 3-phase Hillis-Steele scan (mirrors inverse.cu, add not mul).
// ===========================================================================

#define LOGUP_BLK 256

// row_sum[i] = sum_c term[(c*num_rows + i)] over all num_cols term columns.
extern "C" __global__ void logup_row_sum_ext3(
    const uint64_t *__restrict__ terms, uint32_t num_cols, uint32_t num_rows,
    uint64_t *__restrict__ row_sum) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= num_rows)
    return;
  Fe3 s = zero();
  for (uint32_t c = 0; c < num_cols; ++c) {
    uint64_t o = ((uint64_t)c * num_rows + i) * 3;
    s = add(s, make(terms[o], terms[o + 1], terms[o + 2]));
  }
  row_sum[i * 3] = s.a;
  row_sum[i * 3 + 1] = s.b;
  row_sum[i * 3 + 2] = s.c;
}

// Per-block inclusive additive scan; writes block totals (last valid element).
extern "C" __global__ void logup_scan_block_add_ext3(
    const uint64_t *__restrict__ input, uint64_t n,
    uint64_t *__restrict__ scan_out, uint64_t *__restrict__ block_totals) {
  __shared__ Fe3 sh[LOGUP_BLK];
  uint32_t tid = threadIdx.x;
  uint64_t gid = blockIdx.x * (uint64_t)LOGUP_BLK + tid;
  Fe3 v = (gid < n) ? make(input[gid * 3], input[gid * 3 + 1], input[gid * 3 + 2])
                    : zero();
  sh[tid] = v;
  __syncthreads();
  for (uint32_t off = 1; off < LOGUP_BLK; off <<= 1) {
    Fe3 t = (tid >= off) ? sh[tid - off] : zero();
    __syncthreads();
    if (tid >= off)
      sh[tid] = add(sh[tid], t);
    __syncthreads();
  }
  if (gid < n) {
    uint64_t o = gid * 3;
    scan_out[o] = sh[tid].a;
    scan_out[o + 1] = sh[tid].b;
    scan_out[o + 2] = sh[tid].c;
  }
  uint64_t block_end = (blockIdx.x + 1) * (uint64_t)LOGUP_BLK;
  uint32_t last = (block_end <= n)
                      ? (LOGUP_BLK - 1)
                      : (uint32_t)(n - blockIdx.x * (uint64_t)LOGUP_BLK - 1);
  if (tid == last) {
    uint64_t b = blockIdx.x * 3;
    block_totals[b] = sh[tid].a;
    block_totals[b + 1] = sh[tid].b;
    block_totals[b + 2] = sh[tid].c;
  }
}

// Phase 3: block b>0 adds the scanned prefix of preceding block totals.
extern "C" __global__ void logup_apply_offsets_add_ext3(
    uint64_t *__restrict__ scan_inout, uint64_t n,
    const uint64_t *__restrict__ block_totals_scanned) {
  if (blockIdx.x == 0)
    return;
  uint64_t gid = blockIdx.x * (uint64_t)LOGUP_BLK + threadIdx.x;
  if (gid >= n)
    return;
  uint64_t ob = (blockIdx.x - 1) * 3;
  Fe3 off = make(block_totals_scanned[ob], block_totals_scanned[ob + 1],
                 block_totals_scanned[ob + 2]);
  uint64_t o = gid * 3;
  Fe3 v = add(make(scan_inout[o], scan_inout[o + 1], scan_inout[o + 2]), off);
  scan_inout[o] = v.a;
  scan_inout[o + 1] = v.b;
  scan_inout[o + 2] = v.c;
}

// acc[i] = scan[i] - (i+1) * (L * inv_N), L = scan[n-1]. inv_N is ext3 (1/N).
extern "C" __global__ void logup_finalize_accum_ext3(
    const uint64_t *__restrict__ scan, uint64_t n, uint64_t inv0, uint64_t inv1,
    uint64_t inv2, uint64_t *__restrict__ acc) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  uint64_t lo = (n - 1) * 3;
  Fe3 L = make(scan[lo], scan[lo + 1], scan[lo + 2]);
  Fe3 offset = mul(L, make(inv0, inv1, inv2));
  Fe3 s = make(scan[i * 3], scan[i * 3 + 1], scan[i * 3 + 2]);
  Fe3 a = sub(s, mul_base(offset, i + 1));
  acc[i * 3] = a.a;
  acc[i * 3 + 1] = a.b;
  acc[i * 3 + 2] = a.c;
}

// Assemble the row-major aux trace buffer from the resident committed term
// columns + the accumulated column:
//   aux[row * num_aux_cols + col] = committed[col][row]   (col < num_committed)
//                                 = accumulated[row]       (col == num_committed)
// terms layout is [col][row] (column-major); aux is row-major [row][col].
extern "C" __global__ void logup_assemble_aux_ext3(
    const uint64_t *__restrict__ committed, uint32_t num_committed,
    const uint64_t *__restrict__ accumulated, uint32_t num_rows,
    uint64_t *__restrict__ aux) {
  uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
  if (i >= num_rows)
    return;
  uint32_t num_aux_cols = num_committed + 1;
  for (uint32_t col = 0; col < num_committed; ++col) {
    uint64_t src = ((uint64_t)col * num_rows + i) * 3;
    uint64_t dst = ((uint64_t)i * num_aux_cols + col) * 3;
    aux[dst] = committed[src];
    aux[dst + 1] = committed[src + 1];
    aux[dst + 2] = committed[src + 2];
  }
  uint64_t asrc = i * 3;
  uint64_t adst = ((uint64_t)i * num_aux_cols + num_committed) * 3;
  aux[adst] = accumulated[asrc];
  aux[adst + 1] = accumulated[asrc + 1];
  aux[adst + 2] = accumulated[asrc + 2];
}
