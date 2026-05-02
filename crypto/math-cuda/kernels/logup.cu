// LogUp aux-trace-build: one pair's fingerprint compute + term assembly
// in two kernels. The host orchestrates:
//   main_cols (already on device from R1)
//   + bytecode descriptor for the pair (uploaded once per pair)
//   → fingerprint kernel writes 2n ext3 fingerprints
//   → batch_inverse_ext3 (existing) inverts in place (or into output)
//   → term-assembly kernel writes n ext3 term values
//
// Wire format (packed C structs, shared with Rust serializer):
//
//   struct FingerprintOp {
//       uint8_t  kind;           // OP_PACK_* / OP_LINEAR
//       uint8_t  pad0[3];
//       uint32_t alpha_offset;   // where to multiply by α into lc
//       uint32_t start_col;      // for Pack ops: first main-trace column
//       uint32_t num_linear_terms;  // for OP_LINEAR: count of terms that follow
//       uint32_t linear_term_offset; // for OP_LINEAR: start in linear_terms[]
//       uint32_t pad1[2];        // align to 32 bytes
//   };
//
//   struct LinearTerm {
//       uint8_t  kind;    // 0 = Column signed, 1 = Column unsigned, 2 = Constant
//       uint8_t  pad[3];
//       uint32_t column;
//       int64_t  value;   // signed coefficient or signed constant
//   };
//
//   struct MultiplicityDesc {
//       uint8_t  kind;       // 0..6 mapping to Rust's Multiplicity variants
//       uint8_t  pad[3];
//       uint32_t cols[3];    // up to 3 columns (Sum3)
//       uint32_t num_linear_terms;
//       uint32_t linear_term_offset;
//   };
//
// All ops reference the same main_cols buffer and the same shared
// linear_terms buffer.

#include "goldilocks.cuh"
#include "ext3.cuh"

// Must match Rust-side `LogupOpKind`.
#define OP_PACK_DIRECT    0
#define OP_PACK_WORD2L    1
#define OP_PACK_WORD4L    2
#define OP_PACK_DWORDWL   3
#define OP_PACK_DWORDHHW  4
#define OP_PACK_DWORDWHH  5
#define OP_PACK_DWORDHL   6
#define OP_PACK_DWORDBL   7
#define OP_PACK_QUADHL    8
#define OP_PACK_QUADWL    9
#define OP_LINEAR        10

// PackingShifts (base field).
#define SHIFT_8  ((uint64_t)(1ULL << 8))
#define SHIFT_16 ((uint64_t)(1ULL << 16))
#define SHIFT_24 ((uint64_t)(1ULL << 24))

struct FingerprintOp {
    uint8_t  kind;
    uint8_t  pad0[3];
    uint32_t alpha_offset;
    uint32_t start_col;
    uint32_t num_linear_terms;
    uint32_t linear_term_offset;
    uint32_t pad1[2];
};

struct LinearTerm {
    uint8_t  kind;      // 0=Column, 2=Constant (Rust canonicalizes both into `value`)
    uint8_t  pad[3];
    uint32_t column;
    uint64_t value;     // canonical Goldilocks field element in [0, p)
};

struct MultiplicityDesc {
    uint8_t  kind;
    uint8_t  pad[3];
    uint32_t cols[3];
    uint32_t num_linear_terms;
    uint32_t linear_term_offset;
};

#define MULT_ONE      0
#define MULT_COLUMN   1
#define MULT_SUM      2
#define MULT_NEGATED  3
#define MULT_DIFF     4
#define MULT_SUM3     5
#define MULT_LINEAR   6

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t read_main(const uint64_t *main_cols,
                                               uint64_t col_stride,
                                               uint32_t col,
                                               uint64_t row) {
    // Column-major: col * col_stride + row.
    return main_cols[(uint64_t)col * col_stride + row];
}

/// Evaluate a Linear term list at `row` → base-field element.
/// `lt.value` is already a canonical Goldilocks field element in [0, p);
/// the Rust serializer is responsible for the canonicalization so the
/// device can skip sign handling entirely.
__device__ __forceinline__ uint64_t eval_linear(
    const uint64_t *main_cols,
    uint64_t col_stride,
    const LinearTerm *linear_terms,
    uint32_t num_terms,
    uint32_t offset,
    uint64_t row) {
    uint64_t result = 0;
    for (uint32_t t = 0; t < num_terms; ++t) {
        const LinearTerm &lt = linear_terms[offset + t];
        if (lt.kind == 2) {
            // Constant.
            result = goldilocks::add(result, lt.value);
        } else {
            // Column (signed or unsigned — canonical coefficient).
            uint64_t v = read_main(main_cols, col_stride, lt.column, row);
            uint64_t prod = goldilocks::mul(v, lt.value);
            result = goldilocks::add(result, prod);
        }
    }
    return result;
}

/// Apply one fingerprint op: reads main_cols[*], multiplies by the
/// appropriate alpha power(s), accumulates into `acc`.
__device__ __forceinline__ void apply_fingerprint_op(
    const uint64_t *main_cols,
    uint64_t col_stride,
    const LinearTerm *linear_terms,
    const uint64_t *alpha_powers, // ext3 interleaved: 3*max_bus_elements u64
    const FingerprintOp &op,
    uint64_t row,
    ext3::Fe3 &acc) {
    uint32_t ao = op.alpha_offset;
    #define ALPHA(i) ext3::make( \
        alpha_powers[((ao) + (i)) * 3 + 0], \
        alpha_powers[((ao) + (i)) * 3 + 1], \
        alpha_powers[((ao) + (i)) * 3 + 2])
    uint32_t c = op.start_col;

    switch (op.kind) {
        case OP_PACK_DIRECT: {
            uint64_t v = read_main(main_cols, col_stride, c, row);
            ext3::Fe3 ap = ALPHA(0);
            acc = ext3::add(acc, ext3::mul_base(ap, v));
            break;
        }
        case OP_PACK_WORD2L: {
            uint64_t v0 = read_main(main_cols, col_stride, c, row);
            uint64_t v1 = read_main(main_cols, col_stride, c + 1, row);
            uint64_t combined = goldilocks::add(v0, goldilocks::mul(v1, SHIFT_16));
            ext3::Fe3 ap = ALPHA(0);
            acc = ext3::add(acc, ext3::mul_base(ap, combined));
            break;
        }
        case OP_PACK_WORD4L: {
            uint64_t v0 = read_main(main_cols, col_stride, c, row);
            uint64_t v1 = read_main(main_cols, col_stride, c + 1, row);
            uint64_t v2 = read_main(main_cols, col_stride, c + 2, row);
            uint64_t v3 = read_main(main_cols, col_stride, c + 3, row);
            uint64_t t1 = goldilocks::mul(v1, SHIFT_8);
            uint64_t t2 = goldilocks::mul(v2, SHIFT_16);
            uint64_t t3 = goldilocks::mul(v3, SHIFT_24);
            uint64_t combined = goldilocks::add(goldilocks::add(v0, t1),
                                                goldilocks::add(t2, t3));
            ext3::Fe3 ap = ALPHA(0);
            acc = ext3::add(acc, ext3::mul_base(ap, combined));
            break;
        }
        case OP_PACK_DWORDWL: {
            uint64_t v0 = read_main(main_cols, col_stride, c, row);
            uint64_t v1 = read_main(main_cols, col_stride, c + 1, row);
            ext3::Fe3 ap0 = ALPHA(0), ap1 = ALPHA(1);
            acc = ext3::add(acc, ext3::mul_base(ap0, v0));
            acc = ext3::add(acc, ext3::mul_base(ap1, v1));
            break;
        }
        case OP_PACK_DWORDHHW: {
            // Direct + Word2L: col, col+1 -> word, col+2 -> half? No — spec: Direct + Word2L
            // columns: [direct c0, word2l c1 c2] → (c0)*α0 + (c1 + c2 << 16)*α1
            uint64_t v0 = read_main(main_cols, col_stride, c, row);
            uint64_t v1 = read_main(main_cols, col_stride, c + 1, row);
            uint64_t v2 = read_main(main_cols, col_stride, c + 2, row);
            ext3::Fe3 ap0 = ALPHA(0), ap1 = ALPHA(1);
            acc = ext3::add(acc, ext3::mul_base(ap0, v0));
            uint64_t w = goldilocks::add(v1, goldilocks::mul(v2, SHIFT_16));
            acc = ext3::add(acc, ext3::mul_base(ap1, w));
            break;
        }
        case OP_PACK_DWORDWHH: {
            uint64_t v0 = read_main(main_cols, col_stride, c, row);
            uint64_t v1 = read_main(main_cols, col_stride, c + 1, row);
            uint64_t v2 = read_main(main_cols, col_stride, c + 2, row);
            ext3::Fe3 ap0 = ALPHA(0), ap1 = ALPHA(1);
            uint64_t w = goldilocks::add(v0, goldilocks::mul(v1, SHIFT_16));
            acc = ext3::add(acc, ext3::mul_base(ap0, w));
            acc = ext3::add(acc, ext3::mul_base(ap1, v2));
            break;
        }
        case OP_PACK_DWORDHL: {
            uint64_t v0 = read_main(main_cols, col_stride, c, row);
            uint64_t v1 = read_main(main_cols, col_stride, c + 1, row);
            uint64_t v2 = read_main(main_cols, col_stride, c + 2, row);
            uint64_t v3 = read_main(main_cols, col_stride, c + 3, row);
            ext3::Fe3 ap0 = ALPHA(0), ap1 = ALPHA(1);
            uint64_t w0 = goldilocks::add(v0, goldilocks::mul(v1, SHIFT_16));
            uint64_t w1 = goldilocks::add(v2, goldilocks::mul(v3, SHIFT_16));
            acc = ext3::add(acc, ext3::mul_base(ap0, w0));
            acc = ext3::add(acc, ext3::mul_base(ap1, w1));
            break;
        }
        case OP_PACK_DWORDBL: {
            // 2× Word4L at start_col and start_col+4
            ext3::Fe3 ap0 = ALPHA(0), ap1 = ALPHA(1);
            for (int hi = 0; hi < 2; ++hi) {
                uint32_t base = c + hi * 4;
                uint64_t v0 = read_main(main_cols, col_stride, base, row);
                uint64_t v1 = read_main(main_cols, col_stride, base + 1, row);
                uint64_t v2 = read_main(main_cols, col_stride, base + 2, row);
                uint64_t v3 = read_main(main_cols, col_stride, base + 3, row);
                uint64_t t1 = goldilocks::mul(v1, SHIFT_8);
                uint64_t t2 = goldilocks::mul(v2, SHIFT_16);
                uint64_t t3 = goldilocks::mul(v3, SHIFT_24);
                uint64_t w = goldilocks::add(goldilocks::add(v0, t1),
                                             goldilocks::add(t2, t3));
                ext3::Fe3 ap = (hi == 0) ? ap0 : ap1;
                acc = ext3::add(acc, ext3::mul_base(ap, w));
            }
            break;
        }
        case OP_PACK_QUADHL: {
            // 4× Word2L at start_col, start_col+2, ..., start_col+6
            for (int k = 0; k < 4; ++k) {
                uint32_t base = c + k * 2;
                uint64_t v0 = read_main(main_cols, col_stride, base, row);
                uint64_t v1 = read_main(main_cols, col_stride, base + 1, row);
                uint64_t w = goldilocks::add(v0, goldilocks::mul(v1, SHIFT_16));
                ext3::Fe3 ap = ALPHA(k);
                acc = ext3::add(acc, ext3::mul_base(ap, w));
            }
            break;
        }
        case OP_PACK_QUADWL: {
            for (int k = 0; k < 4; ++k) {
                uint64_t v = read_main(main_cols, col_stride, c + k, row);
                ext3::Fe3 ap = ALPHA(k);
                acc = ext3::add(acc, ext3::mul_base(ap, v));
            }
            break;
        }
        case OP_LINEAR: {
            uint64_t r = eval_linear(main_cols, col_stride, linear_terms,
                                     op.num_linear_terms, op.linear_term_offset, row);
            ext3::Fe3 ap = ALPHA(0);
            acc = ext3::add(acc, ext3::mul_base(ap, r));
            break;
        }
        default:
            break;
    }
    #undef ALPHA
}

/// Compute one interaction pair's fingerprints: 2n ext3 values
/// `fp[0..n] = z - lc_a(row)`, `fp[n..2n] = z - lc_b(row)`.
extern "C" __global__ void logup_pair_fingerprint(
    const uint64_t *main_cols,    // main LDE, column-major, col_stride u64 per column
    uint64_t col_stride,
    uint64_t n,                   // trace_len
    uint64_t bus_id_a,            // base field
    uint64_t bus_id_b,
    const FingerprintOp *ops_a,   // pair A ops
    uint32_t ops_a_count,
    const FingerprintOp *ops_b,   // pair B ops
    uint32_t ops_b_count,
    const LinearTerm *linear_terms,
    const uint64_t *alpha_powers, // 3 * max_bus_elements u64
    const uint64_t *z,            // 3 u64 (ext3)
    uint64_t *fp_out) {           // 2n * 3 u64 (ext3 interleaved)
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    ext3::Fe3 zf = ext3::make(z[0], z[1], z[2]);

    // Pair A
    {
        ext3::Fe3 alpha0 = ext3::make(
            alpha_powers[0], alpha_powers[1], alpha_powers[2]);
        ext3::Fe3 lc = ext3::mul_base(alpha0, bus_id_a);
        for (uint32_t k = 0; k < ops_a_count; ++k) {
            apply_fingerprint_op(main_cols, col_stride, linear_terms,
                                 alpha_powers, ops_a[k], row, lc);
        }
        ext3::Fe3 fp = ext3::sub(zf, lc);
        fp_out[row * 3 + 0] = fp.a;
        fp_out[row * 3 + 1] = fp.b;
        fp_out[row * 3 + 2] = fp.c;
    }

    // Pair B (output at row + n)
    {
        ext3::Fe3 alpha0 = ext3::make(
            alpha_powers[0], alpha_powers[1], alpha_powers[2]);
        ext3::Fe3 lc = ext3::mul_base(alpha0, bus_id_b);
        for (uint32_t k = 0; k < ops_b_count; ++k) {
            apply_fingerprint_op(main_cols, col_stride, linear_terms,
                                 alpha_powers, ops_b[k], row, lc);
        }
        ext3::Fe3 fp = ext3::sub(zf, lc);
        uint64_t out_row = n + row;
        fp_out[out_row * 3 + 0] = fp.a;
        fp_out[out_row * 3 + 1] = fp.b;
        fp_out[out_row * 3 + 2] = fp.c;
    }
}

/// Evaluate a Multiplicity descriptor at `row` → base-field value.
__device__ __forceinline__ uint64_t eval_multiplicity(
    const uint64_t *main_cols,
    uint64_t col_stride,
    const LinearTerm *linear_terms,
    const MultiplicityDesc &m,
    uint64_t row) {
    switch (m.kind) {
        case MULT_ONE:
            return 1;
        case MULT_COLUMN:
            return read_main(main_cols, col_stride, m.cols[0], row);
        case MULT_SUM: {
            uint64_t a = read_main(main_cols, col_stride, m.cols[0], row);
            uint64_t b = read_main(main_cols, col_stride, m.cols[1], row);
            return goldilocks::add(a, b);
        }
        case MULT_NEGATED: {
            uint64_t v = read_main(main_cols, col_stride, m.cols[0], row);
            return goldilocks::sub(1, v);
        }
        case MULT_DIFF: {
            uint64_t a = read_main(main_cols, col_stride, m.cols[0], row);
            uint64_t b = read_main(main_cols, col_stride, m.cols[1], row);
            return goldilocks::sub(a, b);
        }
        case MULT_SUM3: {
            uint64_t a = read_main(main_cols, col_stride, m.cols[0], row);
            uint64_t b = read_main(main_cols, col_stride, m.cols[1], row);
            uint64_t c = read_main(main_cols, col_stride, m.cols[2], row);
            return goldilocks::add(goldilocks::add(a, b), c);
        }
        case MULT_LINEAR:
            return eval_linear(main_cols, col_stride, linear_terms,
                               m.num_linear_terms, m.linear_term_offset, row);
        default:
            return 0;
    }
}

/// Term-assembly: reads inverted fingerprints + multiplicities,
/// produces the term column.
///   term[row] = (neg_a ? -1 : 1) * mult_a(row) * inv_fp_a[row]
///             + (neg_b ? -1 : 1) * mult_b(row) * inv_fp_b[row]
/// Multiplicities are base-field, inv_fp are ext3; result is ext3.
extern "C" __global__ void logup_pair_term_assembly(
    const uint64_t *inv_fp,      // 2n * 3 u64 (ext3 interleaved)
    const uint64_t *main_cols,   // main LDE
    uint64_t col_stride,
    uint64_t n,
    const LinearTerm *linear_terms,
    const MultiplicityDesc *mult_a,  // device pointer to descriptor (1 struct)
    const MultiplicityDesc *mult_b,
    uint8_t negate_a,
    uint8_t negate_b,
    uint64_t *term_out) {        // n * 3 u64 (ext3)
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    uint64_t m_a_val = eval_multiplicity(main_cols, col_stride, linear_terms,
                                         *mult_a, row);
    uint64_t m_b_val = eval_multiplicity(main_cols, col_stride, linear_terms,
                                         *mult_b, row);

    ext3::Fe3 inv_a = ext3::make(
        inv_fp[row * 3 + 0], inv_fp[row * 3 + 1], inv_fp[row * 3 + 2]);
    uint64_t row_b = n + row;
    ext3::Fe3 inv_b = ext3::make(
        inv_fp[row_b * 3 + 0], inv_fp[row_b * 3 + 1], inv_fp[row_b * 3 + 2]);

    ext3::Fe3 ta = ext3::mul_base(inv_a, m_a_val);
    if (negate_a) ta = ext3::neg(ta);
    ext3::Fe3 tb = ext3::mul_base(inv_b, m_b_val);
    if (negate_b) tb = ext3::neg(tb);

    ext3::Fe3 t = ext3::add(ta, tb);
    term_out[row * 3 + 0] = t.a;
    term_out[row * 3 + 1] = t.b;
    term_out[row * 3 + 2] = t.c;
}

/// Single-pair variant (for the "absorbed" case with 1 interaction).
/// Computes fingerprints and term for a single interaction.
extern "C" __global__ void logup_single_fingerprint(
    const uint64_t *main_cols,
    uint64_t col_stride,
    uint64_t n,
    uint64_t bus_id,
    const FingerprintOp *ops,
    uint32_t ops_count,
    const LinearTerm *linear_terms,
    const uint64_t *alpha_powers,
    const uint64_t *z,
    uint64_t *fp_out) {          // n * 3 u64
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    ext3::Fe3 zf = ext3::make(z[0], z[1], z[2]);
    ext3::Fe3 alpha0 = ext3::make(
        alpha_powers[0], alpha_powers[1], alpha_powers[2]);
    ext3::Fe3 lc = ext3::mul_base(alpha0, bus_id);
    for (uint32_t k = 0; k < ops_count; ++k) {
        apply_fingerprint_op(main_cols, col_stride, linear_terms,
                             alpha_powers, ops[k], row, lc);
    }
    ext3::Fe3 fp = ext3::sub(zf, lc);
    fp_out[row * 3 + 0] = fp.a;
    fp_out[row * 3 + 1] = fp.b;
    fp_out[row * 3 + 2] = fp.c;
}

extern "C" __global__ void logup_single_term_assembly(
    const uint64_t *inv_fp,       // n * 3 u64
    const uint64_t *main_cols,
    uint64_t col_stride,
    uint64_t n,
    const LinearTerm *linear_terms,
    const MultiplicityDesc *mult,
    uint8_t negate,
    uint64_t *term_out) {
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    uint64_t m = eval_multiplicity(main_cols, col_stride, linear_terms,
                                   *mult, row);
    ext3::Fe3 inv = ext3::make(
        inv_fp[row * 3 + 0], inv_fp[row * 3 + 1], inv_fp[row * 3 + 2]);
    ext3::Fe3 t = ext3::mul_base(inv, m);
    if (negate) t = ext3::neg(t);
    term_out[row * 3 + 0] = t.a;
    term_out[row * 3 + 1] = t.b;
    term_out[row * 3 + 2] = t.c;
}
