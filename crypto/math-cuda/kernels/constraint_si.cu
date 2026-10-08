// Bounded-slot composition interpreter.
//
// Evaluates a constraint program lowered by
// `crypto/stark/src/constraint_ir/budgeted.rs` (roots in index order, each
// root's cone on demand, leaves read as operands, a fixed budget of words a
// row) and fuses the composition accumulation, exactly like
// `constraint_composition_kernel` (constraint_interp.cu):
//
//   H(row) = z_inv[row % z_len] * Σ_c beta[c] * C_c(row)
//          + Σ_b z_b_inv[b*num_rows + row] * beta_b * (trace_b - value_b)
//
// What differs is where values live. `constraint_composition_kernel` keeps a
// per-thread slot file of thousands of words in global memory (every root is
// pinned until the end of the walk); here a row's values fit in `num_words`
// words, kept in shared memory (`si_smem_*`) or in a per-thread local array
// (`si_local_*`, cached in L1). With no slot file in DRAM the grid covers
// every row (no thread cap).
//
// ⛔ BIT-IDENTICAL BY CONSTRUCTION. Every step is the call `eval_program_row`
// makes for its node's op, operand dims and result dim, with the operands in
// IR order — including the mixed base/ext shortcuts and the literal
// `sub(0, ·)`. A recomputed value is the same function of the same leaves. The
// accumulation adds root `c` after roots `0..c` with `mul_base` for a base
// root and `mul(beta, ·)` for an ext root. The tail is the interpreter's own.
// The lowering's validator (`budgeted::validate`) replays every program
// symbolically; `tools/si_host_check.cpp` runs this source as host C++
// against `constraint_composition_kernel` limb for limb.
//
// A step is one `uint4` {op, a, b, dst}. Operands are `kind << 29 | payload`:
//   BSLOT word · ESLOT first word · MAIN/AUX col | offset << 20 · BCONST index
//   · EUNI index into the ext uniform table (the program's ext constants, the
//   RAP challenges it reads, the alpha powers it reads, the table offset).
// Op tags and operand kinds MUST stay in sync with budgeted.rs.

#include "goldilocks.cuh"
#include "ext3.cuh"

using ext3::Fe3;

#define SI_BADD 0u
#define SI_BSUB 1u
#define SI_BMUL 2u
#define SI_BNEG 3u
#define SI_EADD 4u
#define SI_ESUB 5u
#define SI_EMUL 6u
#define SI_EADD_BX 7u
#define SI_ESUB_BX 8u
#define SI_EMUL_BX 9u
#define SI_EADD_XB 10u
#define SI_ESUB_XB 11u
#define SI_EMUL_XB 12u
#define SI_ENEG 13u
#define SI_EMBED 14u
#define SI_ACC_B 15u
#define SI_ACC_E 16u

#define SIK_SHIFT 29u
#define SIK_PAYLOAD_MASK 0x1FFFFFFFu
#define SIK_BSLOT 0u
#define SIK_ESLOT 1u
#define SIK_MAIN 2u
#define SIK_AUX 3u
#define SIK_BCONST 4u
#define SIK_EUNI 5u
#define SI_COL_OFFSET_SHIFT 20u
#define SI_COL_MASK 0xFFFFFu

// The staged variants' tile of steps, and who copies which step of it (the
// host parity check builds every thread as a whole block's copier).
#define SI_TILE 256u
#ifndef SI_STAGE_TID
#define SI_STAGE_TID threadIdx.x
#define SI_STAGE_STRIDE blockDim.x
#endif

// The prefetching variants look this many steps ahead in the staged tile and
// prefetch the trace cells that step will read into L1, so the read at the
// step itself hits L1 instead of waiting on L2 or DRAM. A hint only: no value
// depends on it.
#define SI_LOOKAHEAD 8u
#ifndef SI_PREFETCH_L1
#define SI_PREFETCH_L1(p) asm volatile("prefetch.global.L1 [%0];" ::"l"(p))
#endif

// Specialized opcodes (budgeted.rs `SI_FAST`): a generic op whose operand kinds
// the opcode fixes, so the handler loads them directly.
#define SI_F_BADD_SS 32u
#define SI_F_BSUB_SS 33u
#define SI_F_BMUL_SS 34u
#define SI_F_BMUL_MC 35u
#define SI_F_BMUL_CM 36u
#define SI_F_BMUL_SM 37u
#define SI_F_BMUL_MS 38u
#define SI_F_BADD_MS 39u
#define SI_F_BADD_CS 40u
#define SI_F_EADD_SS 41u
#define SI_F_ESUB_SS 42u
#define SI_F_EMUL_SS 43u
#define SI_F_EMULBX_MU 44u
#define SI_F_EMULBX_SU 45u
#define SI_F_EMULBX_MS 46u
#define SI_F_EMULBX_SS 47u
#define SI_F_ACCB_S 48u
#define SI_F_ACCE_S 49u
#define SI_F_ESUB_SA 50u
#define SI_F_EMUL_AS 51u

// Slots in dynamic shared memory: word `w` of the thread's row `j` at
// `smem[(w * R + j) * blockDim.x + threadIdx.x]` (consecutive threads,
// consecutive words: no bank conflict).
template <int R> struct SmemSlots {
    uint64_t *base;
    uint32_t stride;
    __device__ __forceinline__ uint64_t &at(uint32_t w, int j) {
        return base[(uint64_t)(w * R + j) * stride];
    }
};

// Slots in a per-thread local array (local memory, cached in L1).
template <int R, int MAXW> struct LocalSlots {
    uint64_t v[MAXW * R];
    __device__ __forceinline__ uint64_t &at(uint32_t w, int j) { return v[w * R + j]; }
};

// The per-launch read-only inputs a step's operands resolve against.
struct SiInputs {
    const uint64_t *base_consts;
    const Fe3 *uni;
    const uint64_t *main;
    uint64_t main_stride;
    const uint64_t *aux;
    uint64_t aux_stride;
};

template <int R, class S>
__device__ __forceinline__ uint64_t si_base(S &s, const SiInputs &in, uint32_t e, int j,
                                            const uint64_t *r0, const uint64_t *r1) {
    uint32_t p = e & SIK_PAYLOAD_MASK;
    switch (e >> SIK_SHIFT) {
    case SIK_BSLOT:
        return s.at(p, j);
    case SIK_MAIN: {
        uint64_t r = (p >> SI_COL_OFFSET_SHIFT) ? r1[j] : r0[j];
        return in.main[(uint64_t)(p & SI_COL_MASK) * in.main_stride + r];
    }
    default: // SIK_BCONST
        return in.base_consts[p];
    }
}

// Any operand as ext3; base values embed as {x, 0, 0}.
template <int R, class S>
__device__ __forceinline__ Fe3 si_ext(S &s, const SiInputs &in, uint32_t e, int j,
                                      const uint64_t *r0, const uint64_t *r1) {
    uint32_t p = e & SIK_PAYLOAD_MASK;
    switch (e >> SIK_SHIFT) {
    case SIK_ESLOT:
        return ext3::make(s.at(p, j), s.at(p + 1, j), s.at(p + 2, j));
    case SIK_AUX: {
        uint64_t r = (p >> SI_COL_OFFSET_SHIFT) ? r1[j] : r0[j];
        uint64_t c = (uint64_t)(p & SI_COL_MASK) * 3;
        return ext3::make(in.aux[c * in.aux_stride + r], in.aux[(c + 1) * in.aux_stride + r],
                          in.aux[(c + 2) * in.aux_stride + r]);
    }
    case SIK_EUNI:
        return in.uni[p];
    default:
        return ext3::make(si_base<R>(s, in, e, j, r0, r1), 0, 0);
    }
}

// Fixed-kind operand loads for the specialized handlers.
template <class S> __device__ __forceinline__ uint64_t ld_bs(S &s, uint32_t e, int j) {
    return s.at(e & SIK_PAYLOAD_MASK, j);
}
template <class S> __device__ __forceinline__ Fe3 ld_es(S &s, uint32_t e, int j) {
    uint32_t p = e & SIK_PAYLOAD_MASK;
    return ext3::make(s.at(p, j), s.at(p + 1, j), s.at(p + 2, j));
}
__device__ __forceinline__ uint64_t ld_main(const SiInputs &in, uint32_t e, int j, const uint64_t *r0,
                                            const uint64_t *r1) {
    uint32_t p = e & SIK_PAYLOAD_MASK;
    uint64_t r = (p >> SI_COL_OFFSET_SHIFT) ? r1[j] : r0[j];
    return in.main[(uint64_t)(p & SI_COL_MASK) * in.main_stride + r];
}
__device__ __forceinline__ Fe3 ld_aux(const SiInputs &in, uint32_t e, int j, const uint64_t *r0,
                                      const uint64_t *r1) {
    uint32_t p = e & SIK_PAYLOAD_MASK;
    uint64_t r = (p >> SI_COL_OFFSET_SHIFT) ? r1[j] : r0[j];
    uint64_t c = (uint64_t)(p & SI_COL_MASK) * 3;
    return ext3::make(in.aux[c * in.aux_stride + r], in.aux[(c + 1) * in.aux_stride + r],
                      in.aux[(c + 2) * in.aux_stride + r]);
}
__device__ __forceinline__ uint64_t ld_bc(const SiInputs &in, uint32_t e) {
    return in.base_consts[e & SIK_PAYLOAD_MASK];
}
__device__ __forceinline__ Fe3 ld_eu(const SiInputs &in, uint32_t e) {
    return in.uni[e & SIK_PAYLOAD_MASK];
}

template <class S>
__device__ __forceinline__ void si_put(S &s, uint32_t w, int j, const Fe3 &v) {
    s.at(w, j) = v.a;
    s.at(w + 1, j) = v.b;
    s.at(w + 2, j) = v.c;
}

// One step for the thread's R rows: the op's call, the operands loaded by
// their kinds, the result written to its words (or added to the sum).
template <int R, class S>
__device__ __forceinline__ void si_step(S &s, const SiInputs &in, const uint4 st,
                                        const Fe3 *__restrict__ beta, Fe3 *sum, const uint64_t *r0,
                                        const uint64_t *r1) {
    const uint32_t a = st.y, b = st.z, d = st.w;
    switch (st.x) {
    case SI_BADD:
#pragma unroll
        for (int j = 0; j < R; j++) {
            uint64_t x = si_base<R>(s, in, a, j, r0, r1), y = si_base<R>(s, in, b, j, r0, r1);
            s.at(d, j) = goldilocks::add(x, y);
        }
        break;
    case SI_BSUB:
#pragma unroll
        for (int j = 0; j < R; j++) {
            uint64_t x = si_base<R>(s, in, a, j, r0, r1), y = si_base<R>(s, in, b, j, r0, r1);
            s.at(d, j) = goldilocks::sub(x, y);
        }
        break;
    case SI_BMUL:
#pragma unroll
        for (int j = 0; j < R; j++) {
            uint64_t x = si_base<R>(s, in, a, j, r0, r1), y = si_base<R>(s, in, b, j, r0, r1);
            s.at(d, j) = goldilocks::mul(x, y);
        }
        break;
    case SI_BNEG:
#pragma unroll
        for (int j = 0; j < R; j++) {
            s.at(d, j) = goldilocks::neg(si_base<R>(s, in, a, j, r0, r1));
        }
        break;
    case SI_EADD:
#pragma unroll
        for (int j = 0; j < R; j++) {
            Fe3 x = si_ext<R>(s, in, a, j, r0, r1), y = si_ext<R>(s, in, b, j, r0, r1);
            si_put(s, d, j, ext3::add(x, y));
        }
        break;
    case SI_ESUB:
#pragma unroll
        for (int j = 0; j < R; j++) {
            Fe3 x = si_ext<R>(s, in, a, j, r0, r1), y = si_ext<R>(s, in, b, j, r0, r1);
            si_put(s, d, j, ext3::sub(x, y));
        }
        break;
    case SI_EMUL:
#pragma unroll
        for (int j = 0; j < R; j++) {
            Fe3 x = si_ext<R>(s, in, a, j, r0, r1), y = si_ext<R>(s, in, b, j, r0, r1);
            si_put(s, d, j, ext3::mul(x, y));
        }
        break;
    case SI_EADD_BX:
        // {x,0,0} + y = {add(x,y.a), y.b, y.c} (add(0,v) == v).
#pragma unroll
        for (int j = 0; j < R; j++) {
            uint64_t x = si_base<R>(s, in, a, j, r0, r1);
            Fe3 y = si_ext<R>(s, in, b, j, r0, r1);
            si_put(s, d, j, ext3::make(goldilocks::add(x, y.a), y.b, y.c));
        }
        break;
    case SI_ESUB_BX:
        // {x,0,0} - y: sub(0, ·) kept literal (not bitwise neg).
#pragma unroll
        for (int j = 0; j < R; j++) {
            uint64_t x = si_base<R>(s, in, a, j, r0, r1);
            Fe3 y = si_ext<R>(s, in, b, j, r0, r1);
            si_put(s, d, j,
                   ext3::make(goldilocks::sub(x, y.a), goldilocks::sub(0, y.b),
                              goldilocks::sub(0, y.c)));
        }
        break;
    case SI_EMUL_BX:
#pragma unroll
        for (int j = 0; j < R; j++) {
            uint64_t x = si_base<R>(s, in, a, j, r0, r1);
            Fe3 y = si_ext<R>(s, in, b, j, r0, r1);
            si_put(s, d, j, ext3::mul_base(y, x));
        }
        break;
    case SI_EADD_XB:
#pragma unroll
        for (int j = 0; j < R; j++) {
            Fe3 x = si_ext<R>(s, in, a, j, r0, r1);
            uint64_t y = si_base<R>(s, in, b, j, r0, r1);
            si_put(s, d, j, ext3::make(goldilocks::add(x.a, y), x.b, x.c));
        }
        break;
    case SI_ESUB_XB:
#pragma unroll
        for (int j = 0; j < R; j++) {
            Fe3 x = si_ext<R>(s, in, a, j, r0, r1);
            uint64_t y = si_base<R>(s, in, b, j, r0, r1);
            si_put(s, d, j, ext3::make(goldilocks::sub(x.a, y), x.b, x.c));
        }
        break;
    case SI_EMUL_XB:
#pragma unroll
        for (int j = 0; j < R; j++) {
            Fe3 x = si_ext<R>(s, in, a, j, r0, r1);
            uint64_t y = si_base<R>(s, in, b, j, r0, r1);
            si_put(s, d, j, ext3::mul_base(x, y));
        }
        break;
    case SI_ENEG:
#pragma unroll
        for (int j = 0; j < R; j++) {
            si_put(s, d, j, ext3::neg(si_ext<R>(s, in, a, j, r0, r1)));
        }
        break;
    case SI_EMBED:
#pragma unroll
        for (int j = 0; j < R; j++) {
            si_put(s, d, j, si_ext<R>(s, in, a, j, r0, r1));
        }
        break;
    case SI_ACC_B: {
        const Fe3 c = beta[b];
#pragma unroll
        for (int j = 0; j < R; j++) {
            sum[j] = ext3::add(sum[j], ext3::mul_base(c, si_base<R>(s, in, a, j, r0, r1)));
        }
        break;
    }
    case SI_ACC_E: {
        const Fe3 c = beta[b];
#pragma unroll
        for (int j = 0; j < R; j++) {
            sum[j] = ext3::add(sum[j], ext3::mul(c, si_ext<R>(s, in, a, j, r0, r1)));
        }
        break;
    }
    // Specialized: the generic op above, with the operand kinds fixed.
#define SI_FOR_ROWS _Pragma("unroll") for (int j = 0; j < R; j++)
    case SI_F_BADD_SS:
        SI_FOR_ROWS s.at(d, j) = goldilocks::add(ld_bs(s, a, j), ld_bs(s, b, j));
        break;
    case SI_F_BSUB_SS:
        SI_FOR_ROWS s.at(d, j) = goldilocks::sub(ld_bs(s, a, j), ld_bs(s, b, j));
        break;
    case SI_F_BMUL_SS:
        SI_FOR_ROWS s.at(d, j) = goldilocks::mul(ld_bs(s, a, j), ld_bs(s, b, j));
        break;
    case SI_F_BMUL_MC:
        SI_FOR_ROWS s.at(d, j) = goldilocks::mul(ld_main(in, a, j, r0, r1), ld_bc(in, b));
        break;
    case SI_F_BMUL_CM:
        SI_FOR_ROWS s.at(d, j) = goldilocks::mul(ld_bc(in, a), ld_main(in, b, j, r0, r1));
        break;
    case SI_F_BMUL_SM:
        SI_FOR_ROWS s.at(d, j) = goldilocks::mul(ld_bs(s, a, j), ld_main(in, b, j, r0, r1));
        break;
    case SI_F_BMUL_MS:
        SI_FOR_ROWS s.at(d, j) = goldilocks::mul(ld_main(in, a, j, r0, r1), ld_bs(s, b, j));
        break;
    case SI_F_BADD_MS:
        SI_FOR_ROWS s.at(d, j) = goldilocks::add(ld_main(in, a, j, r0, r1), ld_bs(s, b, j));
        break;
    case SI_F_BADD_CS:
        SI_FOR_ROWS s.at(d, j) = goldilocks::add(ld_bc(in, a), ld_bs(s, b, j));
        break;
    case SI_F_EADD_SS:
        SI_FOR_ROWS si_put(s, d, j, ext3::add(ld_es(s, a, j), ld_es(s, b, j)));
        break;
    case SI_F_ESUB_SS:
        SI_FOR_ROWS si_put(s, d, j, ext3::sub(ld_es(s, a, j), ld_es(s, b, j)));
        break;
    case SI_F_EMUL_SS:
        SI_FOR_ROWS si_put(s, d, j, ext3::mul(ld_es(s, a, j), ld_es(s, b, j)));
        break;
    case SI_F_EMULBX_MU:
        SI_FOR_ROWS si_put(s, d, j, ext3::mul_base(ld_eu(in, b), ld_main(in, a, j, r0, r1)));
        break;
    case SI_F_EMULBX_SU:
        SI_FOR_ROWS si_put(s, d, j, ext3::mul_base(ld_eu(in, b), ld_bs(s, a, j)));
        break;
    case SI_F_EMULBX_MS:
        SI_FOR_ROWS si_put(s, d, j, ext3::mul_base(ld_es(s, b, j), ld_main(in, a, j, r0, r1)));
        break;
    case SI_F_EMULBX_SS:
        SI_FOR_ROWS si_put(s, d, j, ext3::mul_base(ld_es(s, b, j), ld_bs(s, a, j)));
        break;
    case SI_F_ACCB_S: {
        const Fe3 c = beta[b];
        SI_FOR_ROWS sum[j] = ext3::add(sum[j], ext3::mul_base(c, ld_bs(s, a, j)));
        break;
    }
    case SI_F_ACCE_S: {
        const Fe3 c = beta[b];
        SI_FOR_ROWS sum[j] = ext3::add(sum[j], ext3::mul(c, ld_es(s, a, j)));
        break;
    }
    case SI_F_ESUB_SA:
        SI_FOR_ROWS si_put(s, d, j, ext3::sub(ld_es(s, a, j), ld_aux(in, b, j, r0, r1)));
        break;
    case SI_F_EMUL_AS:
        SI_FOR_ROWS si_put(s, d, j, ext3::mul(ld_aux(in, a, j, r0, r1), ld_es(s, b, j)));
        break;
#undef SI_FOR_ROWS
    default:
        break;
    }
}

// Prefetch the trace cells an encoded operand will read (main and aux
// columns; every other kind reads nothing from the trace). An accumulation's
// `b` (a root index) and a unary op's unused `b` carry kind 0 and are skipped.
template <int R>
__device__ __forceinline__ void si_prefetch(const SiInputs &in, uint32_t e, const uint64_t *r0,
                                            const uint64_t *r1) {
    const uint32_t kind = e >> SIK_SHIFT;
    if (kind != SIK_MAIN && kind != SIK_AUX) {
        return;
    }
    const uint32_t p = e & SIK_PAYLOAD_MASK;
    const uint64_t col = p & SI_COL_MASK;
#pragma unroll
    for (int j = 0; j < R; j++) {
        const uint64_t r = (p >> SI_COL_OFFSET_SHIFT) ? r1[j] : r0[j];
        if (kind == SIK_MAIN) {
            SI_PREFETCH_L1(&in.main[col * in.main_stride + r]);
        } else {
            SI_PREFETCH_L1(&in.aux[(3 * col) * in.aux_stride + r]);
            SI_PREFETCH_L1(&in.aux[(3 * col + 1) * in.aux_stride + r]);
            SI_PREFETCH_L1(&in.aux[(3 * col + 2) * in.aux_stride + r]);
        }
    }
}

// One thread's walk over the program for its R rows (`row[j]`, with their
// frame-offset-1 rows `r1[j]`), then the interpreter's tail. Rows past the end
// are walked on a clamped row and not written.
// STAGE: 0 the steps from global memory, 1 staged through shared memory, 2
// staged and the trace cells of the step SI_LOOKAHEAD ahead prefetched.
template <int R, int STAGE, class S>
__device__ __forceinline__ void si_rows(
    S &s, const SiInputs &in, const uint4 *__restrict__ prog, uint32_t num_steps,
    const Fe3 *__restrict__ beta, const uint64_t *row, const bool *valid, uint64_t next_step,
    uint64_t num_rows, Fe3 *__restrict__ d_h, const uint64_t *__restrict__ d_z_inv, uint64_t z_len,
    uint64_t num_boundary, const uint64_t *__restrict__ d_b_col,
    const uint64_t *__restrict__ d_b_is_aux, const Fe3 *__restrict__ d_b_value,
    const Fe3 *__restrict__ d_b_beta, const uint64_t *__restrict__ d_b_z_inv) {
    uint64_t r0[R], r1[R];
#pragma unroll
    for (int j = 0; j < R; j++) {
        r0[j] = row[j];
        uint64_t r = row[j] + next_step;
        r1[j] = r >= num_rows ? r - num_rows : r;
    }
    Fe3 sum[R];
#pragma unroll
    for (int j = 0; j < R; j++) {
        sum[j] = ext3::zero();
    }
    if constexpr (STAGE != 0) {
        // The steps tiled through shared memory: one cooperative copy a tile,
        // then every warp reads its steps there (a broadcast) instead of from
        // L2. Every thread of the block walks the same number of tiles (the
        // grid-stride loop is uniform), so the barriers are reached by all.
        __shared__ uint4 si_tile[SI_TILE];
        for (uint32_t base = 0; base < num_steps; base += SI_TILE) {
            const uint32_t n = num_steps - base < SI_TILE ? num_steps - base : SI_TILE;
            __syncthreads();
            for (uint32_t k = SI_STAGE_TID; k < n; k += SI_STAGE_STRIDE) {
                si_tile[k] = prog[base + k];
            }
            __syncthreads();
            for (uint32_t i = 0; i < n; i++) {
                if constexpr (STAGE == 2) {
                    if (i + SI_LOOKAHEAD < n) {
                        const uint4 ahead = si_tile[i + SI_LOOKAHEAD];
                        si_prefetch<R>(in, ahead.y, r0, r1);
                        si_prefetch<R>(in, ahead.z, r0, r1);
                    }
                }
                si_step<R>(s, in, si_tile[i], beta, sum, r0, r1);
            }
        }
    } else {
        // The program is padded with one step, so the prefetch never reads past it.
        uint4 next = __ldg(prog);
        for (uint32_t pc = 0; pc < num_steps; pc++) {
            uint4 st = next;
            next = __ldg(prog + pc + 1);
            si_step<R>(s, in, st, beta, sum, r0, r1);
        }
    }
    // The interpreter's tail, verbatim.
#pragma unroll
    for (int j = 0; j < R; j++) {
        if (!valid[j]) {
            continue;
        }
        const uint64_t rw = row[j];
        Fe3 h = ext3::mul_base(sum[j], d_z_inv[rw % z_len]);
        for (uint64_t bi = 0; bi < num_boundary; bi++) {
            uint64_t col = d_b_col[bi];
            Fe3 tcell;
            if (d_b_is_aux[bi] != 0) {
                uint64_t base = col * 3;
                tcell = ext3::make(in.aux[(base + 0) * in.aux_stride + rw],
                                   in.aux[(base + 1) * in.aux_stride + rw],
                                   in.aux[(base + 2) * in.aux_stride + rw]);
            } else {
                tcell = ext3::make(in.main[col * in.main_stride + rw], 0, 0);
            }
            Fe3 bp = ext3::sub(tcell, d_b_value[bi]);
            Fe3 zb = ext3::mul_base(d_b_beta[bi], d_b_z_inv[bi * num_rows + rw]);
            h = ext3::add(h, ext3::mul(zb, bp));
        }
        d_h[rw] = h;
    }
}

// The kernels' shared parameter list (the launch site passes the same
// arguments to every variant).
#define SI_PARAMS                                                                                  \
    Fe3 *__restrict__ d_h, const uint4 *__restrict__ prog, uint32_t num_steps,                    \
        const uint64_t *__restrict__ d_base_consts, const Fe3 *__restrict__ d_uni,                 \
        const Fe3 *__restrict__ d_beta, const uint64_t *__restrict__ d_main, uint64_t main_stride, \
        const uint64_t *__restrict__ d_aux, uint64_t aux_stride, uint64_t next_step,               \
        uint64_t num_rows, const uint64_t *__restrict__ d_z_inv, uint64_t z_len,                   \
        uint64_t num_boundary, const uint64_t *__restrict__ d_b_col,                               \
        const uint64_t *__restrict__ d_b_is_aux, const Fe3 *__restrict__ d_b_value,                \
        const Fe3 *__restrict__ d_b_beta, const uint64_t *__restrict__ d_b_z_inv

// A thread takes rows g, g + n, …, g + (R−1)·n of each tile of R·n rows
// (n = the grid's threads), grid-striding over the tiles.
template <int R, int STAGE, class S>
__device__ __forceinline__ void si_grid(S &s, SI_PARAMS) {
    SiInputs in;
    in.base_consts = d_base_consts;
    in.uni = d_uni;
    in.main = d_main;
    in.main_stride = main_stride;
    in.aux = d_aux;
    in.aux_stride = aux_stride;
    const uint64_t n = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (uint64_t tile = 0; tile < num_rows; tile += (uint64_t)R * n) {
        uint64_t row[R];
        bool valid[R];
#pragma unroll
        for (int j = 0; j < R; j++) {
            uint64_t r = tile + (uint64_t)j * n + g;
            valid[j] = r < num_rows;
            row[j] = valid[j] ? r : 0;
        }
        si_rows<R, STAGE>(s, in, prog, num_steps, d_beta, row, valid, next_step, num_rows, d_h,
                          d_z_inv, z_len, num_boundary, d_b_col, d_b_is_aux, d_b_value, d_b_beta,
                          d_b_z_inv);
    }
}

template <int R, int STAGE> __device__ __forceinline__ void si_smem(SI_PARAMS) {
    extern __shared__ uint64_t si_smem_words[];
    SmemSlots<R> s;
    s.base = si_smem_words + threadIdx.x;
    s.stride = blockDim.x;
    si_grid<R, STAGE>(s, d_h, prog, num_steps, d_base_consts, d_uni, d_beta, d_main, main_stride, d_aux,
               aux_stride, next_step, num_rows, d_z_inv, z_len, num_boundary, d_b_col, d_b_is_aux,
               d_b_value, d_b_beta, d_b_z_inv);
}

template <int R, int MAXW, int STAGE> __device__ __forceinline__ void si_local(SI_PARAMS) {
    LocalSlots<R, MAXW> s;
    si_grid<R, STAGE>(s, d_h, prog, num_steps, d_base_consts, d_uni, d_beta, d_main, main_stride, d_aux,
               aux_stride, next_step, num_rows, d_z_inv, z_len, num_boundary, d_b_col, d_b_is_aux,
               d_b_value, d_b_beta, d_b_z_inv);
}

#define SI_ARGS                                                                                    \
    d_h, prog, num_steps, d_base_consts, d_uni, d_beta, d_main, main_stride, d_aux, aux_stride,    \
        next_step, num_rows, d_z_inv, z_len, num_boundary, d_b_col, d_b_is_aux, d_b_value,         \
        d_b_beta, d_b_z_inv

// Shared-memory slots: `num_words · R · blockDim.x` u64 of dynamic shared memory.
extern "C" __global__ void si_smem_r1(SI_PARAMS) { si_smem<1, 0>(SI_ARGS); }
extern "C" __global__ void si_smem_r2(SI_PARAMS) { si_smem<2, 0>(SI_ARGS); }

// Local-array slots, one row a thread; the program's `num_words` must not
// exceed the variant's width.
extern "C" __global__ void si_local_w32(SI_PARAMS) { si_local<1, 32, 0>(SI_ARGS); }
extern "C" __global__ void si_local_w48(SI_PARAMS) { si_local<1, 48, 0>(SI_ARGS); }
extern "C" __global__ void si_local_w64(SI_PARAMS) { si_local<1, 64, 0>(SI_ARGS); }
extern "C" __global__ void si_local_w128(SI_PARAMS) { si_local<1, 128, 0>(SI_ARGS); }

// The same kernels with the steps staged through shared memory, a tile of
// SI_TILE steps at a time (`_ps`: 4 KiB of static shared memory a block).
extern "C" __global__ void si_smem_r1_ps(SI_PARAMS) { si_smem<1, 1>(SI_ARGS); }
extern "C" __global__ void si_local_w32_ps(SI_PARAMS) { si_local<1, 32, 1>(SI_ARGS); }
extern "C" __global__ void si_local_w48_ps(SI_PARAMS) { si_local<1, 48, 1>(SI_ARGS); }
extern "C" __global__ void si_local_w64_ps(SI_PARAMS) { si_local<1, 64, 1>(SI_ARGS); }
extern "C" __global__ void si_local_w128_ps(SI_PARAMS) { si_local<1, 128, 1>(SI_ARGS); }

// Staged, with the trace cells SI_LOOKAHEAD steps ahead prefetched into L1 (`_pp`).
extern "C" __global__ void si_smem_r1_pp(SI_PARAMS) { si_smem<1, 2>(SI_ARGS); }
extern "C" __global__ void si_local_w32_pp(SI_PARAMS) { si_local<1, 32, 2>(SI_ARGS); }
extern "C" __global__ void si_local_w48_pp(SI_PARAMS) { si_local<1, 48, 2>(SI_ARGS); }
extern "C" __global__ void si_local_w64_pp(SI_PARAMS) { si_local<1, 64, 2>(SI_ARGS); }
extern "C" __global__ void si_local_w128_pp(SI_PARAMS) { si_local<1, 128, 2>(SI_ARGS); }
