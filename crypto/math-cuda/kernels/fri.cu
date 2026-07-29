// R4 FRI fold + twiddle-update kernels on device. The host orchestrator
// loops log2(N) times: sample zeta on host, fold on device, keccak leaves
// + tree on device, D2H the root, transcript-append on host, update
// twiddles on device.
//
// Layout: ext3 evaluations are stored INTERLEAVED as
// `[a0,b0,c0, a1,b1,c1, ...]`, same layout the deep-poly LDE output
// already produces. Twiddles are base-field, one u64 per entry.

#include "goldilocks.cuh"
#include "ext3.cuh"

// GPU port of fold_evaluations_in_place. Port is out-of-place to avoid
// races across threads:
//   out[j] = (lo + hi) + inv_tw[j] * zeta * (lo - hi)
// where lo = evals[2j], hi = evals[2j+1]. Both lo/hi and zeta are ext3.
// inv_tw[j] is a base-field twiddle (F * E -> E).
//
// Writes N/2 ext3 outputs (3 * n_out u64 total) into `out`. `in` is the
// previous layer of 2 * n_out ext3 values (6 * n_out u64 total).
extern "C" __global__ void fri_fold_ext3(
    const uint64_t *in,        // 3 * 2*n_out u64 (ext3 interleaved)
    uint64_t n_out,            // number of output ext3 elements (= N/2)
    const uint64_t *inv_tw,    // n_out base-field twiddles
    const uint64_t *zeta,      // 3 u64 (ext3)
    uint64_t *out) {           // 3 * n_out u64 (ext3 interleaved)
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;

    const uint64_t *lo_p = in + 2 * j * 3;
    const uint64_t *hi_p = lo_p + 3;

    ext3::Fe3 lo = ext3::make(lo_p[0], lo_p[1], lo_p[2]);
    ext3::Fe3 hi = ext3::make(hi_p[0], hi_p[1], hi_p[2]);
    ext3::Fe3 sum = ext3::add(lo, hi);
    ext3::Fe3 diff = ext3::sub(lo, hi);

    ext3::Fe3 z = ext3::make(zeta[0], zeta[1], zeta[2]);
    ext3::Fe3 zd = ext3::mul(z, diff);      // ext3 * ext3 = ext3
    uint64_t tw = inv_tw[j];
    ext3::Fe3 tzd = ext3::mul_base(zd, tw); // base * ext3 = ext3 (componentwise)
    ext3::Fe3 res = ext3::add(sum, tzd);

    uint64_t *out_p = out + j * 3;
    out_p[0] = res.a;
    out_p[1] = res.b;
    out_p[2] = res.c;
}

// First DEEP->FRI fold when the DEEP producer hands over its natural-order
// codeword directly. The legacy host bridge bit-reverses the full codeword and
// then pairs entries (2*j, 2*j+1). Bit reversal is an involution, so reading
// natural input at br(2*j) / br(2*j+1) is byte-identical while avoiding both
// the CPU permutation and a separate device permutation pass. Output j is in
// the ordinary FRI layer order, so every later fold uses fri_fold_ext3.
extern "C" __global__ void fri_fold_ext3_from_natural(
    const uint64_t *__restrict__ in,
    uint64_t n_out,
    uint64_t log_n_in,
    const uint64_t *__restrict__ inv_tw,
    const uint64_t *__restrict__ zeta,
    uint64_t *__restrict__ out) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;

    uint64_t br_lo = __brevll(2 * j) >> (64 - log_n_in);
    uint64_t br_hi = __brevll(2 * j + 1) >> (64 - log_n_in);
    const uint64_t *lo_p = in + br_lo * 3;
    const uint64_t *hi_p = in + br_hi * 3;

    ext3::Fe3 lo = ext3::make(lo_p[0], lo_p[1], lo_p[2]);
    ext3::Fe3 hi = ext3::make(hi_p[0], hi_p[1], hi_p[2]);
    ext3::Fe3 sum = ext3::add(lo, hi);
    ext3::Fe3 diff = ext3::sub(lo, hi);
    ext3::Fe3 z = ext3::make(zeta[0], zeta[1], zeta[2]);
    ext3::Fe3 res = ext3::add(sum, ext3::mul_base(ext3::mul(z, diff), inv_tw[j]));

    uint64_t *out_p = out + j * 3;
    out_p[0] = res.a;
    out_p[1] = res.b;
    out_p[2] = res.c;
}

// update_twiddles: tw_out[j] = tw_in[2j]^2 for j in 0..n_out.
// Separate input/output buffers: thread j reads tw_in[2j] while thread 2j
// writes tw_out[2j], so an in-place version would race across threads.
extern "C" __global__ void fri_update_twiddles(
    const uint64_t *tw_in,
    uint64_t *tw_out,
    uint64_t n_out) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_out) return;
    uint64_t old = tw_in[2 * j];
    tw_out[j] = goldilocks::mul(old, old);
}
