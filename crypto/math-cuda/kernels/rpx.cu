// RPX256 (Rescue-Prime eXtended / XHash12) over Goldilocks at width 12 on
// device — the permutation, the rate-8 overwrite-duplex leaf sponge and the
// Merkle parent (lane K phase 1, arithmetic), then the leaf/tree kernels that
// stream table rows through `rpx::Sponge` and `rpx::compress` and the
// permutation probe (phase 2, the `extern "C"` surface at the end of the
// file, kernel for kernel the twin of `blake3.cu:338-620`).
//
// THE ORACLE is the Rust host implementation, byte for byte:
//   `prover/src/lfm/rpx.rs`   `Rpx256::permute` (:280-316) — schedule FB E FB E FB E M,
//                             `cubic_ext::{mul, power7}` (:118-140);
//   `prover/src/lfm/rpo.rs`   ARK1/ARK2 (:119-321, RPX imports RPO's tables
//                             literally), `sbox` (:455-460), `inv_sbox_layer`
//                             (:481-509, the 72-multiplication chain), `mds`
//                             (:539-557, the u128 accumulation);
//   `prover/src/lfm/algebraic_commit.rs`  `leaf_capacity` (:142-147),
//                             `sponge_leaf` (:169-184), `parent` (:248-252);
//   `prover/src/lfm/hash.rs`  `permute_two_cells` (:95-108): `[a ‖ b ‖ iv]`,
//                             digest = lanes 0..4.
//
// PROVENANCE, layered exactly as the Rust module's own (rpx.rs "PROVENANCE"):
// the FB round IS RPO's round with RPO's constants, and those are pinned by
// nineteen EXTERNAL miden-crypto vectors, which `tests/host_kat/rpx_host_kat.cpp`
// replays through `fb_round(s, r)` composed seven times. The E round (the cubic
// extension) and the schedule have no published vector anywhere; they are
// pinned to the Rust oracle's output (`prover/tests/rpx_host_kat_vectors.rs`)
// and, independently, to naive polynomial arithmetic in the harness.
//
// REPRESENTATION. Inputs may be raw `[0, 2^64)` Goldilocks storage exactly as
// `goldilocks.cuh` allows everywhere else; every step here (`add`, `mul`,
// `dot3`, the MDS bound) accepts that. `permute` CANONICALISES its output, so
// digests are canonical `< p` and their big-endian bytes are what
// `digest_to_commitment` (algebraic_commit.rs:112-118) writes — the device
// Merkle tree can be compared to the host's byte for byte.
//
// ⚠ TWO CUBIC EXTENSIONS EXIST AND THIS FILE USES THE OTHER ONE. `ext3.cuh` is
// the VM's `w³ = 2`; RPX's is `φ³ = φ + 1` (rpx.rs:98-103). Only the GENERIC
// three-term dot product `ext3::dot3` is borrowed from that header — never
// `ext3::mul`. The reduction polynomial lives in `rpx::ext_mul` alone.
//
// COST MODEL (one permutation; counted by the harness's op counters, static
// for the MDS):
//   FB round  ×3 : 12·(4 + 72) = 912 Goldilocks multiplications (48 forward
//                  S-box, 864 inverse), 2 MDS, 24 constant adds;
//   E round   ×3 : 4 triples × 4 extension products = 16 `ext_mul` = 144 wide
//                  64×64 products folded into 48 reductions (`dot3`), 12
//                  constant adds, 32 operand pre-adds;
//   M round   ×1 : 1 MDS, 12 constant adds;
//   MDS       ×7 : 288 32×32→64 multiply-adds + 12 reductions each — the ported
//                  u128 property (see `mds`), ~6× under twelve field
//                  multiplications per lane.
//   Total: 2736 field multiplications + 144 dot3 (432 wide products) + 300 adds
//   + 2016 narrow MACs. The inverse S-box is 2592/2736 = 95% of the field
//   multiplications; RPO spends 7 such layers, RPX 3 — that is the whole
//   reason RPX exists (rpx.rs:22-28).
//
// PHASE-2 TUNING NOTES (not done here, do not guess at them): `inv_sbox` is a
// serial 72-deep chain per lane — one thread per permutation interleaves twelve
// of them; register pressure is what to measure (`-Xptxas -v`). `Sponge::absorb`
// indexes the state dynamically, which nvcc lowers to local memory unless the
// caller's loop is unrolled — the same trade `Blake3Chain::push_word` makes.
// ARK reads are warp-uniform constant-bank operands and cost nothing.

#include <cstdint>
#include "goldilocks.cuh"
#include "ext3.cuh"

// `permute` is a REAL device function, never inlined (see its CODE SHAPE
// note). The host shim has no `__noinline__`; on the host the attribute only
// matters to the code-size probe, which asks for it explicitly.
#if defined(__CUDACC__)
#define RPX_NOINLINE __noinline__
#elif defined(RPX_HOST_NOINLINE)
#define RPX_NOINLINE __attribute__((noinline))
#else
#define RPX_NOINLINE
#endif

namespace rpx {

enum : int {
    STATE_FELTS = 12,
    RATE_FELTS = 8,
    CAPACITY_FELTS = 4,
    DIGEST_FELTS = 4,
    NUM_ROUNDS = 7,
    EXT_DEGREE = 3,
    EXT_ELEMENTS = 4,
    // Absolute lanes of the two capacity cells the socket names: the padding
    // flag (`rpo.rs:339` CAPACITY_PAD_LANE = 0 within the capacity) and the
    // domain tag (`rpo.rs:343` CAPACITY_DOMAIN_LANE = 1). Capacity = lanes 8..12.
    CAPACITY_PAD_LANE = RATE_FELTS + 0,
    CAPACITY_DOMAIN_LANE = RATE_FELTS + 1,
};

// The Merkle-parent domain is ZERO on purpose (rpo.rs:350): a parent is a
// standard `Rpx256::merge`, checkable against miden without this codebase.
__device__ constexpr uint64_t DOMAIN_COMPRESS = 0;
// The leaf domain: `u32::from_le_bytes(b"LFML")` (rpo.rs:358) = 1280132684.
__device__ constexpr uint64_t DOMAIN_LEAF = 0x4C4D464CULL;

// ---------------------------------------------------------------------------
// Constants. Transcribed MECHANICALLY (a script over rpo.rs, not by hand) from
// `rpo.rs` ARK1 (:119-218), ARK2 (:222-321) and MDS_CIRC_ROW (:114). RPX
// imports exactly these (rpx.rs:69); `rpx_uses_rpos_constant_tables` asserts
// the import on the host, and the miden vectors in the harness pin them here.
// Only the FB rounds (0, 2, 4) consume ARK2; E and M rounds add ARK1 alone.
// ---------------------------------------------------------------------------
__device__ __constant__ uint64_t ARK1[NUM_ROUNDS][STATE_FELTS] = {
    {5789762306288267392ull, 6522564764413701783ull, 17809893479458208203ull, 107145243989736508ull,
     6388978042437517382ull, 15844067734406016715ull, 9975000513555218239ull, 3344984123768313364ull,
     9959189626657347191ull, 12960773468763563665ull, 9602914297752488475ull, 16657542370200465908ull},
    {12987190162843096997ull, 653957632802705281ull, 4441654670647621225ull, 4038207883745915761ull,
     5613464648874830118ull, 13222989726778338773ull, 3037761201230264149ull, 16683759727265180203ull,
     8337364536491240715ull, 3227397518293416448ull, 8110510111539674682ull, 2872078294163232137ull},
    {18072785500942327487ull, 6200974112677013481ull, 17682092219085884187ull, 10599526828986756440ull,
     975003873302957338ull, 8264241093196931281ull, 10065763900435475170ull, 2181131744534710197ull,
     6317303992309418647ull, 1401440938888741532ull, 8884468225181997494ull, 13066900325715521532ull},
    {5674685213610121970ull, 5759084860419474071ull, 13943282657648897737ull, 1352748651966375394ull,
     17110913224029905221ull, 1003883795902368422ull, 4141870621881018291ull, 8121410972417424656ull,
     14300518605864919529ull, 13712227150607670181ull, 17021852944633065291ull, 6252096473787587650ull},
    {4887609836208846458ull, 3027115137917284492ull, 9595098600469470675ull, 10528569829048484079ull,
     7864689113198939815ull, 17533723827845969040ull, 5781638039037710951ull, 17024078752430719006ull,
     109659393484013511ull, 7158933660534805869ull, 2955076958026921730ull, 7433723648458773977ull},
    {16308865189192447297ull, 11977192855656444890ull, 12532242556065780287ull, 14594890931430968898ull,
     7291784239689209784ull, 5514718540551361949ull, 10025733853830934803ull, 7293794580341021693ull,
     6728552937464861756ull, 6332385040983343262ull, 13277683694236792804ull, 2600778905124452676ull},
    {7123075680859040534ull, 1034205548717903090ull, 7717824418247931797ull, 3019070937878604058ull,
     11403792746066867460ull, 10280580802233112374ull, 337153209462421218ull, 13333398568519923717ull,
     3596153696935337464ull, 8104208463525993784ull, 14345062289456085693ull, 17036731477169661256ull},
};

__device__ __constant__ uint64_t ARK2[NUM_ROUNDS][STATE_FELTS] = {
    {6077062762357204287ull, 15277620170502011191ull, 5358738125714196705ull, 14233283787297595718ull,
     13792579614346651365ull, 11614812331536767105ull, 14871063686742261166ull, 10148237148793043499ull,
     4457428952329675767ull, 15590786458219172475ull, 10063319113072092615ull, 14200078843431360086ull},
    {6202948458916099932ull, 17690140365333231091ull, 3595001575307484651ull, 373995945117666487ull,
     1235734395091296013ull, 14172757457833931602ull, 707573103686350224ull, 15453217512188187135ull,
     219777875004506018ull, 17876696346199469008ull, 17731621626449383378ull, 2897136237748376248ull},
    {8023374565629191455ull, 15013690343205953430ull, 4485500052507912973ull, 12489737547229155153ull,
     9500452585969030576ull, 2054001340201038870ull, 12420704059284934186ull, 355990932618543755ull,
     9071225051243523860ull, 12766199826003448536ull, 9045979173463556963ull, 12934431667190679898ull},
    {18389244934624494276ull, 16731736864863925227ull, 4440209734760478192ull, 17208448209698888938ull,
     8739495587021565984ull, 17000774922218161967ull, 13533282547195532087ull, 525402848358706231ull,
     16987541523062161972ull, 5466806524462797102ull, 14512769585918244983ull, 10973956031244051118ull},
    {6982293561042362913ull, 14065426295947720331ull, 16451845770444974180ull, 7139138592091306727ull,
     9012006439959783127ull, 14619614108529063361ull, 1394813199588124371ull, 4635111139507788575ull,
     16217473952264203365ull, 10782018226466330683ull, 6844229992533662050ull, 7446486531695178711ull},
    {3736792340494631448ull, 577852220195055341ull, 6689998335515779805ull, 13886063479078013492ull,
     14358505101923202168ull, 7744142531772274164ull, 16135070735728404443ull, 12290902521256031137ull,
     12059913662657709804ull, 16456018495793751911ull, 4571485474751953524ull, 17200392109565783176ull},
    {17130398059294018733ull, 519782857322261988ull, 9625384390925085478ull, 1664893052631119222ull,
     7629576092524553570ull, 3485239601103661425ull, 9755891797164033838ull, 15218148195153269027ull,
     16460604813734957368ull, 9643968136937729763ull, 3611348709641382851ull, 18256379591337759196ull},
};

// First ROW of the circulant MDS, `M[i][j] = ROW[(j − i) mod 12]`
// (rpo.rs:107-114), stored TWICE so that `MDS_CIRC_ROW2[j + 12 − i]` is the
// entry with no modulo: the output-lane loop in `mds` is rolled, so `i` is a
// runtime value there. 32-bit so each MDS term is one 32×32→64 MAC. The row
// sums to 160, which is the bound `mds` rests on.
__device__ __constant__ uint32_t MDS_CIRC_ROW2[2 * STATE_FELTS] = {
    7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8, 7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8};

// ---------------------------------------------------------------------------
// Field-op forwarders. Under nvcc they are the `goldilocks.cuh` / `ext3.cuh`
// primitives, nothing more. The host-KAT harness defines RPX_HOST_OP_COUNT
// before including this file so it can COUNT them per round kind and print the
// cost model above as a measurement rather than a claim.
// ---------------------------------------------------------------------------
#ifdef RPX_HOST_OP_COUNT
struct OpCount {
    unsigned long long mul, dot3, add;
};
static OpCount g_ops = {0, 0, 0};
#define RPX_COUNT(field) (++g_ops.field)
#else
#define RPX_COUNT(field) ((void)0)
#endif

__device__ __forceinline__ uint64_t fmul(uint64_t a, uint64_t b) {
    RPX_COUNT(mul);
    return goldilocks::mul(a, b);
}

__device__ __forceinline__ uint64_t fadd(uint64_t a, uint64_t b) {
    RPX_COUNT(add);
    return goldilocks::add(a, b);
}

// `a0·b0 + a1·b1 + a2·b2` with ONE reduction — the generic part of `ext3.cuh`,
// independent of that header's reduction polynomial.
__device__ __forceinline__ uint64_t fdot3(uint64_t a0, uint64_t b0, uint64_t a1, uint64_t b1,
                                          uint64_t a2, uint64_t b2) {
    RPX_COUNT(dot3);
    return ext3::dot3(a0, b0, a1, b1, a2, b2);
}

// ---------------------------------------------------------------------------
// The circulant MDS, `out_i = Σ_j MDS_CIRC_ROW[(j − i) mod 12] · s_j`, in
// `rpo.rs:539-557`'s orientation (the one the miden vectors pin).
//
// ★ THE PORTED PROPERTY (rpo.rs:527-536): one accumulation and ONE reduction
// per output lane, no per-term field multiplication. Every coefficient is ≤ 26
// and every stored lane is < 2^64, so the twelve-term row sum is < 12·26·2^64
// < 2^73 and needs no reduction before the end. The host accumulates it in a
// u128; the device has no u128, so the SAME integer is assembled from 32-bit
// halves. With `s_j = h_j·2^32 + l_j`,
//
//     acc = 2^32 · Σ_j c_j·h_j  +  Σ_j c_j·l_j ,
//
// and each half-sum is ≤ 160·(2^32 − 1) < 2^40 — the row sums to 160 — so both
// fit a u64 with 24 bits to spare and every term is a single 32×32→64
// multiply-add (no 64-bit multiplier anywhere in the MDS). The halves are then
// recombined into the u128's `(lo, hi)` exactly as the host holds them and
// reduced the host's way: `acc = hi·2^64 + lo ≡ lo + hi·EPSILON (mod p)`, with
// `hi < 2^9` so `hi·EPSILON < 2^41` needs no reduction of its own
// (`the_mds_row_sum_cannot_overflow_a_u128` asserts the same bound on the host).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mds(uint64_t s[STATE_FELTS]) {
    uint32_t lo32[STATE_FELTS], hi32[STATE_FELTS];
#pragma unroll
    for (int j = 0; j < STATE_FELTS; ++j) {
        lo32[j] = (uint32_t)s[j];
        hi32[j] = (uint32_t)(s[j] >> 32);
    }
    uint64_t out[STATE_FELTS];
    // Rolled over output lanes: twelve iterations of twenty-four MACs, one
    // twelfth of the unrolled body's code for the same instruction count.
#pragma unroll 1
    for (int i = 0; i < STATE_FELTS; ++i) {
        uint64_t acc_lo = 0, acc_hi = 0;  // Σ c·l_j and Σ c·h_j, each < 2^40
        const int rot = STATE_FELTS - i;   // MDS_CIRC_ROW2[j + rot] = ROW[(j − i) mod 12]
#pragma unroll
        for (int j = 0; j < STATE_FELTS; ++j) {
            const uint32_t c = MDS_CIRC_ROW2[j + rot];
            acc_lo += (uint64_t)c * (uint64_t)lo32[j];
            acc_hi += (uint64_t)c * (uint64_t)hi32[j];
        }
        // acc = acc_hi·2^32 + acc_lo, exactly. Split it at bit 64.
        const uint64_t lo = (acc_hi << 32) + acc_lo;
        const uint64_t carry = (lo < acc_lo) ? 1ull : 0ull;
        const uint64_t hi = (acc_hi >> 32) + carry;  // < 2^9
        out[i] = fadd(lo, hi * goldilocks::EPSILON);
    }
#pragma unroll
    for (int i = 0; i < STATE_FELTS; ++i) s[i] = out[i];
}

// ---------------------------------------------------------------------------
// S-boxes.
// ---------------------------------------------------------------------------

// `x^7` in the association the AIR's degree-3 lowering uses (rpo.rs:455-460):
// `x², x³ = x²·x, x^7 = (x³)²·x`. Two squarings, two products.
__device__ __forceinline__ uint64_t sbox(uint64_t x) {
    const uint64_t x2 = fmul(x, x);
    const uint64_t x3 = fmul(x2, x);
    const uint64_t x6 = fmul(x3, x3);
    return fmul(x6, x);
}

template <int N>
__device__ __forceinline__ uint64_t square_n(uint64_t x) {
    // Rolled: the chain is serial anyway, and unrolled it is what made one
    // permutation ~49k lines of PTX. The unroll factor here is a tuning knob.
#pragma unroll 1
    for (int i = 0; i < N; ++i) x = fmul(x, x);
    return x;
}

// `base^(2^M) · tail` — the inverse chain's one building block (rpo.rs:483-495).
template <int M>
__device__ __forceinline__ uint64_t exp_acc(uint64_t base, uint64_t tail) {
    return fmul(square_n<M>(base), tail);
}

// `x^{1/7} = x^10540996611094048183` by miden-crypto's addition chain, as
// `rpo.rs:481-509` runs it lane-wise: 63 squarings + 9 products = 72
// multiplications against ~93 for square-and-multiply. Per lane rather than
// whole-state: on a GPU the twelve lanes' independence is the compiler's to
// interleave, and a lane-wise body keeps only six values live.
__device__ __forceinline__ uint64_t inv_sbox(uint64_t x) {
    const uint64_t t1 = fmul(x, x);            // x^2
    const uint64_t t2 = fmul(t1, t1);          // x^4
    const uint64_t t3 = exp_acc<3>(t2, t2);    // x^36
    const uint64_t t4 = exp_acc<6>(t3, t3);    // x^(36·65)
    const uint64_t t5 = exp_acc<12>(t4, t4);   // x^(36·65·4097)
    const uint64_t t6 = exp_acc<6>(t5, t3);    // x^0x24924924
    const uint64_t t7 = exp_acc<31>(t6, t6);   // x^0x1249249224924924
    // ((t7² · t6)²)² · ((t1 · t2) · x)  — rpo.rs:504-508.
    const uint64_t a = square_n<2>(fmul(fmul(t7, t7), t6));
    const uint64_t b = fmul(fmul(t1, t2), x);
    return fmul(a, b);
}

// ---------------------------------------------------------------------------
// The cubic extension `GF(p)[φ] / (φ³ − φ − 1)` — rpx.rs:98-140. NOT `ext3.cuh`'s.
// ---------------------------------------------------------------------------
struct CubicExt {
    uint64_t c0, c1, c2;  // c0 + c1·φ + c2·φ²
};

// The product reduced by `φ³ = φ + 1`, `φ⁴ = φ² + φ`. `rpx.rs:118-125`'s
// closed form, regrouped so each coefficient is ONE three-term dot product
// with a single reduction (the same fold `dot_product_3` gives the VM's own
// extension):
//   c0 = a0·b0 + a1·b2 + a2·b1
//   c1 = a0·b1 + a1·(b0 + b2) + a2·(b1 + b2)   [= a0b1 + a1b0 + a1b2 + a2b1 + a2b2]
//   c2 = a0·b2 + a1·b1 + a2·(b0 + b2)          [= a0b2 + a1b1 + a2b0 + a2b2]
// Nine wide products, three reductions, two operand pre-adds.
__device__ __forceinline__ CubicExt ext_mul(const CubicExt &a, const CubicExt &b) {
    const uint64_t b02 = fadd(b.c0, b.c2);
    const uint64_t b12 = fadd(b.c1, b.c2);
    CubicExt r;
    r.c0 = fdot3(a.c0, b.c0, a.c1, b.c2, a.c2, b.c1);
    r.c1 = fdot3(a.c0, b.c1, a.c1, b02, a.c2, b12);
    r.c2 = fdot3(a.c0, b.c2, a.c1, b.c1, a.c2, b02);
    return r;
}

// One function for squaring and product, as on the host (rpx.rs:128-130).
__device__ __forceinline__ CubicExt ext_square(const CubicExt &a) { return ext_mul(a, a); }

// `a^7` by `a² → a³ → a⁶ → a⁷` (rpx.rs:135-140): two squarings, two products.
__device__ __forceinline__ CubicExt ext_power7(const CubicExt &a) {
    const CubicExt a2 = ext_square(a);
    const CubicExt a3 = ext_mul(a2, a);
    const CubicExt a6 = ext_square(a3);
    return ext_mul(a6, a);
}

// ---------------------------------------------------------------------------
// Rounds. `r` is the round index into ARK1/ARK2 — a runtime value, so one copy
// of each round body serves every round; the constant-bank address is
// computed, which costs nothing next to the round's arithmetic. Every lane
// loop is rolled for the same reason (see `permute`'s CODE SHAPE note).
// ---------------------------------------------------------------------------

// FB: `MDS → +ARK1 → x^7 → MDS → +ARK2 → x^{1/7}` — RPO's round exactly
// (rpo.rs:561-582, rpx.rs:283-295). RPX runs it at R = 0, 2, 4; RPO at 0..7.
__device__ __forceinline__ void fb_round(uint64_t s[STATE_FELTS], int r) {
    mds(s);
#pragma unroll 1
    for (int i = 0; i < STATE_FELTS; ++i) s[i] = fadd(s[i], ARK1[r][i]);
#pragma unroll 1
    for (int i = 0; i < STATE_FELTS; ++i) s[i] = sbox(s[i]);
    mds(s);
#pragma unroll 1
    for (int i = 0; i < STATE_FELTS; ++i) s[i] = fadd(s[i], ARK2[r][i]);
    // The twelve chains are independent; a GPU hides their latency with other
    // warps, not by unrolling one thread's twelve chains into straight line.
#pragma unroll 1
    for (int i = 0; i < STATE_FELTS; ++i) s[i] = inv_sbox(s[i]);
}

// E: `+ARK1 → x^7` in the cubic extension on four lane-triples, NO linear
// layer (rpx.rs:296-307; the design, not an omission — rpx.rs:275-279).
__device__ __forceinline__ void ext_round(uint64_t s[STATE_FELTS], int r) {
#pragma unroll 1
    for (int i = 0; i < STATE_FELTS; ++i) s[i] = fadd(s[i], ARK1[r][i]);
#pragma unroll 1
    for (int e = 0; e < EXT_ELEMENTS; ++e) {
        const int base = e * EXT_DEGREE;
        CubicExt x;
        x.c0 = s[base];
        x.c1 = s[base + 1];
        x.c2 = s[base + 2];
        const CubicExt y = ext_power7(x);
        s[base] = y.c0;
        s[base + 1] = y.c1;
        s[base + 2] = y.c2;
    }
}

// M: `MDS → +ARK1`, a linear finish with no S-box (rpx.rs:308-313).
__device__ __forceinline__ void final_round(uint64_t s[STATE_FELTS], int r) {
    mds(s);
#pragma unroll 1
    for (int i = 0; i < STATE_FELTS; ++i) s[i] = fadd(s[i], ARK1[r][i]);
}

// The permutation: `FB E FB E FB E M` (rpx.rs:280-316), output CANONICAL.
//
// ★ CODE SHAPE. A real (`RPX_NOINLINE`) function with rolled loops, on
// purpose. The first cubin build of the fully inlined, fully unrolled form ran
// 41 minutes and emitted 56 MB of PTX: one permutation was ~49k straight-line
// lines (the inverse S-box chain unrolled over twelve lanes, three times) and
// every leaf kernel carried one copy per `permute` call site — seven in the
// comp-poly kernel. Rolled and called, the whole file is a few thousand lines
// and every kernel shares one body. The price is loop overhead of order 10% of
// the permutation's instructions and the state living in local memory across
// the call; the `-Xptxas -v` report and the unroll factors of `square_n` and
// the lane loops are the tuning knobs, in that order.
RPX_NOINLINE __device__ void permute(uint64_t s[STATE_FELTS]) {
#pragma unroll 1
    for (int r = 0; r + 1 < NUM_ROUNDS; r += 2) {
        fb_round(s, r);
        ext_round(s, r + 1);
    }
    final_round(s, NUM_ROUNDS - 1);
#pragma unroll 1
    for (int i = 0; i < STATE_FELTS; ++i) s[i] = goldilocks::canonical(s[i]);
}

// ---------------------------------------------------------------------------
// The socket's two constructions over the permutation.
// ---------------------------------------------------------------------------

// The rate-8 OVERWRITE duplex — `algebraic_commit::sponge_leaf` (:169-184)
// with `leaf_capacity` (:142-147), streamed. Capacity lane 8 carries the
// padding flag `len mod 8`, lane 9 the LEAF domain, lanes 10-11 zero. Each
// block OVERWRITES the eight rate lanes (spec §2.6): absorption is a store, no
// field arithmetic. The total length is needed BEFORE the first permutation
// (algebraic_commit.rs "A1"), hence `init(num_felts)`; callers absorb exactly
// that many felts.
struct Sponge {
    uint64_t s[STATE_FELTS];
    int pos;

    __device__ __forceinline__ void init(uint64_t num_felts) {
#pragma unroll
        for (int i = 0; i < RATE_FELTS; ++i) s[i] = 0;
        s[CAPACITY_PAD_LANE] = num_felts % RATE_FELTS;
        s[CAPACITY_DOMAIN_LANE] = DOMAIN_LEAF;
        s[CAPACITY_DOMAIN_LANE + 1] = 0;
        s[CAPACITY_DOMAIN_LANE + 2] = 0;
        pos = 0;
    }

    __device__ __forceinline__ void absorb(uint64_t felt) {
        s[pos++] = felt;
        if (pos == RATE_FELTS) {
            permute(s);
            pos = 0;
        }
    }

    // A pending partial block is zero-padded and permuted. An exact multiple of
    // the rate spends no trailing permutation — including the EMPTY leaf, whose
    // digest is therefore the untouched zero rate lanes, exactly what
    // `sponge_leaf` returns for `felts.is_empty()` (:174-176).
    __device__ __forceinline__ void finalize(uint64_t digest[DIGEST_FELTS]) {
        if (pos != 0) {
            for (int k = pos; k < RATE_FELTS; ++k) s[k] = 0;
            permute(s);
            pos = 0;
        }
#pragma unroll
        for (int i = 0; i < DIGEST_FELTS; ++i) digest[i] = s[i];
    }
};

// `sponge_leaf` over a contiguous array — the one-call form for the KAT and
// for any phase-2 kernel that has its felts in hand.
__device__ __forceinline__ void sponge_leaf(const uint64_t *felts, uint64_t num_felts,
                                            uint64_t digest[DIGEST_FELTS]) {
    Sponge sp;
    sp.init(num_felts);
    for (uint64_t i = 0; i < num_felts; ++i) sp.absorb(felts[i]);
    sp.finalize(digest);
}

// The Merkle parent: ONE permutation of `[left ‖ right ‖ capacity]` with the
// compress domain, which is zero (algebraic_commit.rs:248-252 →
// hash.rs:95-108). Capacity = `domain_iv(0)` = all zeros.
__device__ __forceinline__ void compress(const uint64_t left[DIGEST_FELTS],
                                         const uint64_t right[DIGEST_FELTS],
                                         uint64_t out[DIGEST_FELTS]) {
    uint64_t s[STATE_FELTS];
#pragma unroll
    for (int i = 0; i < DIGEST_FELTS; ++i) {
        s[i] = left[i];
        s[DIGEST_FELTS + i] = right[i];
        s[RATE_FELTS + i] = 0;
    }
    s[CAPACITY_DOMAIN_LANE] = DOMAIN_COMPRESS;
    permute(s);
#pragma unroll
    for (int i = 0; i < DIGEST_FELTS; ++i) out[i] = s[i];
}

}  // namespace rpx

// ===========================================================================
// PHASE 2 — the device-facing surface: node bytes, leaf kernels, Merkle
// compressors and the permutation probe. Kernel for kernel the twin of
// `blake3.cu:338-620`, with the chain replaced by `rpx::Sponge` and the parent
// by `rpx::compress`.
//
// NODE BYTES. A node is four canonical felts, each stored as eight BIG-ENDIAN
// bytes — `digest_to_commitment` (algebraic_commit.rs:112-118) — so 32 bytes,
// the same slot width as a BLAKE3 or keccak node, and the device tree's bytes
// equal the host's. Digests leave `permute` canonical; a parent reads its
// children back with `commitment_to_digest`'s big-endian decoding. The device
// is little-endian, so both directions byte-swap (`bswap64`); the 32-byte node
// offsets inside a 256-byte-aligned `cuMemAlloc` buffer make the u64 accesses
// aligned, the same precondition the BLAKE3 u32 accesses rest on.
//
// A LEAF absorbs exactly the felt sequence the host leaf hashes: the same
// read pattern as the BLAKE3 kernel it twins (`leaves_bit_reversed_grouped`,
// commitment.rs:67 — bit-reversed rows, each column by column, an ext3 element
// as its three components), which is the sequence `felts_from_bytes` rebuilds
// from the leaf bytes, so `hash_bytes == hash_data` holds on device by
// construction. The felt count is known before the loop, as the overwrite
// duplex's padding flag needs it (A1). Raw `[0, 2^64)` storage is absorbed as
// is: the permutation is representation-independent, and the host
// canonicalises before serialising — same field value, same digest.
// ===========================================================================

namespace rpx {

// Byte-swap a u64: the device reads a host big-endian felt from a node and
// writes one back. Plain shifts so the host shim compiles it; nvcc lowers it
// to two PRMTs.
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    x = ((x & 0x00FF00FF00FF00FFull) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFull);
    x = ((x & 0x0000FFFF0000FFFFull) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFull);
    return (x << 32) | (x >> 32);
}

// Four felts → one 32-byte node, `digest_to_commitment`'s layout.
__device__ __forceinline__ void store_digest_be(const uint64_t digest[DIGEST_FELTS], uint8_t *node) {
    uint64_t *dst = reinterpret_cast<uint64_t *>(node);
#pragma unroll
    for (int i = 0; i < DIGEST_FELTS; ++i) dst[i] = bswap64(digest[i]);
}

// One 32-byte node → four felts, `commitment_to_digest`'s decoding.
__device__ __forceinline__ void load_digest_be(const uint8_t *node, uint64_t digest[DIGEST_FELTS]) {
    const uint64_t *src = reinterpret_cast<const uint64_t *>(node);
#pragma unroll
    for (int i = 0; i < DIGEST_FELTS; ++i) digest[i] = bswap64(src[i]);
}

// A Merkle parent in place in the node buffer — `parent` (algebraic_commit.rs
// :248-252): decode both children, `compress`, encode. Node buffer layout as
// `blake3.cu` / `keccak.cu` / the CPU `merkle.rs`: children at
// `nodes[parent_begin + n_pairs .. parent_begin + 3*n_pairs]`, parents at
// `nodes[parent_begin .. parent_begin + n_pairs]`, 32 bytes per node.
__device__ __forceinline__ void hash_merkle_parent(uint8_t *nodes, uint64_t parent_begin,
                                                   uint64_t n_pairs, uint64_t tid) {
    uint64_t left[DIGEST_FELTS], right[DIGEST_FELTS], out[DIGEST_FELTS];
    load_digest_be(nodes + (parent_begin + n_pairs + 2 * tid) * 32, left);
    load_digest_be(nodes + (parent_begin + n_pairs + 2 * tid + 1) * 32, right);
    compress(left, right, out);
    store_digest_be(out, nodes + (parent_begin + tid) * 32);
}

}  // namespace rpx

// ---------------------------------------------------------------------------
// Leaf kernels. Twins of `blake3_leaves_*` / `blake3_comp_poly_leaves_ext3` /
// `blake3_fri_leaves_ext3`, argument for argument; one thread hashes one leaf.
// ---------------------------------------------------------------------------

// Goldilocks BASE-FIELD leaf hashing, one leaf per bit-reversed row: column
// `c` of row `br` at `columns_base_ptr[c * col_stride + br]`.
// Twin of `blake3_leaves_base_batched` (`blake3.cu:346`).
extern "C" __global__ void rpx_leaves_base_batched(
    const uint64_t *columns_base_ptr,
    uint64_t col_stride,
    uint64_t num_cols,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    uint64_t br = __brevll(tid) >> (64 - log_num_rows);

    rpx::Sponge sp;
    sp.init(num_cols);
    for (uint64_t c = 0; c < num_cols; ++c) sp.absorb(columns_base_ptr[c * col_stride + br]);
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, hashed_leaves_out + tid * 32);
}

// BASE-FIELD row-pair leaf hashing: leaf `tid` hashes bit-reversed rows
// `2*tid` and `2*tid+1`, each column by column, first row then second.
// `num_leaves = num_rows / 2`. Twin of `blake3_leaves_base_row_pair_batched`.
extern "C" __global__ void rpx_leaves_base_row_pair_batched(
    const uint64_t *columns_base_ptr,
    uint64_t col_stride,
    uint64_t num_cols,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_leaves = num_rows >> 1;
    if (tid >= num_leaves) return;
    uint64_t br_0 = __brevll(2 * tid) >> (64 - log_num_rows);
    uint64_t br_1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);

    rpx::Sponge sp;
    sp.init(2 * num_cols);
    for (uint64_t c = 0; c < num_cols; ++c) sp.absorb(columns_base_ptr[c * col_stride + br_0]);
    for (uint64_t c = 0; c < num_cols; ++c) sp.absorb(columns_base_ptr[c * col_stride + br_1]);
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, hashed_leaves_out + tid * 32);
}

// EXT3 leaf hashing, one leaf per bit-reversed row, components in three
// separate base slabs: column `c` component `k` at
// `columns_base_ptr[(c*3 + k) * col_stride + br]`; an element is absorbed as
// `[comp0, comp1, comp2]`, matching `write_bytes_be`.
// Twin of `blake3_leaves_ext3_batched`.
extern "C" __global__ void rpx_leaves_ext3_batched(
    const uint64_t *columns_base_ptr,
    uint64_t col_stride,
    uint64_t num_cols,  // number of ext3 columns (NOT slabs)
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    uint64_t br = __brevll(tid) >> (64 - log_num_rows);

    rpx::Sponge sp;
    sp.init(3 * num_cols);
    for (uint64_t c = 0; c < num_cols; ++c) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            sp.absorb(columns_base_ptr[(c * 3 + (uint64_t)k) * col_stride + br]);
        }
    }
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, hashed_leaves_out + tid * 32);
}

// Composition-polynomial leaf hashing: each leaf absorbs `2 * num_parts` ext3
// values from bit-reversed rows `2*tid` and `2*tid+1`, (row 0: parts) then
// (row 1: parts), three base components per value.
// Twin of `blake3_comp_poly_leaves_ext3`.
extern "C" __global__ void rpx_comp_poly_leaves_ext3(
    const uint64_t *parts_base_ptr,
    uint64_t col_stride,
    uint64_t num_parts,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_leaves = num_rows >> 1;
    if (tid >= num_leaves) return;
    uint64_t br_0 = __brevll(2 * tid) >> (64 - log_num_rows);
    uint64_t br_1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);

    rpx::Sponge sp;
    sp.init(2 * 3 * num_parts);
    for (uint64_t p = 0; p < num_parts; ++p) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            sp.absorb(parts_base_ptr[(p * 3 + (uint64_t)k) * col_stride + br_0]);
        }
    }
    for (uint64_t p = 0; p < num_parts; ++p) {
#pragma unroll
        for (int k = 0; k < 3; ++k) {
            sp.absorb(parts_base_ptr[(p * 3 + (uint64_t)k) * col_stride + br_1]);
        }
    }
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, leaves_out + tid * 32);
}

// FRI layer leaf hashing: each leaf absorbs two consecutive ext3 values from an
// interleaved eval vector `[a0,a1,a2,b0,b1,b2,...]` — six felts, so a single
// block, no padding flag (`6 mod 8 = 6` in capacity lane 8). No bit reversal.
// The host is `AlgebraicPairBackend::hash_data` (algebraic_commit.rs:318-329).
// Twin of `blake3_fri_leaves_ext3`.
extern "C" __global__ void rpx_fri_leaves_ext3(
    const uint64_t *evals_interleaved,  // 3 * num_evals u64s
    uint64_t num_leaves,                 // = num_evals / 2
    uint8_t *leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    const uint64_t *pair = evals_interleaved + 2 * tid * 3;

    rpx::Sponge sp;
    sp.init(6);
#pragma unroll
    for (int i = 0; i < 6; ++i) sp.absorb(pair[i]);
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, leaves_out + tid * 32);
}

// Row-major ROW-PAIR leaf hashing: leaf `tid` absorbs row `reverse_index(2*tid)`
// then row `reverse_index(2*tid+1)`, each `m` lanes read contiguously from
// `data + br * m`. `m` is the row stride in u64s: base trace = column count,
// ext3 trace = 3 * column count (an ext3 element's components are consecutive).
// Twin of `blake3_leaves_base_row_major_row_pair`; the fused LDE+commit
// pipeline's leaf kernel (`lde.rs` `coset_lde_row_major_inner`).
extern "C" __global__ void rpx_leaves_base_row_major_row_pair(
    const uint64_t *data,
    uint64_t m,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_leaves = num_rows >> 1;
    if (tid >= num_leaves) return;
    uint64_t br_0 = __brevll(2 * tid) >> (64 - log_num_rows);
    uint64_t br_1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);
    const uint64_t *row_0 = data + br_0 * m;
    const uint64_t *row_1 = data + br_1 * m;

    rpx::Sponge sp;
    sp.init(2 * m);
    for (uint64_t c = 0; c < m; ++c) sp.absorb(row_0[c]);
    for (uint64_t c = 0; c < m; ++c) sp.absorb(row_1[c]);
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, hashed_leaves_out + tid * 32);
}

// Column-range variant: each leaf absorbs only columns `[col_start, col_end)`
// of the two bit-reversed rows while `m` stays the full row stride — the CPU
// `commit_rows_bit_reversed_subset`, how preprocessed tables commit their
// precomputed and multiplicity column ranges to separate trees over one LDE.
// Twin of `blake3_leaves_base_row_major_row_pair_range`.
extern "C" __global__ void rpx_leaves_base_row_major_row_pair_range(
    const uint64_t *data,
    uint64_t m,
    uint64_t col_start,
    uint64_t col_end,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_leaves = num_rows >> 1;
    if (tid >= num_leaves) return;
    uint64_t br_0 = __brevll(2 * tid) >> (64 - log_num_rows);
    uint64_t br_1 = __brevll(2 * tid + 1) >> (64 - log_num_rows);
    const uint64_t *row_0 = data + br_0 * m;
    const uint64_t *row_1 = data + br_1 * m;

    rpx::Sponge sp;
    sp.init(2 * (col_end - col_start));
    for (uint64_t c = col_start; c < col_end; ++c) sp.absorb(row_0[c]);
    for (uint64_t c = col_start; c < col_end; ++c) sp.absorb(row_1[c]);
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, hashed_leaves_out + tid * 32);
}

// ---------------------------------------------------------------------------
// Merkle level / tail. Same launch split as BLAKE3's: one thread per pair per
// level while a level is wide, then ONE single-block launch that grid-strides
// every remaining level with a barrier between them.
// ---------------------------------------------------------------------------

// One level of the inner tree: each thread compresses one child pair.
extern "C" __global__ void rpx_merkle_level(uint8_t *nodes,
                                            uint64_t parent_begin,  // in 32-byte nodes
                                            uint64_t n_pairs) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_pairs) return;
    rpx::hash_merkle_parent(nodes, parent_begin, n_pairs, tid);
}

// Every remaining level from `level_begin` up to the root, in one block.
// Twin of `blake3_merkle_tail`.
extern "C" __global__ void rpx_merkle_tail(uint8_t *nodes, uint64_t level_begin) {
    uint64_t lb = level_begin;
    while (lb != 0) {
        uint64_t nb = lb / 2;
        uint64_t n_pairs = lb - nb;
        for (uint64_t tid = threadIdx.x; tid < n_pairs; tid += blockDim.x) {
            rpx::hash_merkle_parent(nodes, nb, n_pairs, tid);
        }
        __syncthreads();
        lb = nb;
    }
}

// ---------------------------------------------------------------------------
// Parity-harness entry point: `n` independent permutations, one thread each.
// The bare device permutation is otherwise unreachable from host code; this is
// what lets the GPU be checked against the host `Rpx256` (and the host-KAT's
// oracle tables) before any tree is built. Not on any production path.
// ---------------------------------------------------------------------------
extern "C" __global__ void rpx_permute_probe(const uint64_t *states, uint64_t n, uint64_t *out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t s[rpx::STATE_FELTS];
#pragma unroll
    for (int i = 0; i < rpx::STATE_FELTS; ++i) s[i] = states[tid * rpx::STATE_FELTS + i];
    rpx::permute(s);
#pragma unroll
    for (int i = 0; i < rpx::STATE_FELTS; ++i) out[tid * rpx::STATE_FELTS + i] = s[i];
}

// ===========================================================================
// PHASE 3 (lane K, not yet coded) — the proof-of-work GRIND on device.
//
// THE MAPPING the kernel must reproduce, derived from the host and pinned here
// so it is not re-derived (✓ VERIFIED against the sources named):
//
//   host predicate   stark/src/grinding.rs::is_valid_nonce_for_inner_hash (:85-97):
//                    valid ⇔ u64::from_be_bytes(D::digest(inner ‖ nonce.to_be_bytes())[..8]) < limit,
//                    limit = 1 << (64 − grinding_factor); inner = D::digest(PREFIX ‖ seed ‖ factor),
//                    41 bytes, computed ONCE per table on the host (:102-113) and never on device.
//   D for RPX        prover/src/lfm/algebraic_commit.rs AlgebraicDigest<RpxCommit> (:440-500):
//                    D::digest(bytes) = digest_to_commitment(sponge_leaf(Rpx, felts_from_bytes(bytes)))
//                    — the LEAF construction, on purpose (:421-436).
//   bytes → felts    felts_from_bytes (:196-206): consecutive 8-byte groups, each
//                    FE::from(u64::from_be_bytes(group)) — BIG-endian, and `FE::from` is
//                    `from_u64`, which maps a raw value ≥ p to raw − p (one subtraction).
//                    The 40-byte outer block is therefore EXACTLY five felts:
//                        f0..f3 = the inner hash's four big-endian u64s (canonical already —
//                                 they are digest_to_commitment output, so each < p),
//                        f4     = the nonce itself (any u64; ≥ p means the element nonce − p,
//                                 which absorbing the raw nonce yields identically because
//                                 the permutation is representation-independent).
//                    ⚠ Big-endian, unlike keccak's `inner_hash_lanes` (LITTLE-endian lanes,
//                    grinding.rs:123-129). A separate host helper, `inner_hash_felts` (BE),
//                    must feed this kernel; feeding it the keccak lanes is a silent wrong hash.
//   sponge mode      sponge_leaf over five felts (:169-184): ONE permutation of
//                        [f0, f1, f2, f3, nonce, 0, 0, 0 | 5, 0x4C4D464C, 0, 0]
//                    — rate lanes 5..8 zero-padded, capacity lane 8 = padding flag
//                    `5 mod 8 = 5`, lane 9 = DOMAIN_LEAF ("LFML"), lanes 10, 11 = 0.
//                    Identical to `Sponge::init(5)`, five `absorb`s, `finalize`.
//   the head         digest_to_commitment writes lane 0 CANONICAL as 8 big-endian bytes and the
//                    host reads those 8 bytes back big-endian, so `seed_head` IS the canonical
//                    value of state lane 0 after the permutation — no byte reinterpretation.
//                    `permute` canonicalises, hence on device: valid ⇔ s[0] < limit.
//
// THE KERNEL (twin of keccak.cu:158 `grind_search`, same signature shape):
//   rpx_grind_search(const uint64_t *inner_felts /*4, BE*/, uint64_t limit, uint64_t base,
//                    uint64_t count, volatile unsigned long long *result)
//   grid-stride over [base, base+count); per candidate: build the 12-lane state above, ONE
//   `permute` call (it stays __noinline__), `if (s[0] < limit) atomicMin(result, nonce)`;
//   early exit `if (nonce >= *result) break;` and the u64-wrap guard exactly as keccak's.
//   Host launcher `rpx::generate_nonce_gpu(&[u64; 4] /*BE felts*/, factor)` mirrors
//   `grinding::generate_nonce_gpu` (min factor 12 → CPU, count = clamp(8·2^f, 2^18, 2^28),
//   grid 1024×256, sentinel loop, None on any error). Dispatch arm in stark grinding.rs keyed
//   through crate::config (GrindingDigest<DefaultStarkHash> + COMMITMENT_HASH == Rpx256), so
//   the pin decides and no prover type is named from the stark crate.
//
// PREDICTION (before the box build): entry ≈ 120–200 PTX lines (state setup, one call, one
// compare, one atomic, the loop); the permute `.func` is shared, so rpx.ptx grows from 6,241 to
// ≈ 6,450 lines; ≈ 40 registers like rpx_permute_probe, ≈ 200 B stack (the state crossing the
// call), 0 spills. Per table at factor 20 the expected work is ≈ 2^20 + 2^18 permutations
// (the smallest valid nonce plus the in-flight tail) ≈ 4·10^9 field multiplications, i.e.
// of order 10 ms on a 5090; the host's rayon `find_any` on 48 cores is of order 50–150 ms.
// Both are measurements to be taken ×10 on one box, never quoted from here.
// ===========================================================================
