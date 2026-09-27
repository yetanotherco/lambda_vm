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
#define RPX_LAUNCH_BOUNDS(n) __launch_bounds__(n)
#elif defined(RPX_HOST_NOINLINE)
#define RPX_NOINLINE __attribute__((noinline))
#define RPX_LAUNCH_BOUNDS(n)
#else
#define RPX_NOINLINE
#define RPX_LAUNCH_BOUNDS(n)
#endif

// ★ GAP K5 — which field multiply the permutation runs, as a compile-time bit
// set. 0 is the shipped arithmetic (`goldilocks::mul` for every product and
// square), and every kernel in a cubin uses `RPX_PERMUTE_VARIANT`. build.rs
// compiles this file as `rpx.cubin` at 0 and as `rpx_k5_v<V>.cubin` at each
// candidate V; `LAMBDA_VM_GAP_K5=<V>` loads the latter. The bits:
//   RPX_V_LIMB_MUL     products through `mul_limb` (32-bit limbs, carry chains);
//   RPX_V_LIMB_SQR     squarings through `sqr_limb` (three limb products);
//   RPX_V_UNROLL_SQN   `square_n` unrolled by four;
//   RPX_V_UNROLL_LANES the FB round's twelve lane loops unrolled.
// ⚠ Every variant computes the same FIELD values; only the raw representation
// of intermediates may differ, and `permute` canonicalises its output, so
// digests, nodes and grind heads are bit-identical across variants.
#define RPX_V_LIMB_MUL 1
#define RPX_V_LIMB_SQR 2
#define RPX_V_UNROLL_SQN 4
#define RPX_V_UNROLL_LANES 8
#ifndef RPX_PERMUTE_VARIANT
#define RPX_PERMUTE_VARIANT 0
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

// ---------------------------------------------------------------------------
// GAP K5: the 32-bit-limb multiply. The 128-bit product is built from four
// 32×32 partial products with the carries in the add chain, then folded to
// `[0, 2^64)` in three steps (the reduction sppark's `gl64_t` ships, Apache-2.0,
// which pil2-stark uses): `2^64 ≡ 2^32 − 1` and `2^96 ≡ −1 (mod p)`.
//
// The result is congruent to `a·b` and below 2^64, NOT canonical, and may differ
// in representation from `goldilocks::mul`'s — which is why only this file uses
// it: `permute` canonicalises its output, whereas the NTT/LDE parity tests
// compare `goldilocks::mul`'s raw words.
//
// Why the steps cannot wrap (inputs anywhere in `[0, 2^64)`): limbs t0..t3 of
// the product; step 1 forms `c·2^64 + s1 = (t1:t0) + t2·(2^32 − 1)` with
// `c ∈ {0, 1}`, and `c = 1` forces `s1 < 2^64 − 2^33 + 1`; step 2 subtracts t3
// with the borrow going into `c`, leaving `c ∈ {−1, 0, 1}`; step 3 subtracts
// `c·p` as ONE 64-bit subtraction: `c = 1` subtracts p (≡ adding 2^32 − 1 mod
// 2^64, and s2 ≤ s1 < 2^64 − 2^33 + 1 leaves room), `c = −1` subtracts
// 2^32 − 1 (s2 = s1 − t3 + 2^64 > 2^64 − 2^32 there, so no borrow).
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint64_t reduce_limbs(uint32_t t0, uint32_t t1, uint32_t t2,
                                                 uint32_t t3) {
#if defined(__CUDA_ARCH__)
    const uint32_t w = 0xFFFFFFFFu;
    // Step 1: (t1:t0) += t2·(2^32 − 1); the carry out replaces t2.
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1; addc.u32 %2, 0, 0;"
        : "+r"(t0), "+r"(t1), "+r"(t2)
        : "r"(w));
    // Step 2: − t3, the borrow running into t2.
    asm("sub.cc.u32 %0, %0, %3; subc.cc.u32 %1, %1, 0; subc.u32 %2, %2, 0;"
        : "+r"(t0), "+r"(t1), "+r"(t2)
        : "r"(t3));
    // Step 3: − c·p with c = t2 ∈ {0, 1, 0xFFFFFFFF}: (c, 0xFFFFFFFF) for c = 1,
    // (0xFFFFFFFF, 0) for c = −1, nothing for c = 0.
    const uint32_t hi_sub = (t2 == 1u) ? 0xFFFFFFFFu : 0u;
    asm("sub.cc.u32 %0, %0, %2; subc.u32 %1, %1, %3;" : "+r"(t0), "+r"(t1) : "r"(t2), "r"(hi_sub));
    return ((uint64_t)t1 << 32) | t0;
#else
    // The same three steps in portable C, for the host KAT.
    const uint64_t s = ((uint64_t)t1 << 32) | t0;
    const uint64_t s1 = s + (uint64_t)t2 * 0xFFFFFFFFull;
    int c = (s1 < s) ? 1 : 0;
    const uint64_t s2 = s1 - t3;
    c -= (s1 < t3) ? 1 : 0;
    if (c == 1) return s2 + 0xFFFFFFFFull;
    if (c == -1) return s2 - 0xFFFFFFFFull;
    return s2;
#endif
}

__device__ __forceinline__ uint64_t mul_limb(uint64_t a, uint64_t b) {
    const uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32);
    const uint32_t b0 = (uint32_t)b, b1 = (uint32_t)(b >> 32);
#if defined(__CUDA_ARCH__)
    uint32_t t0, t1, t2, t3, c;
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(t0), "=r"(t1) : "r"(a0), "r"(b0));
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(t2), "=r"(t3) : "r"(a1), "r"(b1));
    asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, 0, 0;"
        : "+r"(t1), "+r"(t2), "=r"(c)
        : "r"(a0), "r"(b1));
    asm("mad.lo.cc.u32 %0, %3, %4, %0; madc.hi.cc.u32 %1, %3, %4, %1; addc.u32 %2, %2, %5;"
        : "+r"(t1), "+r"(t2), "+r"(t3)
        : "r"(a1), "r"(b0), "r"(c));
    return reduce_limbs(t0, t1, t2, t3);
#else
    const uint64_t p00 = (uint64_t)a0 * b0, p01 = (uint64_t)a0 * b1;
    const uint64_t p10 = (uint64_t)a1 * b0, p11 = (uint64_t)a1 * b1;
    const uint64_t mid = (p00 >> 32) + (uint32_t)p01 + (uint32_t)p10;
    const uint64_t top = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return reduce_limbs((uint32_t)p00, (uint32_t)mid, (uint32_t)top, (uint32_t)(top >> 32));
#endif
}

// `a²` with the cross product `a0·a1` formed once and added twice: three limb
// products where `mul_limb` needs four, paid for in adds.
__device__ __forceinline__ uint64_t sqr_limb(uint64_t a) {
    const uint32_t a0 = (uint32_t)a, a1 = (uint32_t)(a >> 32);
#if defined(__CUDA_ARCH__)
    uint32_t t0, t1, t2, t3, x0, x1;
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(t0), "=r"(t1) : "r"(a0), "r"(a0));
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(t2), "=r"(t3) : "r"(a1), "r"(a1));
    asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;" : "=r"(x0), "=r"(x1) : "r"(a0), "r"(a1));
    asm("add.cc.u32 %0, %0, %3; addc.cc.u32 %1, %1, %4; addc.u32 %2, %2, 0;"
        : "+r"(t1), "+r"(t2), "+r"(t3)
        : "r"(x0), "r"(x1));
    asm("add.cc.u32 %0, %0, %3; addc.cc.u32 %1, %1, %4; addc.u32 %2, %2, 0;"
        : "+r"(t1), "+r"(t2), "+r"(t3)
        : "r"(x0), "r"(x1));
    return reduce_limbs(t0, t1, t2, t3);
#else
    const uint64_t p00 = (uint64_t)a0 * a0, x = (uint64_t)a0 * a1, p11 = (uint64_t)a1 * a1;
    const uint64_t mid = (p00 >> 32) + 2 * (uint64_t)(uint32_t)x;
    const uint64_t top = p11 + 2 * (x >> 32) + (mid >> 32);
    return reduce_limbs((uint32_t)p00, (uint32_t)mid, (uint32_t)top, (uint32_t)(top >> 32));
#endif
}

// The multiply and the square as the variant `V` spells them. Both count as
// one multiplication for the host cost model.
template <int V>
__device__ __forceinline__ uint64_t fmul_v(uint64_t a, uint64_t b) {
    RPX_COUNT(mul);
    if constexpr ((V & RPX_V_LIMB_MUL) != 0) {
        return mul_limb(a, b);
    } else {
        return goldilocks::mul(a, b);
    }
}

template <int V>
__device__ __forceinline__ uint64_t fsqr_v(uint64_t a) {
    if constexpr ((V & RPX_V_LIMB_SQR) != 0) {
        RPX_COUNT(mul);
        return sqr_limb(a);
    } else {
        return fmul_v<V>(a, a);
    }
}

__device__ __forceinline__ uint64_t fmul(uint64_t a, uint64_t b) {
    return fmul_v<RPX_PERMUTE_VARIANT>(a, b);
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
template <int V>
__device__ __forceinline__ uint64_t sbox_v(uint64_t x) {
    const uint64_t x2 = fsqr_v<V>(x);
    const uint64_t x3 = fmul_v<V>(x2, x);
    const uint64_t x6 = fsqr_v<V>(x3);
    return fmul_v<V>(x6, x);
}

template <int V, int N>
__device__ __forceinline__ uint64_t square_n_v(uint64_t x) {
    // Rolled: the chain is serial anyway, and unrolled it is what made one
    // permutation ~49k lines of PTX. The unroll factor here is a tuning knob
    // (GAP K5's RPX_V_UNROLL_SQN).
    if constexpr ((V & RPX_V_UNROLL_SQN) != 0) {
#pragma unroll 4
        for (int i = 0; i < N; ++i) x = fsqr_v<V>(x);
    } else {
#pragma unroll 1
        for (int i = 0; i < N; ++i) x = fsqr_v<V>(x);
    }
    return x;
}

// `base^(2^M) · tail` — the inverse chain's one building block (rpo.rs:483-495).
template <int V, int M>
__device__ __forceinline__ uint64_t exp_acc_v(uint64_t base, uint64_t tail) {
    return fmul_v<V>(square_n_v<V, M>(base), tail);
}

// `x^{1/7} = x^10540996611094048183` by miden-crypto's addition chain, as
// `rpo.rs:481-509` runs it lane-wise: 63 squarings + 9 products = 72
// multiplications against ~93 for square-and-multiply. Per lane rather than
// whole-state: on a GPU the twelve lanes' independence is the compiler's to
// interleave, and a lane-wise body keeps only six values live.
template <int V>
__device__ __forceinline__ uint64_t inv_sbox_v(uint64_t x) {
    const uint64_t t1 = fsqr_v<V>(x);                 // x^2
    const uint64_t t2 = fsqr_v<V>(t1);                // x^4
    const uint64_t t3 = exp_acc_v<V, 3>(t2, t2);      // x^36
    const uint64_t t4 = exp_acc_v<V, 6>(t3, t3);      // x^(36·65)
    const uint64_t t5 = exp_acc_v<V, 12>(t4, t4);     // x^(36·65·4097)
    const uint64_t t6 = exp_acc_v<V, 6>(t5, t3);      // x^0x24924924
    const uint64_t t7 = exp_acc_v<V, 31>(t6, t6);     // x^0x1249249224924924
    // ((t7² · t6)²)² · ((t1 · t2) · x)  — rpo.rs:504-508.
    const uint64_t a = square_n_v<V, 2>(fmul_v<V>(fsqr_v<V>(t7), t6));
    const uint64_t b = fmul_v<V>(fmul_v<V>(t1, t2), x);
    return fmul_v<V>(a, b);
}

// The shipped names, at this cubin's variant (the host KAT calls these).
__device__ __forceinline__ uint64_t sbox(uint64_t x) { return sbox_v<RPX_PERMUTE_VARIANT>(x); }

template <int N>
__device__ __forceinline__ uint64_t square_n(uint64_t x) {
    return square_n_v<RPX_PERMUTE_VARIANT, N>(x);
}

template <int M>
__device__ __forceinline__ uint64_t exp_acc(uint64_t base, uint64_t tail) {
    return exp_acc_v<RPX_PERMUTE_VARIANT, M>(base, tail);
}

__device__ __forceinline__ uint64_t inv_sbox(uint64_t x) {
    return inv_sbox_v<RPX_PERMUTE_VARIANT>(x);
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
template <int V>
__device__ __forceinline__ void fb_round_v(uint64_t s[STATE_FELTS], int r) {
    mds(s);
    if constexpr ((V & RPX_V_UNROLL_LANES) != 0) {
#pragma unroll
        for (int i = 0; i < STATE_FELTS; ++i) s[i] = sbox_v<V>(fadd(s[i], ARK1[r][i]));
        mds(s);
#pragma unroll
        for (int i = 0; i < STATE_FELTS; ++i) s[i] = inv_sbox_v<V>(fadd(s[i], ARK2[r][i]));
    } else {
#pragma unroll 1
        for (int i = 0; i < STATE_FELTS; ++i) s[i] = fadd(s[i], ARK1[r][i]);
#pragma unroll 1
        for (int i = 0; i < STATE_FELTS; ++i) s[i] = sbox_v<V>(s[i]);
        mds(s);
#pragma unroll 1
        for (int i = 0; i < STATE_FELTS; ++i) s[i] = fadd(s[i], ARK2[r][i]);
        // The twelve chains are independent; a GPU hides their latency with
        // other warps, not by unrolling one thread's twelve chains into
        // straight line.
#pragma unroll 1
        for (int i = 0; i < STATE_FELTS; ++i) s[i] = inv_sbox_v<V>(s[i]);
    }
}

__device__ __forceinline__ void fb_round(uint64_t s[STATE_FELTS], int r) {
    fb_round_v<RPX_PERMUTE_VARIANT>(s, r);
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
//
// `permute_v<V>` is the permutation at GAP K5 variant V; every instantiation
// is its own called body. `permute` — what every kernel calls — is this
// cubin's variant.
template <int V>
RPX_NOINLINE __device__ void permute_v(uint64_t s[STATE_FELTS]) {
#pragma unroll 1
    for (int r = 0; r + 1 < NUM_ROUNDS; r += 2) {
        fb_round_v<V>(s, r);
        ext_round(s, r + 1);
    }
    final_round(s, NUM_ROUNDS - 1);
#pragma unroll 1
    for (int i = 0; i < STATE_FELTS; ++i) s[i] = goldilocks::canonical(s[i]);
}

__device__ __forceinline__ void permute(uint64_t s[STATE_FELTS]) {
    permute_v<RPX_PERMUTE_VARIANT>(s);
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

// FRI GROUP-leaf hashing (S3): leaf `tid` absorbs the `group` consecutive ext3
// values `evals[tid*group .. (tid+1)*group]` of an interleaved eval vector — the
// `3*group` contiguous felts at `evals_interleaved + tid*group*3` — in order, a
// sponge over `3*group` felts (the count keys the padding). The host
// `AlgebraicBatchBackend` leaf over the group; at `group = 2` exactly
// `rpx_fri_leaves_ext3`'s six felts. Twin of `keccak_fri_group_leaves_ext3`.
extern "C" __global__ void rpx_fri_group_leaves_ext3(
    const uint64_t *evals_interleaved,  // 3 * num_leaves * group u64s
    uint64_t num_leaves,
    uint64_t group,                      // ext3 values per leaf (2^d)
    uint8_t *leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;
    const uint64_t *g = evals_interleaved + tid * group * 3;

    rpx::Sponge sp;
    sp.init(3 * group);
    for (uint64_t i = 0; i < 3 * group; ++i) sp.absorb(g[i]);
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

// Row-major ONE-ROW leaf hashing (S2, rows_per_leaf = 1): leaf `tid` absorbs the
// single row `reverse_index(tid)`, columns `[col_start, col_end)` of the
// row-major buffer (`m` the full row stride) — a sponge over
// `col_end - col_start` felts (the count keys the padding). The CPU
// `commit_rows_bit_reversed_subset_with(.., 1)`. Twin of
// `keccak256_leaves_base_row_major_row_range`.
extern "C" __global__ void rpx_leaves_base_row_major_row_range(
    const uint64_t *data,
    uint64_t m,
    uint64_t col_start,
    uint64_t col_end,
    uint64_t num_rows,
    uint64_t log_num_rows,
    uint8_t *hashed_leaves_out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_rows) return;
    uint64_t br = __brevll(tid) >> (64 - log_num_rows);
    const uint64_t *row = data + br * m;

    rpx::Sponge sp;
    sp.init(col_end - col_start);
    for (uint64_t c = col_start; c < col_end; ++c) sp.absorb(row[c]);
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, hashed_leaves_out + tid * 32);
}

// ---------------------------------------------------------------------------
// COSET leaf hashing — the WHIR shape, and the two kernels the per-table branch
// has no twin for.
//
// Every other leaf kernel in this file hashes a ROW GROUP: a leaf is a row (or
// a row pair) read across the columns. WHIR's leaf is a fold COSET: leaf `j`
// holds the `2^log_folding` codeword positions that fold onto `j`, which are
// strided by `num_leaves`. Twins of `keccak256_leaves_base_coset` /
// `keccak256_leaves_ext3_coset` (`keccak.cu`), argument for argument, with the
// sponge swapped.
//
// The felt count is known before the loop, which the overwrite duplex needs for
// its padding flag: a base leaf is `block` felts, an ext3 leaf `3 * block`. Raw
// `[0, 2^64)` storage is absorbed as is — the permutation is
// representation-independent and the host canonicalises before serialising, so
// the same field value gives the same digest either way.
// ---------------------------------------------------------------------------

// Goldilocks BASE-FIELD coset leaves: leaf `tid` hashes
// `codeword[tid + t * num_leaves]` for `t` in `[0, block)`.
extern "C" __global__ void rpx_leaves_base_coset(const uint64_t *__restrict__ codeword,
                                                 uint64_t num_leaves, uint64_t block,
                                                 uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;

    rpx::Sponge sp;
    sp.init(block);
    for (uint64_t t = 0; t < block; ++t) sp.absorb(codeword[tid + t * num_leaves]);
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, out + tid * 32);
}

// EXT3 coset leaves: the same stride, each element as its three components in
// order — what `element_felts` produces for a cubic-extension element, and what
// the base kernel above does one component at a time.
extern "C" __global__ void rpx_leaves_ext3_coset(const uint64_t *__restrict__ codeword,
                                                 uint64_t num_leaves, uint64_t block,
                                                 uint8_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_leaves) return;

    rpx::Sponge sp;
    sp.init(block * 3);
    for (uint64_t t = 0; t < block; ++t) {
        const uint64_t *at = codeword + (tid + t * num_leaves) * 3;
#pragma unroll
        for (int k = 0; k < 3; ++k) sp.absorb(at[k]);
    }
    uint64_t digest[rpx::DIGEST_FELTS];
    sp.finalize(digest);
    rpx::store_digest_be(digest, out + tid * 32);
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

// ---------------------------------------------------------------------------
// Proof-of-work grinding search, RPX arm.
//
// Twin of `keccak.cu`'s `grind_search`, same signature shape and the same
// first-hit reduction; only the outer hash differs. The host path it replaces
// is `stark::grinding::generate_nonce`, a per-table ~2^grinding_factor search
// that is the prover's dominant CPU cost once the transcript is algebraic.
//
// THE MAPPING this reproduces, derived from the host (✓ VERIFIED against the
// sources named) and stated here so it is not re-derived at each reading:
//
//   host predicate   stark/src/grinding.rs::is_valid_nonce_for_inner_hash:
//                    valid ⇔ u64::from_be_bytes(D::digest(inner ‖ nonce.to_be_bytes())[..8]) < limit,
//                    limit = 1 << (64 − grinding_factor); inner = D::digest(PREFIX ‖ seed ‖ factor),
//                    41 bytes, computed ONCE per table on the host and never on device.
//   D for RPX        prover/src/lfm/algebraic_commit.rs AlgebraicDigest<RpxCommit>:
//                    D::digest(bytes) = digest_to_commitment(sponge_leaf(Rpx, felts_from_bytes(bytes)))
//                    — the LEAF construction, on purpose.
//   bytes → felts    felts_from_bytes: consecutive 8-byte groups, each
//                    FE::from(u64::from_be_bytes(group)) — BIG-endian, and `FE::from` is
//                    `from_u64`, which maps a raw value ≥ p to raw − p (ONE subtraction,
//                    goldilocks.rs:172-178 — exactly `goldilocks::canonical`).
//                    The 40-byte outer block is therefore EXACTLY five felts:
//                        f0..f3 = the inner hash's four big-endian u64s (canonical already —
//                                 they are digest_to_commitment output, so each < p),
//                        f4     = the nonce.
//                    ⚠ Big-endian, unlike keccak's `inner_hash_lanes` (LITTLE-endian lanes).
//                    A separate host helper, `stark::grinding::inner_hash_felts` (BE), feeds
//                    this kernel; feeding it the keccak lanes is a silent wrong hash.
//   sponge mode      sponge_leaf over five felts: ONE permutation of
//                        [f0, f1, f2, f3, nonce, 0, 0, 0 | 5, 0x4C4D464C, 0, 0]
//                    — rate lanes 5..8 zero-padded, capacity lane 8 = padding flag
//                    `5 mod 8 = 5`, lane 9 = DOMAIN_LEAF ("LFML"), lanes 10, 11 = 0.
//                    ★ Built here through `rpx::Sponge` — `init(5)`, five `absorb`s,
//                    `finalize` — rather than by writing those twelve lanes out, so the
//                    capacity rule has ONE statement on device and a change to it cannot
//                    leave the grind behind.
//   the head         digest_to_commitment writes lane 0 CANONICAL as 8 big-endian bytes and
//                    the host reads those 8 bytes back big-endian, so `seed_head` IS the
//                    canonical value of state lane 0 after the permutation — no byte
//                    reinterpretation. `permute` canonicalises its output, so on device the
//                    predicate is just `digest[0] < limit`.
//
// THE NONCE LANE is `goldilocks::canonical(nonce)`, matching `FE::from(nonce)`
// exactly. It is a no-op for every nonce this search can reach (the first
// nonce ≥ p is 2^64 − 2^32 + 1, and the launcher's range walk bails long
// before), and the device representation is lazy anyway — but absorbing the
// canonical value is what makes "the device absorbs what `FE::from` produces"
// true by inspection rather than by an argument about reachability.
//
// Each thread strides over `[base, base+count)` and `atomicMin`s the smallest
// valid nonce it finds into `*result` (initialised to U64_MAX by the caller),
// so the launch returns the globally smallest valid nonce in the searched
// block — deterministic despite the parallel grid, and any valid nonce
// satisfies the verifier.
// ---------------------------------------------------------------------------

// The outer block is `inner_hash ‖ nonce`: 40 bytes, five felts. Named because
// the capacity's padding flag is `5 mod 8` and the count is what `init` needs.
__device__ constexpr uint64_t GRIND_FELTS = 5;

extern "C" __global__ void rpx_grind_search(const uint64_t *inner_felts,
                                            uint64_t limit,
                                            uint64_t base,
                                            uint64_t count,
                                            volatile unsigned long long *result) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t f0 = inner_felts[0], f1 = inner_felts[1], f2 = inner_felts[2],
                   f3 = inner_felts[3];
    for (uint64_t i = tid; i < count; i += stride) {
        uint64_t nonce = base + i;
        // Guard the u64 wrap on the final block (the launcher bails before it,
        // so this is unreachable in practice): a wrapped nonce is < base, so
        // stop rather than re-scan from 0.
        if (nonce < base) break;
        // A thread's nonces only increase, so once a smaller valid one is known
        // this thread can never beat it — stop scanning. `result` is volatile
        // so this load re-reads L2 (where the atomicMin writes land) instead of
        // being hoisted into a register or served stale from L1; the early exit
        // depends on that, though correctness does not.
        if (nonce >= (uint64_t)*result) break;
        rpx::Sponge sp;
        sp.init(GRIND_FELTS);
        sp.absorb(f0);
        sp.absorb(f1);
        sp.absorb(f2);
        sp.absorb(f3);
        sp.absorb(goldilocks::canonical(nonce));
        uint64_t digest[rpx::DIGEST_FELTS];
        sp.finalize(digest);
        if (digest[0] < limit) {
            atomicMin((unsigned long long *)result, (unsigned long long)nonce);
        }
    }
}

// ---------------------------------------------------------------------------
// ⛔ A DIAGNOSTIC TWIN, NEVER A PROVING PATH.
//
// `rpx_grind_search` above is the shipped kernel and this file does not change
// it. This twin is the same search with three counters, and it exists to answer
// ONE question that no timing can: when a launch is slow, does it EXECUTE MORE
// PERMUTATIONS, or the same ones more slowly?
//
// The question is not idle. The paired microbench reads a mean 20% above the
// model on the record posture while the MEDIAN seed sits on it, and the excess
// saturates in `count` rather than growing with it: converted to iterations per
// thread the excess fits `T·(1 − e^(−(N−8)/τ))` with τ ≈ 19 on three
// independent points. That is the shape of a thread that keeps scanning for a
// bounded TIME after the answer is known — a poll of `*result` served stale —
// and not the shape of a thread running to the loop bound. `executed` settles
// it: stale polls mean real extra permutations, bounded by time and therefore
// the SAME at scan 8 and scan 64; anything else means the work is unchanged and
// the cost is outside this loop.
//
// ⚠ WASTED WORK, NEVER A WRONG ANSWER. The search returns the globally smallest
// valid nonce whatever any thread does after the `atomicMin`, which is why the
// counters can be read at leisure while the answer stays pinned by
// `gpu_grind_returns_smallest_valid_nonce`. Nothing here is a soundness matter.
//
// ⛔ `#if defined(__CUDACC__)`, and that is deliberate rather than defensive:
// the host-KAT compiles this file through `cuda_host_shim.h`, which supplies
// `gridDim`, `blockDim` and `atomicMin` but NOT `__shfl_down_sync`. A KAT has
// no answer to check here anyway — this kernel produces no digest — and its
// nonce agreement with the shipped kernel is asserted on the device, seed by
// seed, by the bench that launches it.
//
// SIZING, before the first cubin, by the rule this file learnt the hard way
// (`permute` stays a called function; see its CODE SHAPE note): one more
// `permute` CALL SITE, not one more inlined copy, so this adds an entry of
// order 150-250 PTX lines against a file of 6,241 — not a duplicated
// permutation body. If the `.ptx` grows by thousands, something inlined that
// must not.
#if defined(__CUDACC__)

// One warp's reduction of a sum and a max, so the counters cost four atomics a
// warp instead of one per thread.
//
// ⛔ A per-thread `atomicAdd` would be 131,072 serialised updates of one L2
// line per launch — the same order as the 5-11 ms effect being measured. An
// instrument that manufactures its own signal answers a different question.
// Every thread in a warp reaches this (the loop's `break`s leave the loop, not
// the function), so the full mask is correct.
// ⛔ `unsigned long long`, NOT `uint64_t`, and that is not a style choice. The
// shuffle intrinsics are overloaded on `int`, `unsigned int`, `long long`,
// `unsigned long long`, `float` and `double`. On an LP64 host `uint64_t` is
// `unsigned long`, which is NONE of them — the call would be ambiguous or
// absent rather than wrong, so it fails at compile time on the box and not
// here, where no nvcc runs. The file already casts for exactly this reason
// where it calls `atomicMin`.
__device__ __forceinline__ void warp_reduce_counts(unsigned long long &sum,
                                                   unsigned long long &max_v,
                                                   unsigned long long &ends) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, off);
        ends += __shfl_down_sync(0xffffffffu, ends, off);
        const unsigned long long other = __shfl_down_sync(0xffffffffu, max_v, off);
        if (other > max_v) max_v = other;
    }
}

// The three counters live in ONE device array so a search reads them back in a
// single copy: `counts[COUNT_EXECUTED]` and `counts[COUNT_RAN_TO_END]` are
// accumulated with `atomicAdd` across every launch of the search,
// `counts[COUNT_MAX_ITERS]` with `atomicMax`. The host mirrors these names.
__device__ constexpr int COUNT_EXECUTED = 0;
__device__ constexpr int COUNT_MAX_ITERS = 1;
__device__ constexpr int COUNT_RAN_TO_END = 2;

// ⭐ THE POLL-RATE KNOB (`poll_period`, the RPX grind k-sweep's k). The shipped
// kernel polls `*result` every iteration; this twin polls it every
// `poll_period`-th iteration, STAGGERED across threads. It answers whether the
// stale-poll overrun is driven by CONTENTION on that one address: 131,072
// resident threads each issue a system-scope load to it every permutation, and
// the finder's `atomicMin` queues behind that flood. If lowering the poll rate
// makes the overrun FALL, contention is the cause and a warp- or block-level
// poll is the fix; if it RISES ~linearly in `poll_period`, polling less just
// wastes more scanning and the poll family is dead. `poll_period == 1`
// reproduces the shipped every-iteration poll exactly, which is the sweep's
// admissibility control. ⚠ Wasted work, never a wrong answer — the search still
// returns the globally smallest valid nonce whatever the poll rate; see the
// soundness note above.
extern "C" __global__ void rpx_grind_search_counted(const uint64_t *inner_felts,
                                                    uint64_t limit,
                                                    uint64_t base,
                                                    uint64_t count,
                                                    volatile unsigned long long *result,
                                                    unsigned long long *counts,
                                                    uint64_t poll_period) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    const uint64_t f0 = inner_felts[0], f1 = inner_felts[1], f2 = inner_felts[2],
                   f3 = inner_felts[3];
    // Permutations THIS thread ran. Counted after both exit tests and before
    // the sponge, so it counts work done and never work declined.
    uint64_t iters = 0;
    // Did this thread leave by the loop bound rather than by an early exit?
    // Only meaningful for a thread that did some work: a thread whose `tid` is
    // past `count` never enters the loop and must not be scored as having
    // scanned to the end.
    uint64_t to_end = 0;
    // ⭐ STAGGERED POLL. `poll_period` is a power of two (caller-guaranteed), so
    // the period is a bitmask; `(n + tid) & poll_mask == 0` fires once every
    // `poll_period` iterations with a phase that differs per thread, so at any
    // one iteration only 1/poll_period of threads issue the system-scope load.
    // A synchronised `n & poll_mask` would fire on the SAME iteration for every
    // thread, holding the peak request rate constant and changing only the duty
    // cycle. poll_mask == 0 (poll_period == 1) polls every iteration, identical
    // to the shipped kernel.
    const uint64_t poll_mask = poll_period - 1;
    // This thread's own loop-iteration counter, kept independent of `i / stride`
    // (that identity holds only while `tid < stride`; a counter cannot be broken
    // by a future change to the indexing).
    uint64_t n = 0;
    uint64_t i = tid;
    for (; i < count; i += stride) {
        uint64_t nonce = base + i;
        if (nonce < base) break;
        if (((n + tid) & poll_mask) == 0 && nonce >= (uint64_t)*result) break;
        ++n;
        ++iters;
        rpx::Sponge sp;
        sp.init(GRIND_FELTS);
        sp.absorb(f0);
        sp.absorb(f1);
        sp.absorb(f2);
        sp.absorb(f3);
        sp.absorb(goldilocks::canonical(nonce));
        uint64_t digest[rpx::DIGEST_FELTS];
        sp.finalize(digest);
        if (digest[0] < limit) {
            atomicMin((unsigned long long *)result, (unsigned long long)nonce);
        }
    }
    if (i >= count && iters > 0) to_end = 1;

    unsigned long long sum = (unsigned long long)iters;
    unsigned long long max_v = (unsigned long long)iters;
    unsigned long long ends = (unsigned long long)to_end;
    warp_reduce_counts(sum, max_v, ends);
    if ((threadIdx.x & 31u) == 0u) {
        atomicAdd(&counts[COUNT_EXECUTED], sum);
        atomicAdd(&counts[COUNT_RAN_TO_END], ends);
        atomicMax(&counts[COUNT_MAX_ITERS], max_v);
    }
}

#endif  // __CUDACC__

// ===========================================================================
// ★ GAP CAMPAIGN, HASH LANE — kernels behind LAMBDA_VM_GAP_K3 / K4 / K5.
//
// Nothing on a proving path launches these unless its knob is set; the probes
// are for the parity tests and the benches. Every kernel here produces the
// same bytes (nodes, nonces, digests) as the shipped kernel it stands in for.
// ===========================================================================

// ---------------------------------------------------------------------------
// K5 probes. `rpx_permute_chain_probe_v<V>` permutes each thread's state `k`
// times at variant V — chained, so a launch is compute rather than memory and
// a wrong step anywhere reaches the output. `rpx_limb_probe` returns the two
// limb primitives and `goldilocks::mul` side by side on raw pairs, so the PTX
// is checked against 128-bit arithmetic on the device itself. V is a template
// argument, so probe v0 is the shipped arithmetic in every cubin.
// ---------------------------------------------------------------------------
template <int V>
__device__ __forceinline__ void permute_chain_probe(const uint64_t *states, uint64_t n,
                                                    uint64_t k, uint64_t *out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t s[rpx::STATE_FELTS];
#pragma unroll
    for (int i = 0; i < rpx::STATE_FELTS; ++i) s[i] = states[tid * rpx::STATE_FELTS + i];
    for (uint64_t j = 0; j < k; ++j) rpx::permute_v<V>(s);
#pragma unroll
    for (int i = 0; i < rpx::STATE_FELTS; ++i) out[tid * rpx::STATE_FELTS + i] = s[i];
}

#define RPX_CHAIN_PROBE(V)                                                                   \
    extern "C" __global__ void rpx_permute_chain_probe_v##V(const uint64_t *states, uint64_t n, \
                                                            uint64_t k, uint64_t *out) {       \
        permute_chain_probe<V>(states, n, k, out);                                           \
    }
RPX_CHAIN_PROBE(0)
RPX_CHAIN_PROBE(1)
RPX_CHAIN_PROBE(3)
RPX_CHAIN_PROBE(4)
RPX_CHAIN_PROBE(5)
RPX_CHAIN_PROBE(7)
RPX_CHAIN_PROBE(9)
#undef RPX_CHAIN_PROBE

extern "C" __global__ void rpx_limb_probe(const uint64_t *a, const uint64_t *b, uint64_t n,
                                          uint64_t *out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    out[3 * tid] = rpx::mul_limb(a[tid], b[tid]);
    out[3 * tid + 1] = rpx::sqr_limb(a[tid]);
    out[3 * tid + 2] = goldilocks::mul(a[tid], b[tid]);
}

// The poll of a word other threads update with atomics: a volatile load on the
// device (the shipped grind's `LDG.E.64.STRONG.SYS`), an atomic load under the
// host SIMT shim, whose lanes are real threads.
__device__ __forceinline__ unsigned long long rpx_poll_u64(unsigned long long *p) {
#if defined(__CUDA_ARCH__)
    return *(volatile unsigned long long *)p;
#else
    return __atomic_load_n(p, __ATOMIC_SEQ_CST);
#endif
}

#if defined(__CUDACC__) || defined(RPX_HOST_SIMT)

// ---------------------------------------------------------------------------
// K4 — the grind with nonces claimed from a WORK QUEUE, in increasing order.
//
// The shipped kernel gives thread `t` the fixed nonces `t, t + stride, …`, so a
// warp that runs ahead scans nonces far above the answer while a slow warp is
// still below it; the search cannot stop until the slowest owner of a nonce
// below the answer gets there. The counted twin measured the cost: 3.30 strides
// of over-scan past `nonce + stride` at the record posture, with a ZERO
// contention component (the O6 poll k-sweep), i.e. skew, not stale polls.
//
// Here a warp claims the next 32 unclaimed nonces (one per lane) with ONE
// `atomicAdd` on a queue head, so the nonces are processed in (nearly)
// increasing order whatever the warps' speeds, and the waste is bounded by the
// work in flight when the answer lands — at most one permutation per lane.
//
// ★ THE ANSWER IS UNCHANGED: the smallest valid nonce in `[base, base+count)`,
// or the sentinel. The head only grows and the result only shrinks, to valid
// nonces. A warp stops when its claimed chunk starts at or above the result
// (every later chunk starts higher still) or past `count`. A chunk holding a
// nonce below the final result is therefore always claimed — the head passed
// it before any warp could stop above it — and a claimed chunk that starts
// below the result is hashed in full. So the launch returns what
// `rpx_grind_search` returns, which the parity test pins seed by seed.
//
// `state[0]` = the result (U64_MAX sentinel), `state[1]` = the queue head (an
// offset into `[0, count)`); the launcher resets both before each launch.
// `blockDim.x` must be a multiple of 32 (the claim is per warp).
// ---------------------------------------------------------------------------
__device__ constexpr int GRIND_Q_RESULT = 0;
__device__ constexpr int GRIND_Q_HEAD = 1;
__device__ constexpr unsigned long long GRIND_Q_CHUNK = 32;

__device__ __forceinline__ unsigned long long shfl_u64(unsigned long long v, int src) {
    return __shfl_sync(0xffffffffu, v, src);
}

// Warp butterfly for the counted twin: sum, max and the ran-to-end count, left
// on every lane. `__shfl_sync` only, so the host SIMT shim runs it too.
__device__ __forceinline__ void warp_sum_max(unsigned long long &sum, unsigned long long &max_v,
                                             unsigned long long &ends) {
    const int lane = (int)(threadIdx.x & 31u);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += shfl_u64(sum, lane ^ off);
        ends += shfl_u64(ends, lane ^ off);
        const unsigned long long other = shfl_u64(max_v, lane ^ off);
        if (other > max_v) max_v = other;
    }
}

template <bool COUNTED>
__device__ __forceinline__ void grind_queue(const uint64_t *inner_felts, uint64_t limit,
                                            uint64_t base, uint64_t count,
                                            unsigned long long *state,
                                            unsigned long long *counts) {
    const unsigned lane = threadIdx.x & 31u;
    const uint64_t f0 = inner_felts[0], f1 = inner_felts[1], f2 = inner_felts[2],
                   f3 = inner_felts[3];
    unsigned long long iters = 0, to_end = 0;
    for (;;) {
        unsigned long long off = 0, best = 0;
        if (lane == 0) {
            off = atomicAdd(&state[GRIND_Q_HEAD], GRIND_Q_CHUNK);
            best = rpx_poll_u64(&state[GRIND_Q_RESULT]);
        }
        off = shfl_u64(off, 0);
        best = shfl_u64(best, 0);
        if (off >= count) {
            to_end = iters > 0 ? 1 : 0;
            break;
        }
        const uint64_t first = base + off;
        // `first < base`: the u64 wrap on the final block, unreachable in
        // practice (the launcher bails first), as in the shipped kernel.
        if (first < base || first >= best) break;
        const uint64_t i = off + lane;
        if (i < count) {
            const uint64_t nonce = base + i;
            rpx::Sponge sp;
            sp.init(GRIND_FELTS);
            sp.absorb(f0);
            sp.absorb(f1);
            sp.absorb(f2);
            sp.absorb(f3);
            sp.absorb(goldilocks::canonical(nonce));
            uint64_t digest[rpx::DIGEST_FELTS];
            sp.finalize(digest);
            if (digest[0] < limit) {
                atomicMin(&state[GRIND_Q_RESULT], (unsigned long long)nonce);
            }
            if constexpr (COUNTED) ++iters;
        }
    }
    if constexpr (COUNTED) {
        unsigned long long sum = iters, max_v = iters, ends = to_end;
        warp_sum_max(sum, max_v, ends);
        if (lane == 0) {
            atomicAdd(&counts[0], sum);
            atomicMax(&counts[1], max_v);
            atomicAdd(&counts[2], ends);
        }
    }
}

extern "C" __global__ void rpx_grind_search_queue(const uint64_t *inner_felts, uint64_t limit,
                                                  uint64_t base, uint64_t count,
                                                  unsigned long long *state) {
    grind_queue<false>(inner_felts, limit, base, count, state, nullptr);
}

// ⛔ DIAGNOSTIC twin: the same search plus the counters `rpx_grind_search_counted`
// keeps, in the same slots (executed, max iterations, ran to end).
extern "C" __global__ void rpx_grind_search_queue_counted(const uint64_t *inner_felts,
                                                          uint64_t limit, uint64_t base,
                                                          uint64_t count,
                                                          unsigned long long *state,
                                                          unsigned long long *counts) {
    grind_queue<true>(inner_felts, limit, base, count, state, counts);
}

// ---------------------------------------------------------------------------
// K3 — one permutation across the lanes of a HALF-WARP.
//
// Narrow Merkle levels and the tail are latency-bound: one thread runs one
// permutation as a single dependent chain (~119 µs at the block's clock),
// whatever the width. Here lane `e` of a 16-lane half holds state element `e`,
// so the twelve inverse S-box chains — 95% of the multiplications — run side by
// side, and the MDS and the cubic-extension rounds read the other elements by
// `__shfl_sync`. Lanes 12-15 mirror elements 0-3 (they compute the same values
// and never store), so every shuffle has its whole warp and nothing diverges.
//
// ★ SAME WORDS as `permute`: every element sees the same field operations in
// the same order — `mds` per output lane is the same twelve exact integer terms
// (summed in another order; each half-sum is < 2^40, so no reduction happens
// before the end), and the E round's triple is gathered and raised by the same
// `ext_power7`, each of its three lanes keeping its own coefficient.
// ---------------------------------------------------------------------------
namespace rpx {

struct WarpLane {
    int e;      // this lane's state element, 0..11 (idle lanes mirror 0..3)
    int group;  // the half-warp's first lane: 0 or 16
    int tri;    // the first lane of this element's cubic-extension triple
    int k;      // e mod 3: which coefficient of the triple this lane keeps
    bool idle;  // lanes 12..15 of a half
};

__device__ __forceinline__ WarpLane warp_lane() {
    const int lane = (int)(threadIdx.x & 31u);
    const int l16 = lane & 15;
    WarpLane w;
    w.idle = l16 >= STATE_FELTS;
    w.e = w.idle ? l16 - STATE_FELTS : l16;
    w.group = lane & 16;
    w.tri = w.group + EXT_DEGREE * (w.e / EXT_DEGREE);
    w.k = w.e % EXT_DEGREE;
    return w;
}

// `mds` for this lane's element. `out_i = Σ_j ROW[(j − i) mod 12]·s_j` with
// `j = (e + k) mod 12` is `Σ_k ROW[k]·s_{(e+k) mod 12}`: the coefficient is
// warp-uniform (a compile-time index) and the source lane varies instead.
__device__ __forceinline__ uint64_t mds_lane(uint64_t v, const WarpLane &w) {
    uint64_t acc_lo = 0, acc_hi = 0;  // Σ c·l_j and Σ c·h_j, each < 2^40
#pragma unroll
    for (int k = 0; k < STATE_FELTS; ++k) {
        int j = w.e + k;
        if (j >= STATE_FELTS) j -= STATE_FELTS;
        const uint64_t sj = shfl_u64(v, w.group + j);
        const uint64_t c = MDS_CIRC_ROW2[k];
        acc_lo += c * (uint64_t)(uint32_t)sj;
        acc_hi += c * (uint64_t)(uint32_t)(sj >> 32);
    }
    const uint64_t lo = (acc_hi << 32) + acc_lo;
    const uint64_t carry = (lo < acc_lo) ? 1ull : 0ull;
    const uint64_t hi = (acc_hi >> 32) + carry;
    return fadd(lo, hi * goldilocks::EPSILON);
}

template <int V>
__device__ __forceinline__ uint64_t fb_round_lane(uint64_t v, int r, const WarpLane &w) {
    v = mds_lane(v, w);
    v = sbox_v<V>(fadd(v, ARK1[r][w.e]));
    v = mds_lane(v, w);
    return inv_sbox_v<V>(fadd(v, ARK2[r][w.e]));
}

__device__ __forceinline__ uint64_t ext_round_lane(uint64_t v, int r, const WarpLane &w) {
    v = fadd(v, ARK1[r][w.e]);
    CubicExt x;
    x.c0 = shfl_u64(v, w.tri);
    x.c1 = shfl_u64(v, w.tri + 1);
    x.c2 = shfl_u64(v, w.tri + 2);
    const CubicExt y = ext_power7(x);
    return w.k == 0 ? y.c0 : (w.k == 1 ? y.c1 : y.c2);
}

// `permute_v<V>` with the state spread over a half-warp: this lane's element
// in, this lane's element out, canonical.
template <int V>
__device__ __forceinline__ uint64_t permute_warp_v(uint64_t v, const WarpLane &w) {
#pragma unroll 1
    for (int r = 0; r + 1 < NUM_ROUNDS; r += 2) {
        v = fb_round_lane<V>(v, r, w);
        v = ext_round_lane(v, r + 1, w);
    }
    v = mds_lane(v, w);
    v = fadd(v, ARK1[NUM_ROUNDS - 1][w.e]);
    return goldilocks::canonical(v);
}

// This lane's element of `compress`'s state `[left ‖ right ‖ capacity]`. The
// two children are adjacent nodes, so elements 0..7 are the eight big-endian
// felts of the 64 bytes at `left_node`; the capacity is zero but for the
// compress domain (itself zero).
__device__ __forceinline__ uint64_t load_compress_lane(const uint8_t *nodes, uint64_t left_node,
                                                       int e) {
    if (e == CAPACITY_DOMAIN_LANE) return DOMAIN_COMPRESS;
    if (e >= 2 * DIGEST_FELTS) return 0;
    const uint64_t *src = reinterpret_cast<const uint64_t *>(nodes + left_node * 32);
    return bswap64(src[e]);
}

// One parent from the lanes holding its digest: `store_digest_be`, a felt per lane.
__device__ __forceinline__ void store_parent_lane(uint8_t *nodes, uint64_t parent, uint64_t v,
                                                  const WarpLane &w, bool live) {
    if (live && !w.idle && w.e < DIGEST_FELTS) {
        uint64_t *dst = reinterpret_cast<uint64_t *>(nodes + parent * 32);
        dst[w.e] = bswap64(v);
    }
}

}  // namespace rpx

// One inner level, two parents per warp (one per half). The node layout is
// `rpx_merkle_level`'s: children at `parent_begin + n_pairs + 2p` and `+ 1`,
// parent `p` at `parent_begin + p`.
extern "C" __global__ void rpx_merkle_level_warp(uint8_t *nodes, uint64_t parent_begin,
                                                 uint64_t n_pairs) {
    const rpx::WarpLane w = rpx::warp_lane();
    const uint64_t first = (((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 2;
    if (first >= n_pairs) return;  // warp-uniform
    const uint64_t pair = first + (uint64_t)(w.group >> 4);
    const bool live = pair < n_pairs;
    // A dead half (odd n_pairs) recomputes its neighbour's pair and stores nothing.
    const uint64_t p = live ? pair : first;
    uint64_t v = rpx::load_compress_lane(nodes, parent_begin + n_pairs + 2 * p, w.e);
    v = rpx::permute_warp_v<RPX_PERMUTE_VARIANT>(v, w);
    rpx::store_parent_lane(nodes, parent_begin + p, v, w, live);
}

// Every remaining level from `level_begin` up to the root, in ONE block, with a
// barrier between levels — `rpx_merkle_tail`'s job with a warp per two parents,
// so a level of up to `blockDim.x / 16` pairs is one permutation latency.
extern "C" __global__ void RPX_LAUNCH_BOUNDS(1024)
    rpx_merkle_tail_warp(uint8_t *nodes, uint64_t level_begin) {
    const rpx::WarpLane w = rpx::warp_lane();
    const uint64_t first_of_warp = (uint64_t)(threadIdx.x >> 5) * 2;
    const uint64_t pairs_per_pass = (uint64_t)(blockDim.x >> 5) * 2;
    const uint64_t half = (uint64_t)(w.group >> 4);
    uint64_t lb = level_begin;
    while (lb != 0) {
        const uint64_t nb = lb / 2;
        const uint64_t n_pairs = lb - nb;
        // Warp-uniform bounds: both halves run every pass together.
        for (uint64_t first = first_of_warp; first < n_pairs; first += pairs_per_pass) {
            const uint64_t pair = first + half;
            const bool live = pair < n_pairs;
            const uint64_t p = live ? pair : first;
            uint64_t v = rpx::load_compress_lane(nodes, nb + n_pairs + 2 * p, w.e);
            v = rpx::permute_warp_v<RPX_PERMUTE_VARIANT>(v, w);
            rpx::store_parent_lane(nodes, nb + p, v, w, live);
        }
        __syncthreads();
        lb = nb;
    }
}

// Parity probe for the half-warp permutation: state `i` is permuted by half
// `i mod 2` of warp `i / 2`; all twelve lanes are written back, raw.
extern "C" __global__ void rpx_permute_warp_probe(const uint64_t *states, uint64_t n,
                                                  uint64_t *out) {
    const rpx::WarpLane w = rpx::warp_lane();
    const uint64_t first = (((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5) * 2;
    if (first >= n) return;
    const uint64_t idx = first + (uint64_t)(w.group >> 4);
    const bool live = idx < n;
    const uint64_t p = live ? idx : first;
    uint64_t v = states[p * rpx::STATE_FELTS + w.e];
    v = rpx::permute_warp_v<RPX_PERMUTE_VARIANT>(v, w);
    if (live && !w.idle) out[p * rpx::STATE_FELTS + w.e] = v;
}

#endif  // __CUDACC__ || RPX_HOST_SIMT
