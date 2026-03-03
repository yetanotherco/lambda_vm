# SIMD Goldilocks Field Acceleration Design

**Goal:** Add AVX2, AVX-512, and NEON SIMD acceleration to Goldilocks field arithmetic, FFT, constraint evaluation, and Merkle tree hashing.

**Architecture:** Plonky3-style `PackedField` trait abstraction with compile-time platform dispatch. Algorithms become generic over packed types, automatically selecting scalar or SIMD paths.

**Scope:** Base field (Goldilocks, 64-bit) + cubic extension field (Fp3, w^3=2). Four integration targets: field ops, FFT, constraint evaluation, Merkle trees.

---

## 1. PackedField Trait & Packed Goldilocks Types

### 1.1 The `PackedField` Trait

New trait in `crypto/math/src/field/packed.rs`:

```rust
pub unsafe trait PackedField: Copy + Send + Sync + Sized
    + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Neg<Output = Self>
    + Add<Self::Scalar, Output = Self> + Mul<Self::Scalar, Output = Self>
{
    type Scalar: IsField;
    const WIDTH: usize;  // 1 (scalar), 2 (NEON), 4 (AVX2), 8 (AVX-512)

    fn from_fn(f: impl FnMut(usize) -> FieldElement<Self::Scalar>) -> Self;
    fn from_slice(slice: &[FieldElement<Self::Scalar>]) -> Self;
    fn as_slice(&self) -> &[FieldElement<Self::Scalar>];
    fn zero() -> Self;
    fn one() -> Self;

    /// Reinterpret a scalar slice as packed. Zero-cost (pointer cast).
    fn pack_slice(buf: &[FieldElement<Self::Scalar>]) -> &[Self];
    fn pack_slice_mut(buf: &mut [FieldElement<Self::Scalar>]) -> &mut [Self];

    /// Pack with scalar suffix for non-aligned tails.
    fn pack_slice_with_suffix(buf: &[FieldElement<Self::Scalar>])
        -> (&[Self], &[FieldElement<Self::Scalar>]);
    fn pack_slice_with_suffix_mut(buf: &mut [FieldElement<Self::Scalar>])
        -> (&mut [Self], &mut [FieldElement<Self::Scalar>]);

    /// Block interleave for in-register FFT transposes.
    fn interleave(&self, other: Self, block_len: usize) -> (Self, Self);
}
```

### 1.2 Adding `Packing` to `IsField`

The `IsField` trait gains an associated type:

```rust
pub trait IsField {
    type Packing: PackedField<Scalar = Self>;
    // ... existing methods unchanged
}
```

For `GoldilocksField`, resolved at compile time via `#[cfg]`:

| Platform | `Packing` type | WIDTH | Register |
|---|---|---|---|
| x86-64 + AVX2 (no AVX-512) | `PackedGoldilocksAVX2` | 4 | `__m256i` |
| x86-64 + AVX-512 | `PackedGoldilocksAVX512` | 8 | `__m512i` |
| AArch64 | `PackedGoldilocksNeon` | 2 | `uint64x2_t` |
| Fallback | `ScalarPacked<GoldilocksField>` | 1 | `u64` |

### 1.3 File Structure

New files under `crypto/math/src/field/fields/fft_friendly/`:

```
u64_goldilocks.rs                     (existing, unchanged)
u64_goldilocks_packed/
    mod.rs                             (trait + cfg dispatch)
    scalar.rs                          (WIDTH=1 fallback)
    x86_64_avx2.rs                     (WIDTH=4, __m256i)
    x86_64_avx512.rs                   (WIDTH=8, __m512i)
    aarch64_neon.rs                    (WIDTH=2, uint64x2_t)
```

### 1.4 AVX2 Arithmetic (WIDTH=4)

Each `PackedGoldilocksAVX2` is `#[repr(transparent)]` wrapping `[GoldilocksField; 4]`, transmutable to/from `__m256i`.

**Addition/Subtraction**: Shifted-representation trick — XOR with `2^63` converts unsigned overflow detection to signed comparison via `_mm256_cmpgt_epi64`. Correction via EPSILON (`2^32 - 1 = 2^64 mod P`):

```rust
// Add: t = a + b, if overflow add EPSILON (= subtract P mod 2^64)
let res_wrapped_s = _mm256_add_epi64(x, y_s);
let mask = _mm256_cmpgt_epi64(y_s, res_wrapped_s);  // overflow?
let correction = _mm256_srli_epi64::<32>(mask);       // 0 or EPSILON
_mm256_add_epi64(res_wrapped_s, correction)
```

**Multiplication**: Decompose 64-bit operands into 32-bit halves. Four `_mm256_mul_epu32` sub-products (each 32×32→64), bignum assembly into 128-bit result, then reduce:

```rust
// mul64_64: 4 sub-multiplications
let x_hi = _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(x)));
let y_hi = /* same for y */;
let mul_ll = _mm256_mul_epu32(x, y);       // x_lo * y_lo
let mul_lh = _mm256_mul_epu32(x, y_hi);    // x_lo * y_hi
let mul_hl = _mm256_mul_epu32(x_hi, y);    // x_hi * y_lo
let mul_hh = _mm256_mul_epu32(x_hi, y_hi); // x_hi * y_hi
// ... bignum combine into (hi, lo) 128-bit result
```

**reduce128**: Exploits `2^96 ≡ -1 (mod P)` and `2^64 ≡ EPSILON (mod P)`:

```rust
// For 128-bit value (hi, lo):
// result = lo - hi_hi + hi_lo * EPSILON
let hi_hi = _mm256_srli_epi64::<32>(hi);              // top 32 bits
let lo1 = sub(lo, hi_hi);                              // 2^96 = -1 mod P
let t1 = _mm256_mul_epu32(hi, EPSILON_VEC);            // hi_lo * EPSILON (naturally extracts lo32)
add(lo1, t1)
```

**Square**: 3 sub-products instead of 4 (exploit symmetry).

**movehdup trick**: `_mm256_movehdup_ps` runs on port 5, avoiding contention with `_mm256_mul_epu32` on ports 0/1.

### 1.5 AVX-512 Arithmetic (WIDTH=8)

Same algorithms as AVX2 but with `__m512i` registers. Key simplification: AVX-512 provides native unsigned 64-bit comparison (`_mm512_cmpge_epu64_mask` → `__mmask8`), eliminating the shifted-representation trick:

```rust
// Add: direct unsigned overflow detection
let res = _mm512_add_epi64(x, y);
let mask = _mm512_cmplt_epu64_mask(res, y);  // native unsigned compare
_mm512_mask_sub_epi64(res, mask, res, FIELD_ORDER)  // conditional correction
```

### 1.6 NEON Arithmetic (WIDTH=2)

Uses `uint64x2_t` (128-bit, 2 lanes). Addition/subtraction use shifted representation (like AVX2) with `vcgtq_s64`. Multiplication uses **inline assembly** exploiting AArch64's native `mul`/`umulh` (64×64→128-bit in two instructions), interleaving both lanes for ILP:

```asm
mul   lo0, a0, b0       // native 64x64→lo64
mul   lo1, a1, b1
umulh hi0, a0, b0       // native 64x64→hi64
umulh hi1, a1, b1
// ... reduce using same EPSILON math, interleaved for ILP
```

No 32-bit decomposition needed on AArch64.

---

## 2. Packed Cubic Extension Field (Fp3)

### 2.1 Representation

```rust
#[derive(Copy, Clone)]
pub struct PackedFp3<P: PackedField<Scalar = GoldilocksField>> {
    pub c0: P,  // coefficient of 1
    pub c1: P,  // coefficient of w
    pub c2: P,  // coefficient of w^2, where w^3 = 2
}
```

On AVX2: 4 independent Fp3 elements across 3 `__m256i` registers (12 u64 values).
On AVX-512: 8 independent Fp3 elements across 3 `__m512i` registers (24 u64 values).

### 2.2 Arithmetic

All ops delegate to packed base field:

- **Add/Sub**: Component-wise — 3 packed base ops each.
- **Mul**: Karatsuba-like using existing formula from `extensions_goldilocks.rs` (residue=2):
  - 6 packed base multiplies + ~9 packed base adds + `mul_by_2` (= `add(x, x)`)
- **Scalar × Extension (F×E→E)**: `(s * c0, s * c1, s * c2)` — 3 packed base muls. Critical path for constraint evaluation.
- **Square**: Optimized variant with fewer multiplies.

### 2.3 File Location

`crypto/math/src/field/fields/fft_friendly/u64_goldilocks_packed/fp3.rs`

---

## 3. FFT Butterfly SIMD

### 3.1 Current Architecture

Bowers FFT in `bowers_fft.rs` uses fused 2-layer butterflies. Each iteration of the inner loop (lines 815-848) processes 4 points with scalar operations: 4 twiddle multiplies + 8 add/sub.

### 3.2 SIMD Strategy

Pack WIDTH consecutive array elements into SIMD registers. Process WIDTH independent butterflies per iteration:

```rust
// Instead of:
let sum = &block[i0] + &block[i2];           // 1 scalar add

// Do:
let p0 = F::Packing::from_slice(&block[i0..]);
let p2 = F::Packing::from_slice(&block[i2..]);
let sum = p0 + p2;                            // WIDTH adds in parallel
```

**Applicability**: Innermost layers (where `quarter >= WIDTH`). Outermost layers fall back to scalar.

**Twiddle handling**: When all butterflies in a pack share the same twiddle (true for inner layers), broadcast twiddle to all lanes. Otherwise, load as packed vector.

### 3.3 Files Modified

- `crypto/math/src/fft/cpu/bowers_fft.rs` — add `_packed` variants of fused butterfly functions
- `crypto/math/src/fft/polynomial.rs` — dispatch to packed FFT when applicable

---

## 4. Constraint Evaluation SIMD

### 4.1 Current Architecture

The evaluator (`evaluator.rs`) iterates one LDE point at a time via Rayon `par_iter`. Per point: extract frame (~129 field elements), evaluate ~100 constraints, combine with zerofier.

### 4.2 SIMD Strategy

Process WIDTH LDE points simultaneously.

**Packed Frame Fill**: Extract WIDTH rows into packed columns:

```rust
// packed_frame[col] = PackedField::from_fn(|k| lde_trace.get_main(base_i + k*stride, col))
```

Produces `Vec<P>` (packed main columns) and `Vec<PackedFp3<P>>` (packed aux columns).

**Packed Constraint Evaluation**: Add parallel evaluation method to constraints:

```rust
fn evaluate_packed<P: PackedField<Scalar = F>>(
    &self,
    frame: &PackedFrame<P>,
    periodic: &[FieldElement<F>],
    rap_challenges: &[FieldElement<E>],
) -> PackedFp3<P>;
```

Default implementation: call scalar `evaluate` per lane (safe fallback). Optimized overrides inline the constraint math on packed types.

**Packed Accumulation**: Zerofier + random linear combination on packed values:

```rust
let packed_z = PackedFp3::from_fn(|k| zerofier[base_i + k*stride]);
let packed_sum = transition_buf.iter()
    .zip(coefficients)
    .fold(PackedFp3::zero(), |acc, (eval, beta)| acc + eval * beta);
packed_z * packed_sum  // WIDTH E-field multiplications in parallel
```

### 4.3 Files Modified

- `crypto/stark/src/frame.rs` — add `PackedFrame` struct
- `crypto/stark/src/constraints/evaluator.rs` — packed evaluation loop
- `crypto/stark/src/traits.rs` — add `evaluate_packed` method with default impl

---

## 5. Merkle Tree SIMD (Multi-Lane Keccak)

### 5.1 Current Architecture

Leaf hashing serializes field elements one-by-one into Keccak256. Internal nodes hash `left || right` (64 bytes).

### 5.2 SIMD Strategy

Keccak doesn't benefit from packing field elements. The win comes from **multi-lane Keccak**: running multiple independent Keccak-f[1600] permutations simultaneously.

| Platform | Lanes | Register per state word |
|---|---|---|
| AVX-512 | 8 | `__m512i` (25 state words) |
| AVX2 | 4 | `__m256i` (25 state words) |
| NEON | 2 | `uint64x2_t` (25 state words) |

**Leaf hashing**: Batch WIDTH leaves into one multi-lane call:

```rust
fn hash_leaves_packed<const LANES: usize>(
    leaves: &[&[u8]; LANES],
) -> [[u8; 32]; LANES]
```

**Internal node compression**: Batch WIDTH compressions in parallel.

**Keccak round SIMD highlights**:
- AVX-512: `_mm512_ternarylogic_epi64` fuses the χ step (AND-NOT + XOR) into one instruction; `_mm512_rol_epi64` for ρ rotations.
- AVX2: `_mm256_andnot_si256` + `_mm256_xor_si256` for χ; shift-and-or for rotations.
- NEON+SHA3: `vbcaxq_u64` for χ, `vxarq_u64` for XOR-and-rotate.

**Alternative**: Integrate the `keccak` crate (which has AVX2/AVX-512 implementations) or XKCP C library. This trades implementation effort for a dependency.

### 5.3 Files Modified/Created

- `crypto/crypto/src/hashing/keccak_simd/` — multi-lane Keccak implementation
- `crypto/crypto/src/merkle_tree/merkle.rs` — batched leaf hashing dispatch
- `crypto/crypto/src/merkle_tree/backends/` — SIMD-aware backend

---

## 6. Phasing

### Phase 1: PackedField Foundation
- `PackedField` trait + `ScalarPacked` fallback
- `PackedGoldilocksAVX2` (add, sub, mul, square, neg, interleave)
- `PackedGoldilocksAVX512`, `PackedGoldilocksNeon`
- `PackedFp3<P>` extension wrapper
- Exhaustive unit tests: arithmetic properties, cross-validation against scalar

### Phase 2: FFT
- Packed Bowers butterfly functions
- Dispatch logic in `polynomial.rs`
- Testing: FFT round-trip matches scalar, full `prove_elfs` tests pass

### Phase 3: Constraint Evaluation
- `PackedFrame`, packed evaluator loop
- `evaluate_packed` on constraints with default + optimized overrides
- Testing: valid proofs, benchmark comparison

### Phase 4: Merkle Trees
- Multi-lane Keccak (implement or integrate crate)
- Batched leaf hashing and compression
- Testing: Merkle roots match scalar, end-to-end prover tests

---

## 7. Expected Speedups (Conservative)

| Component | AVX2 (4-wide) | AVX-512 (8-wide) | NEON (2-wide) |
|---|---|---|---|
| Field multiply | ~2-3x | ~4-6x | ~1.5x |
| FFT | ~2-3x | ~4-6x | ~1.5x |
| Constraint eval | ~3-4x | ~5-7x | ~1.5-2x |
| Merkle (Keccak) | ~3-4x | ~6-8x | ~1.5-2x |

Goldilocks (64-bit) gets lower SIMD multipliers than 31-bit fields (BabyBear) because the 64×64-bit multiply emulation consumes 4 sub-products, using more ALU throughput.

---

## 8. Key Design Decisions

1. **Compile-time dispatch only** — no runtime CPUID detection. Target features set via `RUSTFLAGS=-C target-feature=+avx2` or `target-cpu=native`. Matches Plonky3's approach.

2. **`repr(transparent)` for zero-cost packing** — `pack_slice` is a pointer cast, not a copy. Packed types transmute directly to/from SIMD registers.

3. **Scalar fallback always works** — `ScalarPacked<F>` with WIDTH=1 means all generic code compiles and runs correctly without SIMD.

4. **No `to_extension()` in packed code** — banned pattern carries forward. F×E multiplication uses base-packed × extension-packed directly.

5. **Existing prover structure preserved** — no trait hierarchy changes to `IsField` beyond adding the `Packing` associated type. Existing scalar code paths remain as fallback.
