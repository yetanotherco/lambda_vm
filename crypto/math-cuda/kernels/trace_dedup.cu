// GPU dedup (hash group-by) for ALU operand tuples. Replaces the host
// HashMap<key, multiplicity> fold. The tables are order-independent (the LogUp
// bus is permutation-invariant), so we only need to GROUP equal keys and sum
// multiplicities — no sort required.
//
// Open-addressing hash table with linear probing. A 64-bit slot stores the
// index of the group's representative op (claimed via atomicCAS against the
// EMPTY sentinel); key equality is verified by reading the immutable input
// arrays at that index. Multiplicities accumulate per slot via atomicAdd.
// Sizing M = next_pow2(2*n) keeps load factor <= 0.5, so a free slot always
// exists (probe loop terminates) and probing stays short.
//
// Key is a triple (a, b, c) of u64 (c packs flags/op). Two multiplicity
// counters (mu0, mu1) support the dual-counter tables (MUL mu_lo/mu_hi, DVRM
// mu_q/mu_r); single-counter tables pass sel=NULL and only mu0 is used.

#include <cstdint>

#define EMPTY 0xFFFFFFFFFFFFFFFFULL

__device__ __forceinline__ uint64_t hash3(uint64_t a, uint64_t b, uint64_t c) {
  uint64_t h = a + 0x9E3779B97F4A7C15ULL;
  h ^= b;
  h *= 0xBF58476D1CE4E5B9ULL;
  h ^= c;
  h *= 0x94D049BB133111EBULL;
  h ^= h >> 31;
  return h;
}

extern "C" __global__ void dedup_init(uint64_t *slot, uint64_t *mu0,
                                      uint64_t *mu1, uint64_t M) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) {
    slot[i] = EMPTY;
    mu0[i] = 0;
    mu1[i] = 0;
  }
}

// Insert all n ops into the table. `sel[i]` selects the counter (0 -> mu0,
// 1 -> mu1); single-counter tables pass an all-zero `sel`.
extern "C" __global__ void
dedup_insert(const uint64_t *a, const uint64_t *b, const uint64_t *c,
             const uint64_t *sel, uint64_t n, uint64_t M, uint64_t *slot,
             uint64_t *mu0, uint64_t *mu1) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  uint64_t ka = a[i], kb = b[i], kc = c[i];
  uint64_t h = hash3(ka, kb, kc) & (M - 1);
  uint64_t *counter = (sel[i] != 0) ? mu1 : mu0;
  for (;;) {
    unsigned long long prev =
        atomicCAS((unsigned long long *)&slot[h], (unsigned long long)EMPTY,
                  (unsigned long long)i);
    if (prev == EMPTY) {
      atomicAdd((unsigned long long *)&counter[h], 1ULL);
      return;
    }
    if (a[prev] == ka && b[prev] == kb && c[prev] == kc) {
      atomicAdd((unsigned long long *)&counter[h], 1ULL);
      return;
    }
    h = (h + 1) & (M - 1); // linear probe
  }
}

// Gather occupied slots into a dense unique array (arbitrary order).
extern "C" __global__ void
dedup_compact(const uint64_t *slot, const uint64_t *mu0, const uint64_t *mu1,
              uint64_t M, const uint64_t *a, const uint64_t *b,
              const uint64_t *c, uint64_t *out_a, uint64_t *out_b,
              uint64_t *out_c, uint64_t *out_mu0, uint64_t *out_mu1,
              uint64_t *out_count) {
  uint64_t h = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= M)
    return;
  uint64_t rep = slot[h];
  if (rep == EMPTY)
    return;
  uint64_t pos = atomicAdd((unsigned long long *)out_count, 1ULL);
  out_a[pos] = a[rep];
  out_b[pos] = b[rep];
  out_c[pos] = c[rep];
  out_mu0[pos] = mu0[h];
  out_mu1[pos] = mu1[h];
}
