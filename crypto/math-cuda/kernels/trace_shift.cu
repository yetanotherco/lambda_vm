// GPU SHIFT-table trace generation. One thread per op (SHIFT does NOT dedup;
// μ=1 per row). Mirrors ShiftOperation::compute_aux + generate_shift_trace in
// prover/src/tables/shift.rs. Padding rows (r >= n) set ZBS=1 (all else zero),
// matching the CPU. Raw canonical u64 throughout (halves<2^16, words<2^32).

#include <cstdint>

// SHIFT column indices (must match prover/src/tables/shift.rs `cols`).
#define S_IN_0 0
#define S_SHIFT_AMOUNT 4
#define S_DIRECTION 5
#define S_SIGNED 6
#define S_WORD_INSTR 7
#define S_OUT_0 8
#define S_OUT_1 9
#define S_IS_NEGATIVE 10
#define S_BIT_SHIFT 11
#define S_ZBS 12
#define S_X_0 13
#define S_Y_0 18
#define S_LIMB_SHIFT_RAW_0 22
#define S_MU 25
#define S_SHIFT_B1 26
#define S_SHIFT_H1 27
#define S_SHIFT_HIGH 28

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}

__device__ __forceinline__ uint16_t hwsl(uint16_t h, uint8_t z) {
  return z == 0 ? h : (uint16_t)((uint32_t)h << z);
}
__device__ __forceinline__ uint16_t hwslc(uint16_t h, uint8_t z) {
  return z == 0 ? 0 : (uint16_t)(h >> (16 - (uint16_t)z));
}

extern "C" __global__ void
trace_shift_kernel(const uint64_t *__restrict__ value, // in_halves packed
                   const uint64_t *__restrict__ shift_amount,
                   const uint64_t *__restrict__ flags, // bit0=dir,bit1=signed,bit2=word
                   uint64_t n, uint64_t nrows,
                   uint64_t *__restrict__ cols) { // 29 * nrows, zeroed
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r >= n) {
    sc(cols, nrows, S_ZBS, r, 1); // padding rows: ZBS=1, rest zero
    return;
  }

  uint64_t val = value[r];
  uint64_t sa = shift_amount[r];
  uint64_t f = flags[r];
  int direction = f & 1;
  int is_signed = (f >> 1) & 1;
  int word = (f >> 2) & 1;
  uint8_t shift = (uint8_t)(sa & 0xFF);

  uint16_t in_h[4];
  in_h[0] = (uint16_t)(val & 0xFFFF);
  in_h[1] = (uint16_t)((val >> 16) & 0xFFFF);
  in_h[2] = (uint16_t)((val >> 32) & 0xFFFF);
  in_h[3] = (uint16_t)((val >> 48) & 0xFFFF);

  int left = !direction;
  int right = direction;

  int is_negative = is_signed && ((in_h[3] >> 15) & 1);
  uint16_t extension = is_negative ? 0xFFFF : 0;

  uint8_t bit_shift =
      left ? (uint8_t)(shift & 15)
           : (uint8_t)(((uint16_t)(256 - (uint16_t)shift)) & 15);
  int zbs = (bit_shift == 0);

  uint16_t x[5] = {0, 0, 0, 0, 0};
  uint16_t y[4] = {0, 0, 0, 0};
  if (zbs) {
    for (int i = 0; i < 4; i++) {
      if (left)
        x[i] = in_h[i];
      else
        y[i] = in_h[i];
    }
    x[4] = 0;
  } else {
    for (int i = 0; i < 4; i++) {
      x[i] = hwsl(in_h[i], bit_shift);
      y[i] = hwslc(in_h[i], bit_shift);
    }
    x[4] = hwsl(extension, bit_shift);
  }

  int limb_idx = word ? ((shift >> 4) & 1) : ((shift >> 4) & 3);
  int ls[4] = {0, 0, 0, 0};
  ls[limb_idx] = 1;

  // compute_shifted (DWordHL, 4 halfwords)
  uint16_t shifted[4];
  for (int i = 0; i < 4; i++) {
    uint16_t v = 0;
    if (left) {
      for (int j = 0; j <= i; j++) {
        if (ls[j]) {
          int k = i - j; // intra_left(k) = k==0 ? x[0] : x[k]+y[k-1]
          v = (uint16_t)(v + (k == 0 ? x[0] : (uint16_t)(x[k] + y[k - 1])));
        }
      }
    }
    if (right) {
      for (int j = 0; j <= 3 - i; j++) {
        if (ls[j]) {
          int k = i + j; // intra_right(k) = y[k]+x[k+1]
          v = (uint16_t)(v + (uint16_t)(y[k] + x[k + 1]));
        }
      }
      for (int j = 4 - i; j < 4; j++) {
        if (ls[j])
          v = (uint16_t)(v + extension);
      }
    }
    shifted[i] = v;
  }

  uint32_t out_0 = (uint32_t)shifted[0] | ((uint32_t)shifted[1] << 16);
  uint32_t out_1 = (uint32_t)shifted[2] | ((uint32_t)shifted[3] << 16);

  // ---- column writes ----
  for (int i = 0; i < 4; i++)
    sc(cols, nrows, S_IN_0 + i, r, in_h[i]);
  sc(cols, nrows, S_SHIFT_AMOUNT, r, shift);
  sc(cols, nrows, S_SHIFT_B1, r, (sa >> 8) & 0xFF);
  sc(cols, nrows, S_SHIFT_H1, r, (sa >> 16) & 0xFFFF);
  sc(cols, nrows, S_SHIFT_HIGH, r, (sa >> 32) & 0xFFFFFFFF);
  sc(cols, nrows, S_DIRECTION, r, (uint64_t)direction);
  sc(cols, nrows, S_SIGNED, r, (uint64_t)is_signed);
  sc(cols, nrows, S_WORD_INSTR, r, (uint64_t)word);
  sc(cols, nrows, S_OUT_0, r, out_0);
  sc(cols, nrows, S_OUT_1, r, out_1);
  sc(cols, nrows, S_IS_NEGATIVE, r, (uint64_t)is_negative);
  sc(cols, nrows, S_BIT_SHIFT, r, bit_shift);
  sc(cols, nrows, S_ZBS, r, (uint64_t)zbs);
  for (int i = 0; i < 5; i++)
    sc(cols, nrows, S_X_0 + i, r, x[i]);
  for (int i = 0; i < 4; i++)
    sc(cols, nrows, S_Y_0 + i, r, y[i]);
  for (int i = 0; i < 3; i++) // limb_shift[3] is virtual (not stored)
    sc(cols, nrows, S_LIMB_SHIFT_RAW_0 + i, r, (uint64_t)ls[i]);
  sc(cols, nrows, S_MU, r, 1);
}
