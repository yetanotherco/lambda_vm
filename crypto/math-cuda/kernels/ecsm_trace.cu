// ECSM core chip main-column generation. One thread per row, 427 columns.
// Kernel is a pure column-layout splay: CPU has already formatted the
// byte/halfword/carry data (including padding-row q1 = P_BYTES) into
// flat blobs. Kernel splits the four DWordWL scalar fields and copies
// the blobs into their column slices.
//
// Per-row inputs:
//   timestamps[row]            = u64
//   addr_xgs[row]              = u64
//   addr_ks[row]               = u64
//   addr_xrs[row]              = u64
//   flags_len_k[row] bits:
//     bits 0..7   len_k (Byte)
//     bit  8      active (mu = 1; 0 = padding row)
//   byte_blob[row*257 + i]     = u64 byte-cell, layout (per-row offset):
//     [0..32)    x_r            (32 bytes, → XR..XR+31)
//     [32..64)   y_r            (32 bytes, → YR..)
//     [64..96)   k              (32 bytes, → K..)
//     [96..128)  x_g            (32 bytes, → XG..)
//     [128..160) y_g            (32 bytes, → YG..)
//     [160..192) x2             (32 bytes, → X2..)
//     [192..224) q0             (32 bytes, → Q0..)
//     [224..257) q1             (33 bytes, → Q1..; padding rows hold P_BYTES)
//   hw_blob[row*32 + i]        = u64 halfword:
//     [0..16)    k_sub_n        (16 halfwords)
//     [16..32)   xr_sub_p       (16 halfwords)
//   c_blob[row*128 + i]        = u64 carry (already CPU-converted from
//                                signed i64 to Goldilocks field rep):
//     [0..64)    c0             (64 entries)
//     [64..128)  c1             (64 entries)
//
// Column layout (matches `prover/src/tables/ecsm.rs::cols`):
//   0..1     TIMESTAMP_0/_1
//   2..3     ADDR_XG_0/_1
//   4..5     ADDR_K_0/_1
//   6..7     ADDR_XR_0/_1
//   8..39    XR        (32)
//   40..71   YR        (32)
//   72..103  K         (32)
//   104      LEN_K
//   105..136 XG        (32)
//   137..168 YG        (32)
//   169..200 X2        (32)
//   201..232 Q0        (32)
//   233..296 C0        (64)
//   297..329 Q1        (33)
//   330..393 C1        (64)
//   394..409 K_SUB_N   (16)
//   410..425 XR_SUB_P  (16)
//   426      MU

#include <cuda_runtime.h>
#include <stdint.h>

#define BLOCK_SIZE 64

extern "C" __global__ void generate_ecsm_trace_rows(
    uint64_t num_rows,
    const uint64_t *timestamps,
    const uint64_t *addr_xgs,
    const uint64_t *addr_ks,
    const uint64_t *addr_xrs,
    const uint64_t *flags_len_k,
    const uint64_t *byte_blob,   // num_rows * 257
    const uint64_t *hw_blob,     // num_rows * 32
    const uint64_t *c_blob,      // num_rows * 128
    uint64_t *table_data,
    uint64_t num_cols            // expected = 427
) {
    uint64_t row = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (row >= num_rows) return;
    uint64_t base = row * num_cols;

    uint64_t ts = timestamps[row];
    uint64_t a_xg = addr_xgs[row];
    uint64_t a_k = addr_ks[row];
    uint64_t a_xr = addr_xrs[row];
    uint64_t fk = flags_len_k[row];

    // Four DWordWL fields (each split into lo32/hi32).
    table_data[base + 0] = ts & 0xFFFFFFFFULL;
    table_data[base + 1] = ts >> 32;
    table_data[base + 2] = a_xg & 0xFFFFFFFFULL;
    table_data[base + 3] = a_xg >> 32;
    table_data[base + 4] = a_k & 0xFFFFFFFFULL;
    table_data[base + 5] = a_k >> 32;
    table_data[base + 6] = a_xr & 0xFFFFFFFFULL;
    table_data[base + 7] = a_xr >> 32;

    const uint64_t *bb = byte_blob + row * 257ULL;
    const uint64_t *hw = hw_blob + row * 32ULL;
    const uint64_t *cb = c_blob + row * 128ULL;

    // XR (col 8..39)  = bb[0..32]
    #pragma unroll
    for (int i = 0; i < 32; ++i) table_data[base + 8 + i] = bb[i];
    // YR (col 40..71) = bb[32..64]
    #pragma unroll
    for (int i = 0; i < 32; ++i) table_data[base + 40 + i] = bb[32 + i];
    // K  (col 72..103) = bb[64..96]
    #pragma unroll
    for (int i = 0; i < 32; ++i) table_data[base + 72 + i] = bb[64 + i];

    // LEN_K
    table_data[base + 104] = fk & 0xFFULL;

    // XG (col 105..136) = bb[96..128]
    #pragma unroll
    for (int i = 0; i < 32; ++i) table_data[base + 105 + i] = bb[96 + i];
    // YG (col 137..168) = bb[128..160]
    #pragma unroll
    for (int i = 0; i < 32; ++i) table_data[base + 137 + i] = bb[128 + i];
    // X2 (col 169..200) = bb[160..192]
    #pragma unroll
    for (int i = 0; i < 32; ++i) table_data[base + 169 + i] = bb[160 + i];
    // Q0 (col 201..232) = bb[192..224]
    #pragma unroll
    for (int i = 0; i < 32; ++i) table_data[base + 201 + i] = bb[192 + i];

    // C0 (col 233..296) = cb[0..64]
    #pragma unroll
    for (int i = 0; i < 64; ++i) table_data[base + 233 + i] = cb[i];

    // Q1 (col 297..329) = bb[224..257]  (33 bytes — padding holds P_BYTES)
    #pragma unroll
    for (int i = 0; i < 33; ++i) table_data[base + 297 + i] = bb[224 + i];

    // C1 (col 330..393) = cb[64..128]
    #pragma unroll
    for (int i = 0; i < 64; ++i) table_data[base + 330 + i] = cb[64 + i];

    // K_SUB_N (col 394..409) = hw[0..16]
    #pragma unroll
    for (int i = 0; i < 16; ++i) table_data[base + 394 + i] = hw[i];
    // XR_SUB_P (col 410..425) = hw[16..32]
    #pragma unroll
    for (int i = 0; i < 16; ++i) table_data[base + 410 + i] = hw[16 + i];

    // MU
    table_data[base + 426] = (fk >> 8) & 1ULL;
}
