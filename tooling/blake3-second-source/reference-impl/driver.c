/* Test driver for the round-parameterised upstream BLAKE3 C reference.
 *
 * This file is Lambda VM's; everything it links against in `upstream/` is
 * upstream BLAKE3 (CC0 / Apache-2.0), unmodified except for the round-count
 * loop in blake3_portable_paramrounds.c (see PARAMETERISATION.diff).
 *
 * Modes:
 *   hash    <input_len> <out_len>             default hashing mode
 *   hashhex <msg_hex> <out_len>              default hashing mode, explicit msg
 *   keyed   <key_hex64> <input_len> <out_len>  keyed_hash mode
 *   derive  <context_str> <input_len> <out_len>  derive_key mode
 *   compress                                  raw compression from stdin:
 *       one whitespace-separated record per line --
 *         h[0..8] (8 hex u32) m[0..16] (16 hex u32) t (hex u64)
 *         block_len (dec) flags (dec)
 *       prints the 16-word output as 16 concatenated 8-hex-digit words.
 *
 * The `compress` mode calls blake3_compress_xof_portable directly, which is
 * the 16-word (64-byte) output of the compression function `f` -- exactly the
 * object CANONICAL_VECTORS pins.
 *
 * Input bytes for the hashing modes follow the official test-vector pattern:
 * byte i is (i % 251).
 */
#include "upstream/blake3.h"
#include "upstream/blake3_impl.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_pattern(uint8_t *buf, size_t len) {
  for (size_t i = 0; i < len; i++) {
    buf[i] = (uint8_t)(i % 251);
  }
}

static void print_hex(const uint8_t *b, size_t n) {
  for (size_t i = 0; i < n; i++) {
    printf("%02x", b[i]);
  }
  printf("\n");
}

static int hex_to_bytes(const char *hex, uint8_t *out, size_t out_len) {
  if (strlen(hex) != out_len * 2) {
    return -1;
  }
  for (size_t i = 0; i < out_len; i++) {
    unsigned v;
    if (sscanf(hex + 2 * i, "%2x", &v) != 1) {
      return -1;
    }
    out[i] = (uint8_t)v;
  }
  return 0;
}

static int run_hasher(blake3_hasher *h, size_t input_len, size_t out_len) {
  uint8_t *input = malloc(input_len ? input_len : 1);
  uint8_t *out = malloc(out_len ? out_len : 1);
  if (!input || !out) {
    return 1;
  }
  fill_pattern(input, input_len);
  blake3_hasher_update(h, input, input_len);
  blake3_hasher_finalize(h, out, out_len);
  print_hex(out, out_len);
  free(input);
  free(out);
  return 0;
}

static int mode_compress(void) {
  uint32_t h[8], m[16];
  unsigned long long t;
  unsigned block_len, flags;
  char line[4096];

  while (fgets(line, sizeof(line), stdin)) {
    char *p = line;
    int consumed;
    int ok = 1;

    for (int i = 0; i < 8 && ok; i++) {
      if (sscanf(p, " %x%n", &h[i], &consumed) != 1) {
        ok = 0;
      }
      p += consumed;
    }
    for (int i = 0; i < 16 && ok; i++) {
      if (sscanf(p, " %x%n", &m[i], &consumed) != 1) {
        ok = 0;
      }
      p += consumed;
    }
    if (ok && sscanf(p, " %llx %u %u", &t, &block_len, &flags) != 3) {
      ok = 0;
    }
    if (!ok) {
      continue; /* blank or malformed line */
    }

    /* The compression function takes the message block as 64 little-endian
     * bytes; serialise m[] the way BLAKE3 loads it (load32 is LE). */
    uint8_t block[BLAKE3_BLOCK_LEN];
    for (int i = 0; i < 16; i++) {
      store32(&block[i * 4], m[i]);
    }

    uint8_t out64[64];
    blake3_compress_xof_portable(h, block, (uint8_t)block_len,
                                 (uint64_t)t, (uint8_t)flags, out64);

    for (int i = 0; i < 16; i++) {
      printf("%08x", load32(&out64[i * 4]));
    }
    printf("\n");
    fflush(stdout);
  }
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s hash|keyed|derive|compress ...\n", argv[0]);
    return 2;
  }

  if (strcmp(argv[1], "compress") == 0) {
    return mode_compress();
  }

  if (strcmp(argv[1], "hash") == 0 && argc == 4) {
    blake3_hasher h;
    blake3_hasher_init(&h);
    return run_hasher(&h, strtoul(argv[2], NULL, 10), strtoul(argv[3], NULL, 10));
  }

  /* Whole-message hashing of an explicit byte string. This is what turns the
   * socket spec's "compress(a,b) == BLAKE3(a || b || tag) truncated" claim into
   * something executable against upstream code rather than against a compress
   * call this repo assembled itself. */
  if (strcmp(argv[1], "hashhex") == 0 && argc == 4) {
    size_t msg_len = strlen(argv[2]) / 2;
    if (strlen(argv[2]) % 2 != 0) {
      fprintf(stderr, "message hex must have even length\n");
      return 2;
    }
    uint8_t *msg = malloc(msg_len ? msg_len : 1);
    if (!msg || hex_to_bytes(argv[2], msg, msg_len) != 0) {
      fprintf(stderr, "bad message hex\n");
      return 2;
    }
    size_t out_len = strtoul(argv[3], NULL, 10);
    uint8_t *out = malloc(out_len ? out_len : 1);
    if (!out) {
      return 1;
    }
    blake3_hasher h;
    blake3_hasher_init(&h);
    blake3_hasher_update(&h, msg, msg_len);
    blake3_hasher_finalize(&h, out, out_len);
    print_hex(out, out_len);
    free(msg);
    free(out);
    return 0;
  }

  if (strcmp(argv[1], "keyed") == 0 && argc == 5) {
    uint8_t key[BLAKE3_KEY_LEN];
    if (hex_to_bytes(argv[2], key, BLAKE3_KEY_LEN) != 0) {
      fprintf(stderr, "bad key hex (need %d bytes)\n", BLAKE3_KEY_LEN);
      return 2;
    }
    blake3_hasher h;
    blake3_hasher_init_keyed(&h, key);
    return run_hasher(&h, strtoul(argv[3], NULL, 10), strtoul(argv[4], NULL, 10));
  }

  if (strcmp(argv[1], "derive") == 0 && argc == 5) {
    blake3_hasher h;
    blake3_hasher_init_derive_key(&h, argv[2]);
    return run_hasher(&h, strtoul(argv[3], NULL, 10), strtoul(argv[4], NULL, 10));
  }

  fprintf(stderr, "bad arguments\n");
  return 2;
}
