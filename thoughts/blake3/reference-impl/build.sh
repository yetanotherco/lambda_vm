#!/bin/sh
# Build the two round-parameterised reference binaries.
#
# Only `blake3_portable_paramrounds.c` differs from upstream (see
# PARAMETERISATION.diff); blake3.c / blake3_dispatch.c / blake3_impl.h /
# blake3.h under upstream/ are verbatim BLAKE3 1.8.5.
#
# NEON is disabled and no x86 SIMD is available, so the dispatcher resolves
# every compression to the portable path -- i.e. to the parameterised file.
# That is what makes the round knob apply to the WHOLE tree hasher and not
# only to a directly-called compress.
#
# This is a ~1 second single-file C compile. It is not a cargo build.
set -e
cd "$(dirname "$0")"

SRC="driver.c blake3_portable_paramrounds.c upstream/blake3.c upstream/blake3_dispatch.c"
COMMON="-O2 -Wall -Iupstream -DBLAKE3_USE_NEON=0 -DBLAKE3_NO_SSE2 -DBLAKE3_NO_SSE41 -DBLAKE3_NO_AVX2 -DBLAKE3_NO_AVX512"

cc $COMMON -DBLAKE3_ROUNDS_PARAM=7 -o b3ref7 $SRC
cc $COMMON -DBLAKE3_ROUNDS_PARAM=6 -o b3ref6 $SRC

echo "built: b3ref7 (standard BLAKE3) and b3ref6 (internal variant)"
