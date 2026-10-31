//! The BLAKE3 compression function the LFM chips are built from.
//!
//! **This module is a re-export.** The implementation lives at
//! [`crypto::hash::blake3`], which is the only crate the three callers that must
//! not disagree can all reach: the Merkle commitment backends (in `crypto`), the
//! `LFM_BLAKE3` chip and the `LFM_HASH` socket (here), and the CUDA kernels'
//! parity reference (in `math-cuda`, which `prover` depends on). A chip and a
//! commitment that hash identically because they call one function is a
//! different claim from two that agree today.
//!
//! Everything the chips used before is still reachable under this path and means
//! the same thing: [`blake3_compress_rounds`], [`blake3_compress_6round`],
//! [`BLAKE3_IV`], [`BLAKE3_MSG_PERMUTATION`], [`BLAKE3_ROUNDS`],
//! [`CANONICAL_VECTORS`] and [`canonical_expected_out`].
//!
//! The round count travels with it: this crate's `blake3-6round` feature now
//! forwards to `crypto`'s, so [`BLAKE3_ROUNDS`] — and therefore
//! `blake3_socket::SOCKET_ROUNDS`, which is an alias of it — is one symbol for
//! the whole tree. Enabling `crypto/blake3-6round` alone would leave the chip at
//! 6 rounds and is caught by the `SOCKET_ROUNDS == BLAKE3_ROUNDS` assertion,
//! which is why that assertion stays.
//!
//! # Why the tests stayed here
//!
//! The falsification suite below — the negative controls that break one
//! convention at a time, and the `blake3` crate anchor — tests the primitive
//! *through this path*, which is the path the chips use. Keeping it here means
//! the re-export itself is covered: a shim that resolved to the wrong thing
//! would fail these, and moving them down would have made the chips' view of the
//! primitive untested. `crypto` has its own tests for the construction layer
//! ([`crypto::hash::blake3::chain`]) that this module does not use.

pub use crypto::hash::blake3::*;

#[cfg(test)]
mod tests {
    use super::*;

    /// The conventions a wrong port could get wrong, as data.
    ///
    /// [`CANONICAL_VECTORS`] is supposed to pin every one of these. Naming them
    /// in a struct is what lets the negative control break exactly one at a time.
    #[derive(Clone, Copy)]
    struct Conventions {
        /// The four rotation amounts of `G`, in application order.
        rot: [u32; 4],
        /// The message-schedule permutation applied between rounds.
        perm: [usize; 16],
        rounds: usize,
    }

    /// The conventions [`CANONICAL_VECTORS`] were generated under. `rounds` is
    /// [`BLAKE3_SIX_ROUNDS`], not [`BLAKE3_ROUNDS`]: that table pins the 6-round
    /// variant whatever the build is compiled for, and reading the knob here
    /// would make the "7 rounds" control below silently stop discriminating at
    /// the default.
    const CANONICAL: Conventions = Conventions {
        rot: [16, 12, 8, 7],
        perm: BLAKE3_MSG_PERMUTATION,
        rounds: BLAKE3_SIX_ROUNDS,
    };

    /// A deliberately *parameterised* compression, used only to build negative
    /// controls: the same dataflow with [`Conventions`] as an input.
    ///
    /// It is NOT what [`blake3_compress_6round`] calls. Keeping the two apart
    /// costs a duplicated loop and buys the thing rule 7 is about: the control
    /// tests below compare this function's output against [`CANONICAL_VECTORS`]
    /// — a constant that came from outside this file — so they stay meaningful
    /// no matter how the real function is later refactored.
    fn compress_variant(v: &Vector, c: Conventions) -> [u32; 16] {
        let g = |s: &mut [u32; 16], a: usize, b: usize, cc: usize, d: usize, mx: u32, my: u32| {
            s[a] = s[a].wrapping_add(s[b]).wrapping_add(mx);
            s[d] = (s[d] ^ s[a]).rotate_right(c.rot[0]);
            s[cc] = s[cc].wrapping_add(s[d]);
            s[b] = (s[b] ^ s[cc]).rotate_right(c.rot[1]);
            s[a] = s[a].wrapping_add(s[b]).wrapping_add(my);
            s[d] = (s[d] ^ s[a]).rotate_right(c.rot[2]);
            s[cc] = s[cc].wrapping_add(s[d]);
            s[b] = (s[b] ^ s[cc]).rotate_right(c.rot[3]);
        };
        let h = v.h;
        let mut s: [u32; 16] = [
            h[0],
            h[1],
            h[2],
            h[3],
            h[4],
            h[5],
            h[6],
            h[7],
            BLAKE3_IV[0],
            BLAKE3_IV[1],
            BLAKE3_IV[2],
            BLAKE3_IV[3],
            v.t as u32,
            (v.t >> 32) as u32,
            v.block_len,
            v.flags,
        ];
        let mut m = v.m;
        for r in 0..c.rounds {
            g(&mut s, 0, 4, 8, 12, m[0], m[1]);
            g(&mut s, 1, 5, 9, 13, m[2], m[3]);
            g(&mut s, 2, 6, 10, 14, m[4], m[5]);
            g(&mut s, 3, 7, 11, 15, m[6], m[7]);
            g(&mut s, 0, 5, 10, 15, m[8], m[9]);
            g(&mut s, 1, 6, 11, 12, m[10], m[11]);
            g(&mut s, 2, 7, 8, 13, m[12], m[13]);
            g(&mut s, 3, 4, 9, 14, m[14], m[15]);
            if r < c.rounds - 1 {
                let prev = m;
                for (i, &p) in c.perm.iter().enumerate() {
                    m[i] = prev[p];
                }
            }
        }
        let mut out = [0u32; 16];
        for i in 0..8 {
            out[i] = s[i] ^ s[i + 8];
            out[i + 8] = s[i + 8] ^ h[i];
        }
        out
    }

    /// The port reproduces all ten canonical vectors.
    #[test]
    fn the_compression_matches_the_canonical_six_round_vectors() {
        for (i, v) in CANONICAL_VECTORS.iter().enumerate() {
            assert_eq!(
                blake3_compress_6round(&v.h, &v.m, v.t, v.block_len, v.flags),
                v.out,
                "canonical 6-round vector {i}"
            );
        }
    }

    /// The parameterised control, at canonical parameters, IS the port — so a
    /// negative control below differs from the real thing in exactly the one
    /// convention it names, and nothing else.
    #[test]
    fn the_variant_at_canonical_parameters_is_the_port() {
        for v in CANONICAL_VECTORS.iter() {
            assert_eq!(
                compress_variant(v, CANONICAL),
                v.out,
                "the control must reproduce the vectors at canonical parameters"
            );
        }
    }

    /// NEGATIVE CONTROL (rule 9): each convention the vectors are supposed to
    /// pin, broken one at a time, must stop reproducing them.
    ///
    /// Without this, "the vectors pass" would be evidence only that the vectors
    /// are *reachable*, not that they discriminate. Each case names what would
    /// silently be unpinned if it ever started passing.
    #[test]
    fn breaking_one_convention_at_a_time_breaks_the_vectors() {
        // The message permutation transposed (its own inverse composition):
        // same multiset of indices, same round count, different schedule.
        let mut transposed = [0usize; 16];
        for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
            transposed[p] = i;
        }
        let cases: [(&str, Conventions); 4] = [
            // rotr12 -> rotr13: the one rotation amount that is NOT a byte
            // relabel in the chip, so a wrong value here is the wrong-rotation
            // bug in its most consequential place.
            (
                "rotr12 -> rotr13",
                Conventions {
                    rot: [16, 13, 8, 7],
                    ..CANONICAL
                },
            ),
            // rotr16 and rotr8 swapped: both ARE free byte relabels in the
            // chip, so transposing them costs no columns and no constraints —
            // the cheapest possible way to be wrong.
            (
                "rotr16 <-> rotr8",
                Conventions {
                    rot: [8, 12, 16, 7],
                    ..CANONICAL
                },
            ),
            (
                "message schedule transposed",
                Conventions {
                    perm: transposed,
                    ..CANONICAL
                },
            ),
            (
                "7 rounds (standard BLAKE3)",
                Conventions {
                    rounds: 7,
                    ..CANONICAL
                },
            ),
        ];
        for (what, c) in cases {
            let v = &CANONICAL_VECTORS[0];
            assert_ne!(
                compress_variant(v, c),
                v.out,
                "{what} still reproduces the canonical vector — the vector does not pin it"
            );
        }
    }

    /// The port reproduces all ten canonical inputs at **7 rounds** too.
    ///
    /// [`CANONICAL_OUT_7ROUND`] came from two independently-written Python
    /// references that agree on all ten, and whose 7-round paths are pinned by
    /// the official BLAKE3 vectors. So this is the same shape of check as the
    /// 6-round one above but with a stronger source, and together they are what
    /// let `BLAKE3_ROUNDS` be flipped without the chip losing its vector pin.
    #[test]
    fn the_compression_matches_the_canonical_vectors_at_seven_rounds() {
        for (i, v) in CANONICAL_VECTORS.iter().enumerate() {
            assert_eq!(
                blake3_compress_rounds(
                    &v.h,
                    &v.m,
                    v.t,
                    v.block_len,
                    v.flags,
                    BLAKE3_STANDARD_ROUNDS
                ),
                CANONICAL_OUT_7ROUND[i],
                "7-round canonical vector {i}"
            );
        }
    }

    /// NEGATIVE CONTROL: the two tables really are different data. Without this,
    /// a generation bug that emitted the 6-round outputs twice would leave the
    /// test above passing and pinning nothing new.
    #[test]
    fn the_six_and_seven_round_vector_tables_differ_everywhere() {
        for (i, v) in CANONICAL_VECTORS.iter().enumerate() {
            assert_ne!(v.out, CANONICAL_OUT_7ROUND[i], "vector {i}");
        }
    }

    /// `canonical_expected_out` selects the table matching the compiled knob.
    /// This is the accessor `blake3_probe` asserts the chip's `OUT` columns
    /// against, so a wrong branch here would silently unpin the chip.
    #[test]
    fn canonical_expected_out_follows_the_round_knob() {
        for (i, v) in CANONICAL_VECTORS.iter().enumerate() {
            let want = if BLAKE3_ROUNDS == BLAKE3_STANDARD_ROUNDS {
                CANONICAL_OUT_7ROUND[i]
            } else {
                v.out
            };
            assert_eq!(canonical_expected_out(i), want, "vector {i}");
            assert_eq!(
                canonical_expected_out(i),
                blake3_compress_rounds(&v.h, &v.m, v.t, v.block_len, v.flags, BLAKE3_ROUNDS),
                "the accessor must agree with the primitive at the compiled round count"
            );
        }
    }

    /// ★ **The external anchor, direct.** At 7 rounds this module's compression
    /// function IS standard BLAKE3, checked against the `blake3` crate with no
    /// oracle, no JSON and no transcription in between.
    ///
    /// PLAN §2.2 step 4 asked for exactly this and Phase 1 deferred it for want
    /// of a cargo dependency; this is that check, discharged. A message of at
    /// most 64 bytes is one chunk and one block, so the whole tree hasher
    /// collapses to a single `f` invocation: `h = IV`, the block zero-padded to
    /// 64 bytes and read as 16 little-endian words, `t = 0`, `block_len` the
    /// true length, `flags = CHUNK_START|CHUNK_END|ROOT`. The 32-byte digest is
    /// `out[0..8]` in little-endian order.
    ///
    /// It runs over 65 lengths (0..=64) rather than one, because the length is
    /// what `block_len` and the padding both key off, and a port that ignored
    /// `block_len` would still pass at a single length.
    #[test]
    fn seven_rounds_is_the_blake3_crate() {
        const CHUNK_START: u32 = 1;
        const CHUNK_END: u32 = 2;
        const ROOT: u32 = 8;

        for len in 0..=64usize {
            let msg: Vec<u8> = (0..len)
                .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
                .collect();
            let mut block = [0u8; 64];
            block[..len].copy_from_slice(&msg);
            let words: [u32; 16] = core::array::from_fn(|i| {
                u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
            });

            let out = blake3_compress_rounds(
                &BLAKE3_IV,
                &words,
                0,
                len as u32,
                CHUNK_START | CHUNK_END | ROOT,
                BLAKE3_STANDARD_ROUNDS,
            );
            let mut ours = [0u8; 32];
            for i in 0..8 {
                ours[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
            }

            assert_eq!(
                ours,
                *blake3::hash(&msg).as_bytes(),
                "7-round compression must equal the blake3 crate at length {len}"
            );
        }
    }

    /// NEGATIVE CONTROL for the anchor above: at 6 rounds it must NOT match.
    ///
    /// Without this, `seven_rounds_is_the_blake3_crate` would pass just as well
    /// if `rounds` were being ignored — which is the one bug that would make the
    /// whole external-anchor argument vacuous, since the 6-round variant's only
    /// defence is "the same code path with the loop bound changed".
    #[test]
    fn six_rounds_is_not_the_blake3_crate() {
        let msg: [u8; 36] = core::array::from_fn(|i| i as u8);
        let mut block = [0u8; 64];
        block[..36].copy_from_slice(&msg);
        let words: [u32; 16] = core::array::from_fn(|i| {
            u32::from_le_bytes(block[4 * i..4 * i + 4].try_into().unwrap())
        });
        // BLAKE3_SIX_ROUNDS, not BLAKE3_ROUNDS: the knob defaults to 7, and
        // reading it here would turn this control into a copy of the anchor.
        let out = blake3_compress_rounds(&BLAKE3_IV, &words, 0, 36, 1 | 2 | 8, BLAKE3_SIX_ROUNDS);
        let mut ours = [0u8; 32];
        for i in 0..8 {
            ours[4 * i..4 * i + 4].copy_from_slice(&out[i].to_le_bytes());
        }
        assert_ne!(ours, *blake3::hash(&msg).as_bytes());
    }

    /// The counter split is load-bearing and full-width: `t` reaches the state
    /// as two 32-bit halves in low-then-high order, so swapping them must move
    /// the output. Six of the ten canonical vectors have distinct halves.
    #[test]
    fn the_counter_halves_are_not_interchangeable() {
        let mut checked = 0;
        for v in CANONICAL_VECTORS.iter() {
            let swapped = v.t.rotate_left(32);
            if swapped == v.t {
                continue;
            }
            checked += 1;
            assert_ne!(
                blake3_compress_6round(&v.h, &v.m, swapped, v.block_len, v.flags),
                v.out,
                "swapping the counter halves must change the output"
            );
        }
        assert!(
            checked >= 8,
            "expected most vectors to have distinct halves, got {checked}"
        );
    }
}
