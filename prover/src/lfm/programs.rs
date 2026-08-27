//! Registered LFM programs.
//!
//! Every program here is deterministic — same builder calls, same
//! instructions, same column groups, same digest — which is what lets the
//! registry pin it and the drift tests recompute it on every PR. Arena
//! *values* vary per proof; the program (and its identity) never does.
//!
//! # The wrap hash is named at every constructor that has one
//!
//! `WrapHash::default()` is `Keccak` and stays that way: it is the *unset*
//! value, not the production one. Letting production be inherited from a
//! `derive` three files away is the shape that made the P-a flip a twenty-site
//! audit (PA-PLAN §6.0, finding rev-emit E3), so every constructor here that
//! emits a hash says which one, and the exceptions read as exceptions rather
//! than as omissions:
//!
//! - [`keccak_chain_program_source`], [`keccak_sponge_program_source`] and
//!   [`keccak_sample_program_source`] are instruments ABOUT keccak (R1b/R1c/R1d).
//!   A BLAKE3 twin would be a new program and a new registry row, never a
//!   re-blessing of one of these.
//! - ★ [`program_id_program_source`] mirrors `recursion::program_id_from_digest`,
//!   which names `PlatformKeccak256`. It is the attestation join: it must
//!   compute what the host computes, and the host's program-identity digest is
//!   keccak whatever the commitment hash is. Staying keccak is the binding, not
//!   an oversight to be tidied later.
//!
//! Constructors that emit no hash name nothing, because a hash they never
//! compute is not a property they have: `trivial`, `fri_toy`, `lde_probe`,
//! `l2g_binding` (word equality only), and `permute_coverage` (which drives the
//! `LFM_HASH` socket — a different axis entirely).

use crate::tables::types::{FE, FEE};

use super::builder::{Cell, LfmBuilder, LfmProgramSource};
use super::compiler::{LfmProgram, compile};
use super::edsl::WrapHash;

/// The Milestone-B trivial program: a few hundred instructions exercising
/// every chip — constants, base ALU (incl. the assert lowering), Fp3 ALU,
/// bit decomposition, selects driven by decomposed bits, a chain of hash
/// compressions, hints and public output.
///
/// ## It contains no `permute`, deliberately
///
/// It used to end on a raw `b.permute`, which made it unprovable under the
/// machine's real hash — a REGISTERED program whose cryptographic meaning
/// depended on a placeholder permutation, which is the disclosure this whole
/// effort exists to retire. The permutation is now a third `compress`: every
/// registry entry is provable under every hasher, and the swap is marginally
/// cheaper besides.
///
/// Permute mode did not disappear with it — `Test` and `Poseidon` still
/// implement it, and it still needs coverage or the arms rot. But coverage does
/// not need a registry ENTRY: [`permute_coverage_program_source`] exercises the
/// arms without claiming a program identity.
pub fn trivial_program_source() -> LfmProgramSource {
    let mut b = LfmBuilder::new();

    let arena = b.declare_arena(4);
    let h: Vec<Cell> = (0..4).map(|i| b.hint_word(arena, i)).collect();

    // Base-field leg: s = 16, m = 112, q = m/s = 7; assert q == x.
    let x = b.felt_const(FE::from(7u64));
    let y = b.felt_const(FE::from(9u64));
    let s = b.add(x, y);
    let m = b.mul(s, x);
    let q = b.div(m, s);
    b.assert_eq(q, x);

    // Fp3 leg: product, Horner step, base scaling.
    let e1 = b.ext_const(&FEE::new([FE::from(1u64), FE::from(2u64), FE::from(3u64)]));
    let e2 = b.ext_const(&FEE::new([FE::from(4u64), FE::from(5u64), FE::from(6u64)]));
    let p = b.emul(e1, e2);
    let pm = b.emul_add(p, e1, e2);
    let _pb = b.emul_base(pm, q);

    // Bit-decomposition leg: m = 112 = 0b1110000; bits drive the selects.
    let bits = b.bit_dec(m, 8);
    let (l, _r) = b.select(bits[4], h[0], h[1]); // bit 4 of 112 = 1 → swap
    let (l2, _r2) = b.select(bits[0], l, h[2]); // bit 0 = 0 → pass through

    // Hash leg: three compressions chained through memory. Feeding `d1` back in
    // is the point — a socket's own output must be a legal input to the next
    // one, which is what a Merkle walk does at every level.
    //
    // ✓ SWEPT for the leaf-mode migration and deliberately LEFT as `compress`:
    // these are the only place in any registered program where raw arena data
    // enters a compress, and they form a CHAIN, not a tree. There is no leaf and
    // no parent here, so there is no leaf/parent confusion for `MODE_L` to
    // separate — what the mode buys elsewhere it would not buy here. The
    // consequence to keep in mind is that this program's arena words must be
    // `u32`-laned under BLAKE3 (obligation O1), which its tests supply; data
    // that cannot be is what `leaf` exists for.
    let d0 = b.compress(h[0].as_digest(), h[1].as_digest());
    let d1 = b.compress(d0, l2.as_digest());
    let d2 = b.compress(d1, h[3].as_digest());

    // Public output: two chained digests and one ALU result.
    b.public(d1.as_cell());
    b.public(d2.as_cell());
    b.public(m.as_cell());

    b.finish()
}

pub fn trivial_program() -> LfmProgram {
    compile(trivial_program_source())
}

/// A `permute`-mode fixture — **not a registry entry, and it must not become
/// one.**
///
/// [`trivial_program_source`] gave up its raw `b.permute` so that every
/// registered program runs under the machine's real hash. Permute mode is still
/// live under `Test` and `Poseidon`, so it still needs a program that exercises
/// the executor arm, the trace filler and the AIR's three-cell tuple contract —
/// this is that program. It is deliberately unregistered: a registry entry is a
/// claim about a program's identity, and this one exists only to keep two
/// hashers' arms honest.
///
/// It is unprovable under BLAKE3 by design (`MODE_P = 0`), which is itself worth
/// testing.
#[cfg(test)]
pub fn permute_coverage_program_source() -> LfmProgramSource {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(3);
    let h: Vec<Cell> = (0..3).map(|i| b.hint_word(arena, i)).collect();

    // Two permutations chained, so an output cell is also an input cell.
    let s0 = b.permute([h[0], h[1], h[2]]);
    let s1 = b.permute([s0[2], s0[0], s0[1]]);

    for c in s1 {
        b.public(c);
    }
    b.finish()
}

#[cfg(test)]
pub fn permute_coverage_program() -> LfmProgram {
    compile(permute_coverage_program_source())
}

/// Number of arena words the keccak-chain program ingests: one full state.
pub const KECCAK_CHAIN_ARENA_WORDS: u32 = super::layout::keccak::NUM_WORDS as u32;

/// The R1b keccak program: a hint-fed state pushed through two *chained*
/// `keccak-f[1600]` permutations, the second consuming the first's output words
/// directly out of memory.
///
/// Chaining is the point. It proves the `u32`-half word convention round-trips:
/// the output words `LFM_KECCAK` writes are immediately legal input words, so
/// the halves it produces are canonical `u32`s and the state's two unused top
/// lanes come back zero — no repacking instruction in between.
pub fn keccak_chain_program_source() -> LfmProgramSource {
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Keccak);

    let arena = b.declare_arena(KECCAK_CHAIN_ARENA_WORDS);
    let state: [Cell; 13] = core::array::from_fn(|i| b.hint_word(arena, i as u32));

    let once = b.keccak_f(state);
    let twice = b.keccak_f(once);

    // Expose enough to pin both permutations: the intermediate state's first
    // word and the final state's first two.
    b.public(once[0]);
    b.public(twice[0]);
    b.public(twice[1]);

    b.finish()
}

pub fn keccak_chain_program() -> LfmProgram {
    compile(keccak_chain_program_source())
}

/// Message length of the registered `KeccakSpongeV0`.
///
/// 202 bytes is chosen to exercise all three shapes at once: it crosses the
/// 136-byte rate boundary (2 blocks), it is not a multiple of the rate (so the
/// padding is not a whole block), and `202 % 4 == 2` puts the `0x01` pad byte in
/// the same `u32` half as the message's last two bytes — the mixed-half case the
/// emitter handles by adding a padding constant to the stream half.
pub const KECCAK_SPONGE_LEN: usize = 202;

/// `keccak256` over a hint-supplied byte stream of exactly `len_bytes`, with
/// the 32-byte digest as public output.
///
/// Length is program shape, not data: a straight-line machine has no loops, so
/// each length compiles to its own program and its own identity.
pub fn keccak_sponge_program_source(len_bytes: usize) -> LfmProgramSource {
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Keccak);
    let num_halves = super::keccak_host::num_stream_halves(len_bytes) as u32;
    let arena = b.declare_arena(num_halves);
    let stream: Vec<_> = (0..num_halves).map(|i| b.hint_felt(arena, i)).collect();
    let digest = super::edsl::keccak256(&mut b, &stream, len_bytes);
    b.public(digest[0]);
    b.public(digest[1]);
    b.finish()
}

pub fn keccak_sponge_program(len_bytes: usize) -> LfmProgram {
    compile(keccak_sponge_program_source(len_bytes))
}

/// `Blake3Chain` over a hint-supplied byte stream of exactly `len_bytes`, with
/// the 32-byte digest as public output — the BLAKE3 twin of
/// [`keccak_sponge_program_source`], and the smallest program that exercises
/// `LFM_BLAKE3` end to end.
///
/// The one place in this module that SELECTS the configured hash rather than
/// inheriting the default. Deliberately **not registered**: it is an
/// instrument, and adding registry rows is its own decision — `resolve` keys on
/// `(kind, blowup_factor)` alone, so rows are not a free-form extension point.
pub fn blake3_sponge_program_source(len_bytes: usize) -> LfmProgramSource {
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let num_halves = super::keccak_host::num_stream_halves(len_bytes) as u32;
    let arena = b.declare_arena(num_halves);
    let stream: Vec<_> = (0..num_halves).map(|i| b.hint_felt(arena, i)).collect();
    // The builder is pinned to BLAKE3 above, so this names the byte hash
    // directly — which a program that is ABOUT a hash should do anyway.
    let digest = super::edsl::wrap_hash_bytes(
        &mut b,
        super::edsl::ByteWrapHash::Blake3,
        &stream,
        len_bytes,
    );
    b.public(digest[0]);
    b.public(digest[1]);
    b.finish()
}

pub fn blake3_sponge_program(len_bytes: usize) -> LfmProgram {
    compile(blake3_sponge_program_source(len_bytes))
}

/// `DefaultTranscript::sample()` over a hint-supplied stream: keccak256 of the
/// absorbed bytes, then the 32 digest bytes REVERSED — which is both the
/// challenge the transcript returns and the prefix it re-absorbs.
///
/// This is the R1d groundwork that is independent of the #841 revision:
/// `sample()` itself is unchanged between them.
pub fn keccak_sample_program_source(len_bytes: usize) -> LfmProgramSource {
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Keccak);
    let num_halves = super::keccak_host::num_stream_halves(len_bytes) as u32;
    let arena = b.declare_arena(num_halves);
    let stream: Vec<_> = (0..num_halves).map(|i| b.hint_felt(arena, i)).collect();
    let rev = super::edsl::keccak256_rev(&mut b, &stream, len_bytes);
    b.public(rev[0]);
    b.public(rev[1]);
    b.finish()
}

pub fn keccak_sample_program(len_bytes: usize) -> LfmProgram {
    compile(keccak_sample_program_source(len_bytes))
}

// ==================== R1d: the transcript replay ====================

/// Seed the registered transcript-replay program starts from — a program
/// constant, exactly as a domain separator would be. 24 bytes, so the segment
/// stays half-aligned for the machine-supplied absorbs that follow.
pub const TRANSCRIPT_SEED: &[u8] = b"lfm-transcript-replay-v0";

/// First absorb: 32 bytes, the shape a commitment root arrives in.
pub const TRANSCRIPT_ABSORB_A: usize = 32;

/// Second absorb: one full keccak rate, chosen so the segment it lands in
/// (32 reversed-digest bytes + 136) needs TWO rate blocks — the multi-block
/// path inside a replay, which no earlier test reaches.
pub const TRANSCRIPT_ABSORB_B: usize = 136;

/// Arena words the replay program ingests: both absorbs as `u32` halves.
pub const TRANSCRIPT_ARENA_HALVES: u32 =
    ((TRANSCRIPT_ABSORB_A + TRANSCRIPT_ABSORB_B) / super::keccak_host::BYTES_PER_HALF) as u32;

/// Index bits the replay program's `sample_u64` draw asks for.
pub const TRANSCRIPT_QUERY_BITS: usize = 20;

/// The R1d headline program: a scripted `DefaultTranscript` interleaving,
/// replayed in the machine, with every sampled value published.
///
/// The script is chosen so the emitter's bookkeeping is load-bearing at every
/// step. Buffer positions, in bytes, as the emitter tracks them:
///
/// | step                  | before | after | squeeze |
/// |-----------------------|--------|-------|---------|
/// | `append` A (32 B)     | 32     | 32    | —       |
/// | `sample_felt`         | 32     | 8     | **#1**  |
/// | `sample_felt`         | 8      | 16    | —       |
/// | `sample_ext` (3 draws)| 16     | 8     | **#2**  |
/// | `append` B (136 B)    | 8      | 32    | —       |
/// | `sample_u64_pow2`     | 32     | 8     | **#3**  |
/// | `sample_felt`         | 8      | 16    | —       |
/// | `sample()`            | 16     | 32    | **#4**  |
/// | `sample_felt`         | 32     | 8     | **#5**  |
///
/// So it exercises: a refill in the MIDDLE of an extension draw (squeeze #2
/// lands between coordinates 1 and 2), an absorb that invalidates a buffer with
/// 24 bytes still in it, a raw `sample()` that invalidates with 16 bytes still
/// in it, a two-block segment (squeeze #3), and both draw kinds. Get any of the
/// invalidation rules wrong and the values diverge from the real transcript.
pub fn transcript_replay_program_source() -> LfmProgramSource {
    use super::builder::Felt;
    use super::edsl::bits_to_felt;
    use super::transcript_replay::TranscriptReplay;

    let halves_a = TRANSCRIPT_ABSORB_A / super::keccak_host::BYTES_PER_HALF;

    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let arena = b.declare_arena(TRANSCRIPT_ARENA_HALVES);
    let halves: Vec<Felt> = (0..TRANSCRIPT_ARENA_HALVES)
        .map(|i| b.hint_felt(arena, i))
        .collect();
    let (absorb_a, absorb_b) = halves.split_at(halves_a);

    let mut t = TranscriptReplay::new(TRANSCRIPT_SEED);
    t.append_halves(absorb_a);
    let f0 = t.sample_felt(&mut b);
    let f1 = t.sample_felt(&mut b);
    let e = t.sample_ext(&mut b);
    t.append_halves(absorb_b);
    let q = t.sample_u64_pow2(&mut b, TRANSCRIPT_QUERY_BITS);
    let qf = bits_to_felt(&mut b, &q);
    let f2 = t.sample_felt(&mut b);
    let s = t.sample(&mut b);
    let f3 = t.sample_felt(&mut b);

    b.public(f0.as_cell());
    b.public(f1.as_cell());
    b.public(e.as_cell());
    b.public(qf.as_cell());
    b.public(f2.as_cell());
    b.public(s[0]);
    b.public(s[1]);
    b.public(f3.as_cell());
    b.finish()
}

pub fn transcript_replay_program() -> LfmProgram {
    compile(transcript_replay_program_source())
}

/// Absorbs a machine-COMPUTED keccak digest and samples one challenge from it —
/// the shape a commitment root takes in a real verifier, and the only path that
/// exercises `append_digest`'s word-to-halves byte order.
///
/// Not registered: it exists to pin that byte order against the real transcript,
/// which execution alone establishes (the executor computes the digest FROM the
/// unpacked halves, so a wrong order moves the sampled value).
pub fn transcript_absorb_digest_program_source(len_bytes: usize) -> LfmProgramSource {
    use super::builder::Felt;
    use super::transcript_replay::TranscriptReplay;

    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let num_halves = super::keccak_host::num_stream_halves(len_bytes) as u32;
    let arena = b.declare_arena(num_halves);
    let stream: Vec<Felt> = (0..num_halves).map(|i| b.hint_felt(arena, i)).collect();
    let digest = super::edsl::keccak256(&mut b, &stream, len_bytes);

    let mut t = TranscriptReplay::new(TRANSCRIPT_SEED);
    t.append_digest(&mut b, &digest);
    let f = t.sample_felt(&mut b);
    b.public(f.as_cell());
    b.finish()
}

pub fn transcript_absorb_digest_program(len_bytes: usize) -> LfmProgram {
    compile(transcript_absorb_digest_program_source(len_bytes))
}

// ============ R1e slice a: field elements on the wire (big-endian) ============

/// Absorbs one hint-supplied BASE field element the way `append_field_element`
/// streams it (canonical `u64`, 8 bytes big-endian) and returns the raw squeeze.
///
/// Publishing `sample()` rather than a sampled challenge is deliberate: the test
/// then compares the 32 squeezed bytes directly, so a failure means the ABSORBED
/// BYTES are wrong and nothing else. Not registered — proved through
/// `verify_against`, like the per-length keccak programs.
pub fn append_felt_program_source() -> LfmProgramSource {
    use super::transcript_replay::TranscriptReplay;

    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let arena = b.declare_arena(1);
    let v = b.hint_felt(arena, 0);
    let mut t = TranscriptReplay::new(TRANSCRIPT_SEED);
    t.append_felt(&mut b, v);
    let s = t.sample(&mut b);
    b.public(s[0]);
    b.public(s[1]);
    b.finish()
}

pub fn append_felt_program() -> LfmProgram {
    compile(append_felt_program_source())
}

/// The same for one CUBIC-EXTENSION element: coordinates 0, 1, 2, each 8 bytes
/// big-endian, 24 bytes total.
pub fn append_ext_program_source() -> LfmProgramSource {
    use super::builder::Felt;
    use super::transcript_replay::TranscriptReplay;

    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let arena = b.declare_arena(3);
    let coords: [Felt; 3] = core::array::from_fn(|i| b.hint_felt(arena, i as u32));
    let mut t = TranscriptReplay::new(TRANSCRIPT_SEED);
    t.append_ext(&mut b, coords);
    let s = t.sample(&mut b);
    b.public(s[0]);
    b.public(s[1]);
    b.finish()
}

pub fn append_ext_program() -> LfmProgram {
    compile(append_ext_program_source())
}

// ==================== R1e slice b: the byte-level splice ====================

/// Deterministic constant bytes for the splice programs; the tests build the
/// host reference from the same function, so the two cannot drift apart.
pub fn splice_prefix(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(37).wrapping_add(11))
        .collect()
}

/// Deterministic machine-supplied bytes for the splice programs.
pub fn splice_dynamic(len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(53).wrapping_add(29))
        .collect()
}

/// A constant prefix of `prefix_len` bytes followed by `num_halves` hinted
/// machine halves, then a raw squeeze. The shift under test is
/// `prefix_len % 4`; at 0 it takes the aligned fast path and serves as control.
pub fn splice_program_source(prefix_len: usize, num_halves: u32) -> LfmProgramSource {
    use super::builder::Felt;
    use super::transcript_replay::TranscriptReplay;

    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let arena = b.declare_arena(num_halves);
    let halves: Vec<Felt> = (0..num_halves).map(|i| b.hint_felt(arena, i)).collect();
    let mut t = TranscriptReplay::new(&splice_prefix(prefix_len));
    t.append_halves_misaligned(&halves);
    let s = t.sample(&mut b);
    b.public(s[0]);
    b.public(s[1]);
    b.finish()
}

pub fn splice_program(prefix_len: usize, num_halves: u32) -> LfmProgram {
    compile(splice_program_source(prefix_len, num_halves))
}

/// Tag length of the alternating splice program — the real
/// `LAMBDAVM_CONTINUATION_EPOCH_V2` is exactly this long.
pub const SPLICE_ALT_TAG: usize = 30;
pub const SPLICE_ALT_DIGEST_HALVES: u32 = 8;
pub const SPLICE_ALT_FIELD_HALVES: u32 = 2;

/// The continuation-epoch statement's shape in miniature: alternating constant
/// and dynamic runs, with the shift CHANGING mid-stream.
///
/// The byte offsets are the whole point. A 30-byte tag leaves shift 2; the
/// 32-byte digest and an 8-byte field keep it there; then a ONE-byte field —
/// standing for the real encoding's `fri_final_poly_log_degree` — moves every
/// later dynamic value to shift 3. A splice that handles only a single fixed
/// shift passes the fixed-prefix test above and fails this one.
pub fn splice_alternating_program_source() -> LfmProgramSource {
    use super::builder::Felt;
    use super::transcript_replay::TranscriptReplay;

    let total = SPLICE_ALT_DIGEST_HALVES + 2 * SPLICE_ALT_FIELD_HALVES;
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let arena = b.declare_arena(total);
    let h: Vec<Felt> = (0..total).map(|i| b.hint_felt(arena, i)).collect();
    let d = SPLICE_ALT_DIGEST_HALVES as usize;
    let f = SPLICE_ALT_FIELD_HALVES as usize;

    let mut t = TranscriptReplay::new(&splice_prefix(SPLICE_ALT_TAG));
    t.append_halves_misaligned(&h[..d]);
    t.append_const_bytes(&splice_prefix(8));
    t.append_halves_misaligned(&h[d..d + f]);
    t.append_const_bytes(&splice_prefix(1));
    t.append_halves_misaligned(&h[d + f..]);
    let s = t.sample(&mut b);
    b.public(s[0]);
    b.public(s[1]);
    b.finish()
}

pub fn splice_alternating_program() -> LfmProgram {
    compile(splice_alternating_program_source())
}

// ============ R1e slices c+d: the epoch statement and Phase A ============

/// Public-output length of the acceptance shape. Deliberately NOT a multiple of
/// four: an epoch's public output is collected one byte per COMMIT op, so the
/// unaligned case is the general one and the acceptance must exercise it.
pub const STMT_PUBLIC_OUTPUT_LEN: usize = 14;

/// Whether each of the acceptance shape's sub-proofs is preprocessed. Mixed on
/// purpose: the verifier absorbs a preprocessed commitment only for the airs
/// that have one, so a replay that absorbs unconditionally must diverge.
pub const STMT_PREPROCESSED: [bool; 3] = [true, false, true];

/// Halves per 32-byte commitment.
const ROOT_HALVES: u32 = 8;

/// Arena halves the statement-replay program reads.
pub fn stmt_arena_halves() -> u32 {
    let vars = ROOT_HALVES + STMT_PUBLIC_OUTPUT_LEN.div_ceil(4) as u32 + 2;
    let roots: u32 = STMT_PREPROCESSED
        .iter()
        .map(|&p| if p { 2 * ROOT_HALVES } else { ROOT_HALVES })
        .sum();
    vars + roots
}

/// The acceptance shape's shape-static statement fields.
pub fn epoch_statement_shape() -> super::statement_replay::EpochStatementShape {
    super::statement_replay::EpochStatementShape {
        public_output_len: STMT_PUBLIC_OUTPUT_LEN,
        table_counts: [3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        num_private_input_pages: 2,
        fri_final_poly_log_degree: 7,
        page_ranges: vec![(0x1000, 4), (0x8000, 1)],
    }
}

/// The R1e headline program: a continuation-epoch statement bound into the
/// transcript, then Phase A over three sub-proofs, publishing the shared LogUp
/// challenges `z` and `α`.
///
/// This is the first leg of a real verifier the machine runs end to end —
/// everything a `multi_verify` does before the per-table forks. What `z` and `α`
/// feed into (the bus-balance replay, the chaining obligations) is R1f.
pub fn statement_replay_program_source() -> LfmProgramSource {
    use super::builder::Felt;
    use super::statement_replay::{
        EpochStatementVars, PhaseATable, absorb_epoch_statement, replay_phase_a,
    };
    use super::transcript_replay::TranscriptReplay;

    let shape = epoch_statement_shape();
    let total = stmt_arena_halves();
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let arena = b.declare_arena(total);
    let h: Vec<Felt> = (0..total).map(|i| b.hint_felt(arena, i)).collect();

    let out_halves = STMT_PUBLIC_OUTPUT_LEN.div_ceil(4);
    let (elf, rest) = h.split_at(ROOT_HALVES as usize);
    let (public_output, rest) = rest.split_at(out_halves);
    let (epoch_label, mut roots) = rest.split_at(2);

    // The verifier seeds an empty transcript and binds the statement first.
    let mut t = TranscriptReplay::new(&[]);
    absorb_epoch_statement(
        &mut t,
        &shape,
        &EpochStatementVars {
            elf_digest: elf,
            public_output,
            epoch_label,
        },
    );

    let mut tables = Vec::new();
    for &preprocessed in &STMT_PREPROCESSED {
        let prep = if preprocessed {
            let (p, r) = roots.split_at(ROOT_HALVES as usize);
            roots = r;
            Some(p)
        } else {
            None
        };
        let (main, r) = roots.split_at(ROOT_HALVES as usize);
        roots = r;
        tables.push(PhaseATable {
            // This driver supplies every root as arena cells on purpose: it is the
            // statement/Phase-A differential, and where a root COMES FROM is the
            // assembled verifier's decision (ledger entry 7), not this program's.
            preprocessed_root: prep.map(super::statement_replay::PhaseAPreprocessed::Cells),
            main_root: main,
        });
    }
    let (z, alpha) = replay_phase_a(&mut t, &mut b, &tables);

    b.public(z.as_cell());
    b.public(alpha.as_cell());
    b.finish()
}

pub fn statement_replay_program() -> LfmProgram {
    compile(statement_replay_program_source())
}

/// A harness for the candidate canonicity guard alone: `(lo, hi)` arrive as
/// hinted halves, the guard runs, the recomposed felt is published.
///
/// Not a sound construction on its own — nothing here range-checks the hinted
/// halves to `u32`, which the derivation in
/// [`super::transcript_replay::assert_canonical`] assumes. In the replay they
/// come from an `Unpack` of a `LFM_KECCAK` output word and the adapter
/// range-checks them. This program exists so the guard's PREDICATE can be
/// exercised at the `p − 2 / p − 1 / p` boundary, which is unreachable through
/// the replay: finding a message whose digest yields an out-of-range candidate
/// means about 2^32 keccaks.
pub fn canonicity_guard_program_source() -> LfmProgramSource {
    use super::transcript_replay::{Candidate, assert_canonical, candidate_to_felt};

    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let arena = b.declare_arena(2);
    let c = Candidate {
        lo: b.hint_felt(arena, 0),
        hi: b.hint_felt(arena, 1),
    };
    assert_canonical(&mut b, c);
    let v = candidate_to_felt(&mut b, c);
    b.public(v.as_cell());
    b.finish()
}

pub fn canonicity_guard_program() -> LfmProgram {
    compile(canonicity_guard_program_source())
}

/// The Milestone-C verifier program: verifies the fixture FRI
/// commitment-opening proof (`fixture::fixture_prove`) — sponge transcript
/// replay, two Merkle-authenticated opening sets, the α-combination Horner,
/// two unnormalized folds with index-bit-derived domain-point inverses, and
/// the terminal-polynomial check. Straight-line: every loop below unrolls at
/// emission; the shape is a compile-time constant of the program.
pub fn fri_toy_program_source() -> LfmProgramSource {
    use super::builder::Cell;
    use super::edsl::{self, SpongeVar};
    use super::fixture::{domain_constants, shape};

    let (omega, offset) = domain_constants();
    let omega_inv = omega.inv().expect("root of unity is invertible");
    let offset_inv = offset.inv().expect("coset offset is invertible");
    // Fold-0 point inverses over q0's bits: x = c·ω^{q0} ⇒ factors ω^{-2^i}.
    let invx_factors: Vec<FE> = (0..shape::QUERY_BITS)
        .map(|i| omega_inv.pow(1u64 << i))
        .collect();
    // Fold-1 over j = q0 mod 8: y = c²·ω^{2j} ⇒ factors ω^{-2·2^i}, scale c⁻².
    let invy_factors: Vec<FE> = (0..3).map(|i| omega_inv.pow(2u64 << i)).collect();
    let offset2_inv = offset_inv.square();
    // Terminal point y₂ = c⁴·ω^{4j}.
    let y2_factors: Vec<FE> = (0..3).map(|i| omega.pow(4u64 << i)).collect();
    let offset4 = offset.square().square();

    let mut b = LfmBuilder::new();
    let commits = b.declare_arena(4);
    let opens = b.declare_arena((shape::NUM_QUERIES * shape::WORDS_PER_QUERY) as u32);

    let mut sponge = SpongeVar::new(&mut b);
    let main_root = b.hint_word(commits, 0);
    sponge.absorb(&mut b, main_root);
    let alpha = sponge.squeeze_ext(&mut b);
    let zeta0 = sponge.squeeze_ext(&mut b);
    let l1_root = b.hint_word(commits, 1);
    sponge.absorb(&mut b, l1_root);
    let zeta1 = sponge.squeeze_ext(&mut b);
    let t0w = b.hint_word(commits, 2);
    let t1w = b.hint_word(commits, 3);
    // The terminal coefficients are field DATA, not digests, so they enter the
    // transcript through the leaf encoding — the same rule the trees follow.
    sponge.absorb_felts(&mut b, t0w);
    sponge.absorb_felts(&mut b, t1w);
    let t0 = t0w.as_ext();
    let t1 = t1w.as_ext();

    // Hoisted reference lanes for the per-query root comparisons.
    let main_root_lanes = b.unpack(main_root);
    let l1_root_lanes = b.unpack(l1_root);

    for q in 0..shape::NUM_QUERIES {
        let off = (q * shape::WORDS_PER_QUERY) as u32;
        let bits = sponge.squeeze_bits(&mut b, shape::QUERY_BITS); // q0 = b0..b3
        let zero_bit = b.bit_const(false);
        let one_bit = b.bit_const(true);
        let path_a = [bits[1], bits[2], bits[3], zero_bit];
        let path_b = [bits[1], bits[2], bits[3], one_bit];

        // Main-tree opening A (rows 2·l_A, 2·l_A+1 with l_A = q0 >> 1).
        let row_a_even = b.hint_word(opens, off);
        let row_a_odd = b.hint_word(opens, off + 1);
        let leaf_a = edsl::leaf_hash_pair(&mut b, row_a_even, row_a_odd);
        let sibs_a: Vec<Cell> = (0..4).map(|i| b.hint_word(opens, off + 2 + i)).collect();
        let root_a = edsl::merkle_walk(&mut b, leaf_a, &path_a, &sibs_a);
        edsl::assert_word_eq_lanes(&mut b, root_a.as_cell(), &main_root_lanes);

        // Main-tree opening B (leaf l_A + 8, i.e. rows q0+16's pair).
        let row_b_even = b.hint_word(opens, off + 6);
        let row_b_odd = b.hint_word(opens, off + 7);
        let leaf_b = edsl::leaf_hash_pair(&mut b, row_b_even, row_b_odd);
        let sibs_b: Vec<Cell> = (0..4).map(|i| b.hint_word(opens, off + 8 + i)).collect();
        let root_b = edsl::merkle_walk(&mut b, leaf_b, &path_b, &sibs_b);
        edsl::assert_word_eq_lanes(&mut b, root_b.as_cell(), &main_root_lanes);

        // Row parity: q0 and q0+16 share bit 0.
        let (row_a, _) = b.select(bits[0], row_a_even, row_a_odd);
        let (row_b, _) = b.select(bits[0], row_b_even, row_b_odd);

        // g0 at the two points: α-combination of the opened row columns.
        let la = b.unpack(row_a);
        let lo = edsl::horner_ext(
            &mut b,
            alpha,
            &[
                la[0].as_ext(),
                la[1].as_ext(),
                la[2].as_ext(),
                la[3].as_ext(),
            ],
        );
        let lb = b.unpack(row_b);
        let hi = edsl::horner_ext(
            &mut b,
            alpha,
            &[
                lb[0].as_ext(),
                lb[1].as_ext(),
                lb[2].as_ext(),
                lb[3].as_ext(),
            ],
        );

        // Fold 0 → must equal the opened g1[q0].
        let inv_x = edsl::pow_bits(&mut b, &bits, &invx_factors, offset_inv);
        let v1 = edsl::fri_fold(&mut b, lo, hi, zeta0, inv_x);

        let l1_lo = b.hint_word(opens, off + 12);
        let l1_hi = b.hint_word(opens, off + 13);
        let l1_leaf = edsl::leaf_hash_pair(&mut b, l1_lo, l1_hi);
        let l1_sibs: Vec<Cell> = (0..3).map(|i| b.hint_word(opens, off + 14 + i)).collect();
        let l1_path = [bits[0], bits[1], bits[2]];
        let l1_root_c = edsl::merkle_walk(&mut b, l1_leaf, &l1_path, &l1_sibs);
        edsl::assert_word_eq_lanes(&mut b, l1_root_c.as_cell(), &l1_root_lanes);

        let (g1_at_q0, _) = b.select(bits[3], l1_lo, l1_hi);
        b.assert_eq_ext(v1, g1_at_q0.as_ext());

        // Fold 1 → must equal the terminal polynomial at y₂.
        let inv_y = edsl::pow_bits(&mut b, &bits[0..3], &invy_factors, offset2_inv);
        let v2 = edsl::fri_fold(&mut b, l1_lo.as_ext(), l1_hi.as_ext(), zeta1, inv_y);

        let y2 = edsl::pow_bits(&mut b, &bits[0..3], &y2_factors, offset4);
        let t1y = b.emul_base(t1, y2);
        let t_eval = b.eadd(t0, t1y);
        b.assert_eq_ext(v2, t_eval);
    }

    b.public(main_root);
    b.public(l1_root);
    b.finish()
}

pub fn fri_toy_program() -> LfmProgram {
    compile(fri_toy_program_source())
}

// ============ R1f: a real Merkle opening under the production hash ============

/// Everything about a Merkle-opening program that is compile-time.
///
/// Both fields are SHAPE, in the sense of `others/lfm-target-shape.md`: they fix
/// how many arena words the program reads, how many byteswaps it emits and how
/// many permutations the walk costs. A program that read them from an arena
/// would be claiming to authenticate a tree whose geometry the prover chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MerkleOpeningShape {
    /// Field elements in the leaf. A leaf is a row PAIR (`ROWS_PER_LEAF = 2`),
    /// so this is `2 × columns`.
    pub leaf_values: usize,
    /// Tree depth: index bits consumed, siblings read, permutations walked.
    pub depth: usize,
}

impl MerkleOpeningShape {
    pub const fn columns(self) -> usize {
        self.leaf_values / 2
    }
}

/// Authenticates one FRI query's main-trace opening against a committed root,
/// under the PRODUCTION keccak Merkle conventions.
///
/// Four arenas, each field in its own words (the R1e packing rule):
///
/// 0. the leaf's field elements, one base word each, in hash order
///    (`evaluations ‖ evaluations_sym`);
/// 1. the sibling digests, two `u32`-half words per level, LEAF LEVEL FIRST;
/// 2. the leaf index, one base word;
/// 3. the committed root, two `u32`-half words.
///
/// The walked root is asserted equal to arena 3 and then PUBLISHED. Both matter
/// and they do different jobs. The assert is the composition-ready shape — in
/// the assembled verifier the expected root arrives exactly like this, as an
/// arena value that Phase A has already bound into the transcript, and
/// `fri_toy_program` compares its roots the same way. Publishing is what makes
/// the result a claim rather than an internal fact: public words are absorbed
/// into the LFM statement, so a verifier that supplies the real committed root
/// as the claimed output is checking the machine reached THAT root and not some
/// other one the prover found convenient.
///
/// ## What this program does and does not bind
///
/// It binds the leaf, the path and the low `depth` bits of the index to the
/// root. It does not bind the index to a transcript — `bit_dec` constrains the
/// hinted index to its own decomposition and the walk uses the low `depth`
/// bits, so a prover may add any multiple of `2^depth` without changing
/// anything. That is correct here and unsound alone: in the assembled verifier
/// the bits come from `TranscriptReplay::sample_u64_pow2`, which produces
/// exactly this `Vec<Bit>` from a squeezed candidate. This program is the
/// authentication half of that pair, built and measured before the sampler is
/// wired to it.
pub fn keccak_merkle_opening_program_source(shape: MerkleOpeningShape) -> LfmProgramSource {
    // Keccak by ARGUMENT, not by omission. This is the R1f instrument and a
    // registry fixture: its identity is pinned in `LFM_REGISTRY`, so it stays
    // keccak whatever the wrap's configured hash becomes. A BLAKE3 twin is a
    // second program (`merkle_opening_program_source_with_hash`) and would be a
    // second registry row, never a re-blessing of this one.
    merkle_opening_program_source_with_hash(shape, super::edsl::WrapHash::Keccak)
}

/// [`keccak_merkle_opening_program_source`] under an explicitly chosen wrap
/// hash — the shared body, so the two hashes exercise one emitter rather than
/// two that have to be shown to coincide.
pub fn merkle_opening_program_source_with_hash(
    shape: MerkleOpeningShape,
    wrap_hash: super::edsl::WrapHash,
) -> LfmProgramSource {
    use super::edsl;

    assert!(shape.leaf_values > 0, "a leaf covers at least one column");
    assert!(
        shape.leaf_values.is_multiple_of(2),
        "a leaf is a row PAIR, so it holds an even number of values"
    );
    assert!(
        (1..=32).contains(&shape.depth),
        "depth must be in 1..=32: below, there is no path; above, the index \
         would outrun a single transcript candidate half"
    );

    let mut b = LfmBuilder::new().with_wrap_hash(wrap_hash);
    let leaf_arena = b.declare_arena(shape.leaf_values as u32);
    let sibling_arena = b.declare_arena(2 * shape.depth as u32);
    let index_arena = b.declare_arena(1);
    let root_arena = b.declare_arena(2);

    let values: Vec<_> = (0..shape.leaf_values as u32)
        .map(|i| b.hint_felt(leaf_arena, i))
        .collect();
    let leaf = edsl::wrap_leaf_hash(&mut b, &values);

    let index = b.hint_felt(index_arena, 0);
    let bits = b.bit_dec(index, shape.depth);

    let siblings: Vec<edsl::WrapDigest> = (0..shape.depth as u32)
        .map(|l| {
            edsl::WrapDigest::from_pair(
                b.hint_word(sibling_arena, 2 * l),
                b.hint_word(sibling_arena, 2 * l + 1),
            )
        })
        .collect();

    let root = edsl::wrap_merkle_walk(&mut b, leaf, &bits, &siblings);

    let expected = [b.hint_word(root_arena, 0), b.hint_word(root_arena, 1)];
    edsl::assert_word_eq(&mut b, root[0], expected[0]);
    edsl::assert_word_eq(&mut b, root[1], expected[1]);

    b.public(root[0]);
    b.public(root[1]);
    b.finish()
}

pub fn keccak_merkle_opening_program(shape: MerkleOpeningShape) -> LfmProgram {
    compile(keccak_merkle_opening_program_source(shape))
}

/// The R1f Merkle-opening walk at the hash production commits under — the
/// BLAKE3 twin [`keccak_merkle_opening_program_source`]'s comment anticipates.
///
/// Use this to authenticate a REAL opening: the walk has to re-derive a root
/// the host built, so it must hash the way the host committed. The keccak
/// program above stays exactly as it is — it is the instrument whose identity
/// the registry pins, and this is a second program, not a re-blessing of it.
pub fn merkle_opening_program(shape: MerkleOpeningShape) -> LfmProgram {
    merkle_opening_program_with_hash(shape, WrapHash::production())
}

pub fn merkle_opening_program_with_hash(
    shape: MerkleOpeningShape,
    wrap_hash: super::edsl::WrapHash,
) -> LfmProgram {
    compile(merkle_opening_program_source_with_hash(shape, wrap_hash))
}

// ============ R1g(ii): the cross-epoch L2G commitment binding ============

/// Ties each epoch's own committed L2G root to the corresponding sub-proof of
/// the global proof — `verify_l2g_commitment_binding_view` (`lib.rs:993`),
/// emitted.
///
/// Two arenas, each root in its own two words:
///
/// 0. the per-epoch L2G roots, `EpochProof::l2g_root`, epoch order;
/// 1. the global proof's first `num_epochs` sub-proof main-trace roots.
///
/// Every pair is asserted equal and the epoch side is published. As in
/// [`keccak_merkle_opening_program_source`], the assert is the relation and the
/// publish is what makes it a claim: the equality alone would be satisfied by
/// any two matching arena values, so the published roots are what a verifier
/// pins against the real bundle.
///
/// ## What binds each side, and what this slice does not do
///
/// This program asserts the RELATION. What binds each root to its proof is the
/// composition's job: the epoch root is bound by that epoch's own Phase A
/// absorb, the global root by the global proof's. Until those legs exist, both
/// sides are arena values and a prover could satisfy the equality with two
/// matching lies — which is exactly why the roots are published rather than
/// merely compared.
///
/// ## Why the epoch count is a constant
///
/// `num_epochs` is shape: it fixes how many roots are read and how many asserts
/// are emitted. Production's `final_proof.len() >= epoch_l2g_roots.len()` guard
/// has no counterpart here because a program compiled for `n` epochs cannot read
/// an `n+1`-epoch bundle — the arena schema would not match.
pub fn l2g_binding_program_source(num_epochs: usize) -> LfmProgramSource {
    use super::edsl;

    assert!(num_epochs > 0, "a continuation has at least one epoch");

    let words = 2 * num_epochs as u32;
    let mut b = LfmBuilder::new();
    let epoch_arena = b.declare_arena(words);
    let global_arena = b.declare_arena(words);

    for i in 0..num_epochs as u32 {
        let epoch = [
            b.hint_word(epoch_arena, 2 * i),
            b.hint_word(epoch_arena, 2 * i + 1),
        ];
        let global = [
            b.hint_word(global_arena, 2 * i),
            b.hint_word(global_arena, 2 * i + 1),
        ];
        edsl::assert_word_eq(&mut b, epoch[0], global[0]);
        edsl::assert_word_eq(&mut b, epoch[1], global[1]);
        b.public(epoch[0]);
        b.public(epoch[1]);
    }
    b.finish()
}

pub fn l2g_binding_program(num_epochs: usize) -> LfmProgram {
    compile(l2g_binding_program_source(num_epochs))
}

// ============ R1g(iii): the attestation's program id ============

/// Halves in a `u64` rendered little-endian.
const U64_HALVES: u32 = 2;

/// Everything about a `program_id` fold that is compile-time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgramIdShape {
    /// Page genesis commitments folded in. SHAPE: it fixes the byte length of
    /// the hashed string, hence the block count and every padding position.
    pub num_pages: usize,
}

impl ProgramIdShape {
    /// Bytes the fold hashes — `tag ‖ elf_digest ‖ pc_start ‖ decode ‖ n ‖
    /// (base ‖ commitment)*`.
    pub fn byte_len(self) -> usize {
        use crate::recursion::PROGRAM_ID_TAG;
        PROGRAM_ID_TAG.len() + 32 + 8 + 32 + 8 + 40 * self.num_pages
    }
}

/// Emits `recursion::program_id_from_digest` — the fold the recursion guest
/// commits as the first 32 bytes of its attestation.
///
/// One arena, each field in its own halves (the R1e packing rule):
/// the 32-byte ELF digest, the `u64` entry point, the 32-byte DECODE root, then
/// per page a `u64` base and a 32-byte commitment.
///
/// ## Why the tag makes this the splice case
///
/// `PROGRAM_ID_TAG` is 22 bytes, `≡ 2 (mod 4)`, so the ELF digest immediately
/// after it straddles half boundaries and so does everything behind it — the
/// same shape as R1e's 30-byte epoch tag. [`super::transcript_replay::ByteString`]
/// carries the byte-granular packer that handles it; alignment is a property of
/// the cursor, not of the field.
///
/// ## What this program does NOT establish
///
/// The attestation is deliberately **not self-enforcing**, and emitting the fold
/// in the machine does not change that. The guest uses SUPPLIED roots verbatim
/// without binding them to the inner ELF; the binding happens outside, when a
/// consumer recomputes the id from an ELF it trusts and compares
/// (`recursion::check_attestation`, an expensive native FFT + Merkle pass done
/// once at top level, never in-VM). A machine-emitted attestation inherits that
/// model unchanged — the same consumer-side compare closes it. Do not read
/// "the machine folded the roots" as "the machine bound the roots".
///
/// ## Page ordering
///
/// `program_id_from_digest` SORTS pages by base before folding. This program
/// folds them in supplied order, so the arena filler owes sortedness. That is
/// not a soundness hole: an unsorted fold yields an id that differs from the
/// consumer's recompute, so the proof is rejected there — the prover only
/// breaks their own attestation. It IS a completeness obligation, so it is
/// stated rather than assumed.
pub fn program_id_program_source(shape: ProgramIdShape) -> LfmProgramSource {
    use super::builder::Felt;

    let root_halves = ROOT_HALVES;
    let per_page = U64_HALVES + root_halves;
    let total = root_halves + U64_HALVES + root_halves + per_page * shape.num_pages as u32;

    // ★ Keccak, and it must stay keccak: this mirrors
    // `recursion::program_id_from_digest`, which names `PlatformKeccak256`. The
    // program-identity digest is a host-side artifact that does not move when
    // the commitment hash does, so following the flip here would break the
    // attestation join rather than complete it.
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Keccak);
    let arena = b.declare_arena(total);
    let h: Vec<Felt> = (0..total).map(|i| b.hint_felt(arena, i)).collect();

    let (elf_digest, rest) = h.split_at(root_halves as usize);
    let (pc_start, rest) = rest.split_at(U64_HALVES as usize);
    let (decode, mut pages) = rest.split_at(root_halves as usize);

    let page_cells: Vec<(&[Felt], &[Felt])> = (0..shape.num_pages)
        .map(|_| {
            let (base, r) = pages.split_at(U64_HALVES as usize);
            let (commitment, r) = r.split_at(root_halves as usize);
            pages = r;
            (base, commitment)
        })
        .collect();

    let id = emit_program_id(&mut b, shape, elf_digest, pc_start, decode, &page_cells);
    b.public(id[0]);
    b.public(id[1]);
    b.finish()
}

/// The `program_id` fold over cells the caller already holds — the form the
/// ASSEMBLED verifier needs.
///
/// This exists for assembly ledger entry 7's DECODE half. DECODE's preprocessed
/// commitment is a function of the inner ELF, so it can be neither interned (that
/// would make LFM program identity ELF-dependent) nor left unbound. The
/// resolution ruled on 2026-08-04 is the **attestation join**: the same arena cell
/// Phase A absorbs is the cell this fold consumes, so a prover who substitutes a
/// DECODE root changes the published `program_id` and the consumer's own recompute
/// rejects it. That makes DECODE exactly as bound as `elf_digest` and `pc_start`
/// already are — and the join is only real if it is STRUCTURAL, one cell with two
/// consumers, which is why this takes cells rather than an arena.
///
/// `elf_digest` is the same eight halves the epoch STATEMENT absorbs, so that
/// value's join comes free.
///
/// ⚠ The join's strength is the consumer-side compare
/// (`recursion::check_attestation`), which has zero production call sites. Folding
/// the roots does not bind them by itself; it makes a substitution DETECTABLE by a
/// consumer who performs the ritual.
pub fn emit_program_id(
    b: &mut LfmBuilder,
    shape: ProgramIdShape,
    elf_digest: &[super::builder::Felt],
    pc_start: &[super::builder::Felt],
    decode: &[super::builder::Felt],
    pages: &[(&[super::builder::Felt], &[super::builder::Felt])],
) -> super::edsl::WrapDigest {
    use super::transcript_replay::ByteString;
    use crate::recursion::PROGRAM_ID_TAG;

    assert_eq!(
        elf_digest.len(),
        ROOT_HALVES as usize,
        "the ELF digest is 32 bytes"
    );
    assert_eq!(
        pc_start.len(),
        U64_HALVES as usize,
        "the entry point is one u64"
    );
    assert_eq!(
        decode.len(),
        ROOT_HALVES as usize,
        "the DECODE commitment is 32 bytes"
    );
    assert_eq!(
        pages.len(),
        shape.num_pages,
        "the page count is SHAPE: it fixes the hashed length and every padding \
         position"
    );

    let mut s = ByteString::new();
    s.push_const(PROGRAM_ID_TAG);
    s.push_halves(elf_digest);
    s.push_halves(pc_start);
    s.push_halves(decode);
    s.push_const(&(shape.num_pages as u64).to_le_bytes());
    for (base, commitment) in pages {
        assert_eq!(base.len(), U64_HALVES as usize, "a page base is one u64");
        assert_eq!(
            commitment.len(),
            ROOT_HALVES as usize,
            "a page commitment is 32 bytes"
        );
        s.push_halves(base);
        s.push_halves(commitment);
    }
    assert_eq!(s.len(), shape.byte_len(), "byte accounting must agree");

    // ⛔ **PINNED KECCAK — does not follow the configured wrap hash, and must
    // not.** The host counterpart `recursion::program_id_from_digest` names
    // `PlatformKeccak256` explicitly rather than the configuration's hash, in
    // the same sense `statement::elf_digest` does: this is an INDEPENDENT
    // keccak that identifies a program to consumers, not part of the proof
    // system's commitment layer. Switching it would make the attestation join
    // disagree with every host consumer of a `program_id`, and the disagreement
    // would surface as a consumer-side compare failing, not as an unprovable
    // program.
    //
    // `ByteString` therefore has two hashing methods and this call selects the
    // pinned one deliberately. The naive sweep — "replace every
    // `edsl::keccak256`" — produces a wrong proof exactly here, because
    // grinding (`epoch::emit_grinding_check`) shares this type and DOES follow
    // the configuration.
    let d = s.keccak256(b);
    super::edsl::WrapDigest::from_pair(d[0], d[1])
}

pub fn program_id_program(shape: ProgramIdShape) -> LfmProgram {
    compile(program_id_program_source(shape))
}

// ======== R1g(i): the next epoch's REGISTER preprocessed commitment ========

/// Everything about a REGISTER-derivation program that is compile-time.
///
/// Both fields belong to the INNER proof's `ProofOptions`, and both are SHAPE
/// in the sense of `others/lfm-target-shape.md`: they fix the LDE domain, hence
/// every twiddle, every leaf's byte layout and the whole tree's permutation
/// count. A program that read them from an arena would let the prover pick the
/// domain its commitment was computed over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegisterDerivationShape {
    /// The inner proof's blowup factor.
    pub blowup: usize,
    /// The inner proof's coset offset (`ProofOptions::coset_offset`).
    pub coset_offset: u64,
}

impl RegisterDerivationShape {
    /// Rows in the interpolation domain — `NUM_REGISTER_ADDRESSES` rounded up.
    pub fn num_rows(self) -> usize {
        crate::tables::register::NUM_REGISTER_ADDRESSES.next_power_of_two()
    }

    /// Rows in the LDE domain.
    pub fn lde_rows(self) -> usize {
        self.num_rows() * self.blowup
    }

    /// Merkle leaves — one per row PAIR (`ROWS_PER_LEAF = 2`).
    pub fn leaves(self) -> usize {
        self.lde_rows() / stark::commitment::ROWS_PER_LEAF
    }

    /// Permutations the tree costs: one per leaf plus one per internal node.
    /// Leaves are 48 bytes and parents 64, so each is a single rate block.
    pub fn permutations(self) -> usize {
        2 * self.leaves() - 1
    }
}

/// The REGISTER preprocessed columns' word addresses, in row order.
///
/// Mirrors the private `register::register_word_address_list`, but assembled
/// from the PUBLIC `register_word_addresses` rather than hand-copied, so only
/// the ORDER is restated here. Nothing pins that order locally and nothing
/// needs to: the derived root is compared against production's own
/// `compute_precomputed_commitment_with_fini`, and any disagreement about which
/// address sits in which row moves the root.
fn register_offsets() -> Vec<u64> {
    use crate::tables::register::{NUM_REGISTER_ADDRESSES, register_word_addresses};
    let mut addrs = Vec::with_capacity(NUM_REGISTER_ADDRESSES);
    for reg in 0..32u8 {
        addrs.extend(register_word_addresses(reg));
    }
    addrs.extend(register_word_addresses(254));
    addrs.extend(register_word_addresses(255));
    assert_eq!(
        addrs.len(),
        NUM_REGISTER_ADDRESSES,
        "the register address list must cover every table row"
    );
    addrs
}

/// Derives the next epoch's REGISTER preprocessed commitment from `reg_fini` —
/// `register::compute_precomputed_commitment_with_fini`, emitted.
///
/// Two arenas, one base word per register word address (the R1e packing rule):
///
/// 0. `R_i`, the epoch's INIT register file;
/// 1. `R_{i+1}`, the epoch's `reg_fini`.
///
/// The derived root is PUBLISHED. There is nothing to assert it against, and
/// that is the mechanism rather than an omission — see below.
///
/// ## Why this is a derivation and not a comparison
///
/// The chaining obligation is often written as "check `reg_fini` against the
/// next epoch's supplied REGISTER root". There is no supplied root.
/// `build_epoch_airs` (`continuation.rs:636`) CONSTRUCTS the preprocessed
/// commitment from `register_init` and `reg_fini`, and `VmAirs::new`'s
/// `register_preprocessed` parameter — which every verify caller passes `None`
/// to — must stay unwired: computing the commitment from the values is what
/// ties the values to it. Supply the root instead and `reg_fini` has no
/// remaining role, so a prover could offer a root consistent with a `reg_fini`
/// it never honoured and the cross-epoch chain would go unenforced.
///
/// ## Three columns, one of them free
///
/// Production commits OFFSET ‖ INIT ‖ FINI. OFFSET holds the register word
/// addresses, which are fixed, so its LDE is a program CONSTANT — the shape
/// rule applying in the machine's favour for once. Only INIT and FINI carry
/// arena values and only they pay for a transform, which is why the derivation
/// emits two LDEs for three columns.
///
/// The constant column still costs at leaf-hashing time: its values are
/// byte-swapped into the leaf like any other. That swap is what
/// [`RegisterDerivationShape::permutations`] does NOT count, and the cost test
/// prints both.
///
/// ## What this program does NOT bind
///
/// The two arenas are unbound here. In the assembled verifier `R_{i+1}` is the
/// same vector the next epoch reads as its INIT and the published root is what
/// that epoch's Phase A absorbs; until those joins exist a prover may supply
/// any pair and get the honestly-derived root for it. The derivation is
/// correct in isolation and binds nothing in isolation — the same standing
/// caveat as the L2G binding leg.
pub fn register_derivation_program_source(shape: RegisterDerivationShape) -> LfmProgramSource {
    use crate::tables::register::NUM_REGISTER_ADDRESSES;

    let supplied = NUM_REGISTER_ADDRESSES as u32;
    let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
    let init_arena = b.declare_arena(supplied);
    let fini_arena = b.declare_arena(supplied);
    let init: Vec<_> = (0..supplied).map(|r| b.hint_felt(init_arena, r)).collect();
    let fini: Vec<_> = (0..supplied).map(|r| b.hint_felt(fini_arena, r)).collect();

    let root = emit_register_commitment(&mut b, shape, &init, &fini);
    b.public(root[0]);
    b.public(root[1]);
    b.finish()
}

/// The REGISTER preprocessed commitment over INIT and FINI cells the caller
/// already holds — [`register_derivation_program_source`] without the arenas.
///
/// This is the form the ASSEMBLED verifier needs, and the reason it exists is
/// assembly ledger entries 7 and 2, which close together. The spine declares the
/// register boundary vector as one arena and reads `start_index` out of slot 64
/// for the COMMIT-bus target; passing those very cells here means the root Phase
/// A absorbs is COMPUTED from them. That computation is the binding: production
/// has no arithmetic `start + len` check anywhere, it rebuilds the commitment
/// from the register vectors and rejects unless the absorbed root matches, so
/// `start_index` is tied to the chain exactly when the machine does the same.
///
/// Hinting the root instead — which the spine did until this existed — leaves
/// `start_index` a free arena word: a prover supplies whatever index makes the
/// COMMIT bus close, and the unrelated hinted root satisfies Phase A.
///
/// `init` and `fini` are `NUM_REGISTER_ADDRESSES` cells each. Rows past that are
/// the pooled ZERO constant, matching `zeroed_fe_vec`: production writes only the
/// supplied prefix, and making the padding program text rather than arena data is
/// the same discipline the OOD next-row pruning follows.
pub fn emit_register_commitment(
    b: &mut LfmBuilder,
    shape: RegisterDerivationShape,
    init: &[super::builder::Felt],
    fini: &[super::builder::Felt],
) -> super::edsl::WrapDigest {
    use super::edsl;
    use super::lde::coset_lde;
    use crate::tables::register::{NUM_PREPROCESSED_COLS_WITH_FINI, NUM_REGISTER_ADDRESSES};
    use math::fft::bit_reversing::reverse_index;
    use stark::commitment::ROWS_PER_LEAF;

    assert_eq!(
        NUM_PREPROCESSED_COLS_WITH_FINI, 3,
        "the derivation commits OFFSET ‖ INIT ‖ FINI; a fourth preprocessed \
         column changes the leaf layout and the arena schema together"
    );
    assert_eq!(
        init.len(),
        NUM_REGISTER_ADDRESSES,
        "one INIT cell per register word address"
    );
    assert_eq!(
        fini.len(),
        NUM_REGISTER_ADDRESSES,
        "one FINI cell per register word address"
    );
    let num_rows = shape.num_rows();
    let coset_offset = FE::from(shape.coset_offset);

    // Padding rows are zero in all three columns, exactly as `zeroed_fe_vec`
    // leaves them: production writes only the first NUM_REGISTER_ADDRESSES.
    let zero = b.felt_const(FE::zero());
    let offsets = register_offsets();
    let offset_col: Vec<FE> = (0..num_rows)
        .map(|r| offsets.get(r).map_or(FE::zero(), |&a| FE::from(a)))
        .collect();
    let column = |supplied: &[super::builder::Felt]| {
        (0..num_rows)
            .map(|r| supplied.get(r).copied().unwrap_or(zero))
            .collect::<Vec<_>>()
    };
    let init_col = column(init);
    let fini_col = column(fini);

    // OFFSET is fixed, so its extension is interned constants rather than an
    // emitted transform — and it is taken from PRODUCTION's own transform, not
    // from `lde`'s. That is deliberate: the three columns land in one tree, so
    // a root that matches production pins the emitter against the very function
    // it is emitting, inside the same hash.
    let offset_lde: Vec<_> = {
        use math::polynomial::Polynomial;
        use stark::prover::evaluate_polynomial_on_lde_domain;
        let poly =
            Polynomial::interpolate_fft::<crate::tables::types::GoldilocksField>(&offset_col)
                .expect("the OFFSET column interpolates");
        evaluate_polynomial_on_lde_domain(&poly, shape.blowup, num_rows, &coset_offset)
            .expect("the OFFSET polynomial extends")
            .into_iter()
            .map(|v| b.felt_const(v))
            .collect()
    };
    let init_lde = coset_lde(b, &init_col, shape.blowup, coset_offset);
    let fini_lde = coset_lde(b, &fini_col, shape.blowup, coset_offset);

    // Leaf `i` hashes the bit-reversed rows `2i` and `2i+1`, each written
    // column by column in big-endian — `keccak_leaves_bit_reversed_grouped`.
    let lde_rows = shape.lde_rows();
    let leaves: Vec<_> = (0..shape.leaves())
        .map(|leaf| {
            let mut values = Vec::with_capacity(ROWS_PER_LEAF * NUM_PREPROCESSED_COLS_WITH_FINI);
            for k in 0..ROWS_PER_LEAF {
                let row = reverse_index(ROWS_PER_LEAF * leaf + k, lde_rows as u64);
                values.extend([offset_lde[row], init_lde[row], fini_lde[row]]);
            }
            edsl::wrap_leaf_hash(b, &values)
        })
        .collect();

    edsl::wrap_merkle_tree_root(b, &leaves)
}

pub fn register_derivation_program(shape: RegisterDerivationShape) -> LfmProgram {
    compile(register_derivation_program_source(shape))
}

/// A bare [`super::lde::coset_lde`], publishing every extended value.
///
/// The instrument behind the LDE differential. `register_derivation_program`
/// exercises the transform only at the register shape — `n = 128`, coset offset
/// 3 — and every production REGISTER table has exactly that shape, so a
/// differential over production data cannot distinguish an emitter that is
/// right in general from one that is accidentally right at 128. This drives
/// synthetic sizes and offsets against production's own transform.
pub fn lde_probe_program_source(n: usize, blowup: usize, coset_offset: u64) -> LfmProgramSource {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(n as u32);
    let values: Vec<_> = (0..n as u32).map(|i| b.hint_felt(arena, i)).collect();
    for v in super::lde::coset_lde(&mut b, &values, blowup, FE::from(coset_offset)) {
        b.public(v.as_cell());
    }
    b.finish()
}

pub fn lde_probe_program(n: usize, blowup: usize, coset_offset: u64) -> LfmProgram {
    compile(lde_probe_program_source(n, blowup, coset_offset))
}
