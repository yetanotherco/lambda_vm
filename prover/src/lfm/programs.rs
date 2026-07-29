//! Registered LFM programs.
//!
//! Every program here is deterministic — same builder calls, same
//! instructions, same column groups, same digest — which is what lets the
//! registry pin it and the drift tests recompute it on every PR. Arena
//! *values* vary per proof; the program (and its identity) never does.

use crate::tables::types::{FE, FEE};

use super::builder::{Cell, LfmBuilder, LfmProgramSource};
use super::compiler::{LfmProgram, compile};

/// The Milestone-B trivial program: a few hundred instructions exercising
/// every chip — constants, base ALU (incl. the assert lowering), Fp3 ALU,
/// bit decomposition, selects driven by decomposed bits, both hash modes,
/// hints and public output.
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

    // Hash leg: both modes, chained through memory.
    let d0 = b.compress(h[0].as_digest(), h[1].as_digest());
    let d1 = b.compress(d0, l2.as_digest());
    let st = b.permute([d1.as_cell(), h[3], d0.as_cell()]);

    // Public output: the chained digest, one permuted cell, one ALU result.
    b.public(d1.as_cell());
    b.public(st[0]);
    b.public(m.as_cell());

    b.finish()
}

pub fn trivial_program() -> LfmProgram {
    compile(trivial_program_source())
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
    let mut b = LfmBuilder::new();

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
    let mut b = LfmBuilder::new();
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

/// `DefaultTranscript::sample()` over a hint-supplied stream: keccak256 of the
/// absorbed bytes, then the 32 digest bytes REVERSED — which is both the
/// challenge the transcript returns and the prefix it re-absorbs.
///
/// This is the R1d groundwork that is independent of the #841 revision:
/// `sample()` itself is unchanged between them.
pub fn keccak_sample_program_source(len_bytes: usize) -> LfmProgramSource {
    let mut b = LfmBuilder::new();
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

    let mut b = LfmBuilder::new();
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

    let mut b = LfmBuilder::new();
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

    let mut b = LfmBuilder::new();
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

    let mut b = LfmBuilder::new();
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

    let mut b = LfmBuilder::new();
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
    let mut b = LfmBuilder::new();
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

/// Public-output length of the acceptance shape. A multiple of four, per the
/// documented gap in `statement_replay::absorb_epoch_statement`.
pub const STMT_PUBLIC_OUTPUT_LEN: usize = 12;

/// Whether each of the acceptance shape's sub-proofs is preprocessed. Mixed on
/// purpose: the verifier absorbs a preprocessed commitment only for the airs
/// that have one, so a replay that absorbs unconditionally must diverge.
pub const STMT_PREPROCESSED: [bool; 3] = [true, false, true];

/// Halves per 32-byte commitment.
const ROOT_HALVES: u32 = 8;

/// Arena halves the statement-replay program reads.
pub fn stmt_arena_halves() -> u32 {
    let vars = ROOT_HALVES + (STMT_PUBLIC_OUTPUT_LEN / 4) as u32 + 2;
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
        table_counts: [3, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
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
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(total);
    let h: Vec<Felt> = (0..total).map(|i| b.hint_felt(arena, i)).collect();

    let out_halves = STMT_PUBLIC_OUTPUT_LEN / 4;
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
            preprocessed_root: prep,
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

    let mut b = LfmBuilder::new();
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
    sponge.absorb2(&mut b, t0w, t1w);
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
        let leaf_a = b.compress(row_a_even.as_digest(), row_a_odd.as_digest());
        let sibs_a: Vec<Cell> = (0..4).map(|i| b.hint_word(opens, off + 2 + i)).collect();
        let root_a = edsl::merkle_walk(&mut b, leaf_a, &path_a, &sibs_a);
        edsl::assert_word_eq_lanes(&mut b, root_a.as_cell(), &main_root_lanes);

        // Main-tree opening B (leaf l_A + 8, i.e. rows q0+16's pair).
        let row_b_even = b.hint_word(opens, off + 6);
        let row_b_odd = b.hint_word(opens, off + 7);
        let leaf_b = b.compress(row_b_even.as_digest(), row_b_odd.as_digest());
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
        let l1_leaf = b.compress(l1_lo.as_digest(), l1_hi.as_digest());
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
