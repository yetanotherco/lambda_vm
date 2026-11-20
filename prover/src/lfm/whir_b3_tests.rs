//! The in-guest WHIR verifier's 7-round BLAKE3 arm, against the host.
//!
//! The host is `multilinear::whir_hash::Blake3Whir`: commitments under
//! `Blake3Std`, the transcript `DefaultTranscript<_, Blake3StdTranscriptHash>`,
//! and the grind `is_valid_nonce::<Blake3Std>`. Every gate here runs the same
//! inputs through a [`WrapHash::Blake3`] builder's emitters and through the host
//! functions themselves, and compares what came out. Executed, not proved: the
//! programs run on the executor, whose BLAKE3 rows compute the default build's
//! seven rounds.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::transcript_hash::Blake3StdTranscriptHash;
use crypto::hash::blake3::Blake3Std;

use crate::tables::types::{FE, FEE, GoldilocksExtension};

use super::builder::{Bit, Ext, Felt, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::edsl::{WrapDigest, WrapHash};
use super::executor::execute;
use super::instr::Instr;
use super::validator::validate;
use super::whir_transcript::{WhirTranscript, emit_grind_check};
use super::whir_transcript_kats::{Observed, Script, Step, bytes_from, ext_from, run_with};
use super::word::{LfmWord, ext_word, word_as_ext};

type HostTranscript = DefaultTranscript<GoldilocksExtension, Blake3StdTranscriptHash>;

std::thread_local! {
    /// Whether this thread's programs take the lean tables' rows
    /// ([`LfmBuilder::with_b3_lean`]): each gate runs under both lowerings.
    static LEAN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` once under each lowering — `LFM_BLAKE3`'s rows, then the lean
/// tables' — with the lowering's name: the digests and every refusal must not
/// depend on which chip proves a compression.
fn under_both(mut f: impl FnMut(&str)) {
    for lean in [false, true] {
        LEAN.with(|c| c.set(lean));
        let name = if lean { "lean" } else { "LFM_BLAKE3" };
        println!("lowering: {name}");
        f(name);
    }
    LEAN.with(|c| c.set(false));
}

fn builder() -> LfmBuilder {
    LfmBuilder::new()
        .with_wrap_hash(WrapHash::Blake3)
        .with_b3_lean(LEAN.with(std::cell::Cell::get))
}

fn run_program(program: &LfmProgram, arenas: &[Vec<LfmWord>]) -> Vec<LfmWord> {
    let exec =
        execute(program, arenas, &crate::hash_pin::BLOCK_HASHER).expect("the replay must execute");
    exec.public_words.iter().map(|(_, word)| *word).collect()
}

/// A byte digest's 32 bytes from its two published words: lane `h` of word `w`
/// is the little-endian `u32` of bytes `4(4w + h)..`.
fn digest_bytes(words: &[LfmWord]) -> [u8; 32] {
    assert_eq!(words.len(), 2, "a BLAKE3 digest is two words");
    let mut out = [0u8; 32];
    for (w, word) in words.iter().enumerate() {
        for (h, lane) in word.iter().enumerate() {
            let v = u32::try_from(lane.canonical()).expect("a digest lane is a u32");
            out[4 * (4 * w + h)..4 * (4 * w + h) + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
    out
}

/// A 32-byte digest as the two arena words a hinted root occupies.
fn digest_words(bytes: &[u8; 32]) -> [LfmWord; 2] {
    core::array::from_fn(|w| {
        core::array::from_fn(|h| {
            let at = 4 * (4 * w + h);
            FE::from(u64::from(u32::from_le_bytes(
                bytes[at..at + 4].try_into().expect("four bytes"),
            )))
        })
    })
}

/// The lean field tables' cells for `program`, which
/// `airs::lfm_chip_census_with_hasher` does not count (it lists the LFM chips;
/// the lean tables are apart): `LEAN_MUX` (a lean BLAKE3 program's one-output
/// selects, 13 value columns), and under the indexed cap read `LEAN_CAP` (one
/// row a cap node) and `LEAN_CAP_READ` (one row a read). Every census that
/// prices a leaf's field side adds these.
pub(crate) struct LeanMuxCells {
    /// Value cells at the table's padded heights (its chunks).
    pub padded_main: u64,
    /// LogUp columns at the padded heights, one a pair of interactions.
    pub padded_aux: u64,
    /// Value cells at the real rows.
    pub real_main: u64,
    /// LogUp columns at the real rows.
    pub real_aux: u64,
}

pub(crate) fn lean_mux_cells(program: &LfmProgram) -> LeanMuxCells {
    use super::b3l_tables::B3lShape;
    use super::chips::{cap, cap_read, mux};
    let shape = B3lShape::of_program(program);
    // A program with no compression has no lean table, so none of these rows.
    let lean = !shape.is_empty();
    let g = &program.groups;
    // (padded chunks, real rows, value columns, LogUp columns) of each lean
    // field table: the one-output selects, the cap tables, the cap reads.
    let tables = [
        (
            &shape.muxes,
            g.mux.real_rows,
            mux::cols::NUM_COLUMNS - super::layout::mux::PREP_WIDTH,
            mux::bus_interactions().len().div_ceil(2),
        ),
        (
            &shape.caps,
            g.cap.real_rows,
            cap::cols::NUM_COLUMNS - super::layout::cap::PREP_WIDTH,
            cap::bus_interactions().len().div_ceil(2),
        ),
        (
            &shape.cap_reads,
            g.cap_read.real_rows,
            cap_read::cols::NUM_COLUMNS - super::layout::cap_read::PREP_WIDTH,
            cap_read::bus_interactions().len().div_ceil(2),
        ),
    ];
    let mut cells = LeanMuxCells {
        padded_main: 0,
        padded_aux: 0,
        real_main: 0,
        real_aux: 0,
    };
    for (chunks, real, main, aux) in tables {
        let padded: u64 = chunks.iter().map(|&rows| rows as u64).sum();
        let real = if lean { real as u64 } else { 0 };
        cells.padded_main += padded * main as u64;
        cells.padded_aux += padded * aux as u64;
        cells.real_main += real * main as u64;
        cells.real_aux += real * aux as u64;
    }
    cells
}

fn publish_digest(b: &mut LfmBuilder, d: WrapDigest) {
    for cell in d.iter() {
        b.public(*cell);
    }
}

/// Instructions of each hash kind a program emits: (BLAKE3 rows of this
/// thread's lowering, every other hash row). The BLAKE3 arm emits no
/// felt-sponge row, and one lowering's rows only.
fn hash_rows(program: &LfmProgram) -> (usize, usize) {
    let lean = LEAN.with(std::cell::Cell::get);
    let ours = |i: &&Instr| match i {
        Instr::Blake3(_) => !lean,
        Instr::B3l(_) => lean,
        _ => false,
    };
    let blake3 = program.instrs.iter().filter(ours).count();
    let other = program
        .instrs
        .iter()
        .filter(|i| {
            matches!(
                i,
                Instr::Hash { .. }
                    | Instr::Hash16(_)
                    | Instr::KeccakF(_)
                    | Instr::Blake3(_)
                    | Instr::B3l(_)
            )
        })
        .count()
        - blake3;
    (blake3, other)
}

// ---------------------------------------------------------------------------
// The transcript: the four sequence scripts, replayed under BLAKE3.
// ---------------------------------------------------------------------------

fn runtime_inputs(script: &Script) -> Vec<FEE> {
    script
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::AbsorbExt { seed } => Some(ext_from(*seed)),
            _ => None,
        })
        .collect()
}

struct Emitted {
    program: LfmProgram,
    ext_count: usize,
    u64_bits: Vec<usize>,
}

fn emit_script(script: &Script) -> Emitted {
    let inputs = runtime_inputs(script);
    let mut b = builder();
    let arena = b.declare_arena(inputs.len().max(1) as u32);
    let hinted: Vec<Ext> = (0..inputs.len())
        .map(|index| b.hint_word(arena, index as u32).as_ext())
        .collect();
    let mut transcript = WhirTranscript::for_builder(&mut b);
    assert!(
        transcript.is_bytes(),
        "a BLAKE3 builder replays the byte sponge"
    );
    let mut next_input = 0usize;
    let mut sampled: Vec<Ext> = Vec::new();
    let mut drawn: Vec<Vec<Bit>> = Vec::new();
    let mut u64_bits: Vec<usize> = Vec::new();
    for step in script.steps {
        match *step {
            Step::AbsorbBytes { seed, len } => {
                transcript.absorb_const_bytes(&bytes_from(seed, len));
            }
            Step::AbsorbExt { .. } => {
                transcript.absorb_ext(&mut b, hinted[next_input]);
                next_input += 1;
            }
            Step::SampleExt => sampled.push(transcript.sample_ext(&mut b)),
            Step::SampleU64 { bound } => {
                let nbits = bound.trailing_zeros() as usize;
                assert_eq!(bound, 1u64 << nbits, "a power-of-two bound");
                u64_bits.push(nbits);
                drawn.push(transcript.sample_u64_pow2(&mut b, nbits));
            }
        }
    }
    let state = transcript.state_digest(&mut b);
    for value in &sampled {
        b.public(value.as_cell());
    }
    for bits in &drawn {
        for bit in bits {
            b.public(bit.as_cell());
        }
    }
    publish_digest(&mut b, state);
    let program = compile(b.finish());
    validate(&program).expect("the replay must be admissible");
    Emitted {
        program,
        ext_count: sampled.len(),
        u64_bits,
    }
}

fn observe(script: &Script, emitted: &Emitted) -> Observed {
    let inputs = runtime_inputs(script);
    let arena: Vec<LfmWord> = if inputs.is_empty() {
        vec![[FE::zero(); 4]]
    } else {
        inputs.iter().map(ext_word).collect()
    };
    let words = run_program(&emitted.program, &[arena]);
    let mut at = 0usize;
    let ext: Vec<FEE> = (0..emitted.ext_count)
        .map(|_| {
            let value = word_as_ext(&words[at]).expect("a published challenge");
            at += 1;
            value
        })
        .collect();
    let u64s: Vec<u64> = emitted
        .u64_bits
        .iter()
        .map(|&nbits| {
            let mut value = 0u64;
            for bit in 0..nbits {
                if words[at][0] == FE::one() {
                    value |= 1 << bit;
                }
                at += 1;
            }
            value
        })
        .collect();
    let state = digest_bytes(&words[at..at + 2]);
    Observed { ext, u64s, state }
}

/// ★ THE TRANSCRIPT GATE. The four sequence scripts (order, width and padding,
/// chain advance, sampler offset), replayed by a BLAKE3 builder's
/// [`WhirTranscript`] and by `DefaultTranscript<_, Blake3StdTranscriptHash>`:
/// every extension challenge, every bounded draw, and the state left at the end
/// must agree. The state is a function of the whole segment history, so a
/// wrong reversal, a missing re-absorb or a misplaced byte moves it.
#[test]
fn the_blake3_arm_replays_the_blake3_std_transcript() {
    under_both(|_lowering| {
        for script in super::whir_transcript_kat_tests::scripts() {
            let emitted = emit_script(script);
            let machine = observe(script, &emitted);
            let host = run_with::<Blake3StdTranscriptHash>(script);
            assert_eq!(machine.ext, host.ext, "{}: challenges", script.name);
            assert_eq!(machine.u64s, host.u64s, "{}: bounded draws", script.name);
            assert_eq!(machine.state, host.state, "{}: state", script.name);
            let (blake3, other) = hash_rows(&emitted.program);
            assert!(blake3 > 0, "{}: the arm hashes with BLAKE3", script.name);
            assert_eq!(other, 0, "{}: and with nothing else", script.name);
        }
    });
}

/// The scripts' host vectors under BLAKE3 are not RPX's: a replay that fell
/// back to the felt sponge could not pass the gate above by accident.
#[test]
fn the_blake3_and_rpx_transcripts_differ_on_every_script() {
    for script in super::whir_transcript_kat_tests::scripts() {
        let b3 = run_with::<Blake3StdTranscriptHash>(script);
        let rpx = super::whir_transcript_kats::run(script);
        assert_ne!(b3.state, rpx.state, "{}", script.name);
    }
}

/// ★ The runtime absorbs a verifier makes beyond the scripts': a 32-byte root
/// as two hinted words of `u32` lanes, an extension element, and a nonce's
/// eight big-endian bytes, each between draws, against the host's
/// `append_bytes(root)`, `append_field_element`, `append_bytes(nonce_be)`.
#[test]
fn roots_values_and_nonces_enter_as_the_host_absorbs_them() {
    under_both(|_lowering| {
        let root: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(5));
        let root2: [u8; 32] = core::array::from_fn(|i| 0xFF - (i as u8).wrapping_mul(11));
        let value = ext_from(0xABCD_EF01);
        // A nonce whose two halves differ in every byte: a swapped half or a
        // little-endian render moves the stream.
        let nonce = 0x0123_4567_89AB_CDEFu64;

        let mut host = HostTranscript::new(&[]);
        host.append_bytes(b"lead");
        host.append_bytes(&root);
        let h0 = host.sample_field_element();
        host.append_field_element(&value);
        host.append_bytes(&nonce.to_be_bytes());
        let h1 = host.sample_field_element();
        let hq = host.sample_u64(1 << 21);
        host.append_bytes(&root2);
        let h2 = host.sample_field_element();
        let hstate = host.state();

        let mut b = builder();
        let arena = b.declare_arena(6);
        let r0 = b.hint_word(arena, 0);
        let r1 = b.hint_word(arena, 1);
        let v = b.hint_word(arena, 2).as_ext();
        let n = b.hint_felt(arena, 3);
        let s0 = b.hint_word(arena, 4);
        let s1 = b.hint_word(arena, 5);
        let mut t = WhirTranscript::for_builder(&mut b);
        t.absorb_const_bytes(b"lead");
        t.absorb_root(&mut b, WrapDigest::from_pair(r0, r1));
        let m0 = t.sample_ext(&mut b);
        t.absorb_ext(&mut b, v);
        t.absorb_nonce(&mut b, n);
        let m1 = t.sample_ext(&mut b);
        let mq = t.sample_u64_pow2(&mut b, 21);
        let lanes = [b.unpack(s0), b.unpack(s1)];
        t.absorb_digest_lanes(&mut b, &lanes);
        let m2 = t.sample_ext(&mut b);
        let mstate = t.state_digest(&mut b);
        for e in [m0, m1, m2] {
            b.public(e.as_cell());
        }
        for bit in &mq {
            b.public(bit.as_cell());
        }
        publish_digest(&mut b, mstate);
        let program = compile(b.finish());
        validate(&program).expect("admissible");

        let [w0, w1] = digest_words(&root);
        let [w2, w3] = digest_words(&root2);
        let arena_words = vec![
            w0,
            w1,
            ext_word(&value),
            [FE::from(nonce), FE::zero(), FE::zero(), FE::zero()],
            w2,
            w3,
        ];
        let words = run_program(&program, &[arena_words]);
        let got: Vec<FEE> = (0..3)
            .map(|i| word_as_ext(&words[i]).expect("ext"))
            .collect();
        assert_eq!(got, vec![h0, h1, h2], "the challenges after each absorb");
        let mut q = 0u64;
        for bit in 0..21 {
            if words[3 + bit][0] == FE::one() {
                q |= 1 << bit;
            }
        }
        assert_eq!(q, hq, "the bounded draw");
        assert_eq!(digest_bytes(&words[24..26]), hstate, "the state");
    });
}

/// A root whose two hinted words are swapped is a different 32 bytes: the
/// challenge after it moves. (The words are the digest's byte order, not a set.)
#[test]
fn a_root_absorbed_with_its_words_swapped_moves_the_challenge() {
    under_both(|_lowering| {
        let root: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(29).wrapping_add(3));
        let mut host = HostTranscript::new(&[]);
        host.append_bytes(&root);
        let want = host.sample_field_element();

        let mut b = builder();
        let arena = b.declare_arena(2);
        let r0 = b.hint_word(arena, 0);
        let r1 = b.hint_word(arena, 1);
        let mut t = WhirTranscript::for_builder(&mut b);
        t.absorb_root(&mut b, WrapDigest::from_pair(r0, r1));
        let e = t.sample_ext(&mut b);
        b.public(e.as_cell());
        let program = compile(b.finish());
        let [w0, w1] = digest_words(&root);
        let honest = run_program(&program, &[vec![w0, w1]]);
        assert_eq!(word_as_ext(&honest[0]), Some(want));
        let swapped = run_program(&program, &[vec![w1, w0]]);
        assert_ne!(word_as_ext(&swapped[0]), Some(want));
    });
}

/// A constant run that ends off a 4-byte boundary (a statement's odd-length
/// field) followed by runtime values: the byte sponge has no boundary, so the
/// segment is spliced and the stream is the host's.
#[test]
fn a_runtime_value_after_an_odd_constant_run_is_the_hosts_stream() {
    under_both(|_lowering| {
        let root: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(41).wrapping_add(9));
        let value = ext_from(0x5151_7272);
        for lead in [1usize, 2, 3, 5, 7] {
            let prefix = bytes_from(0x33, lead);
            let mut host = HostTranscript::new(&[]);
            host.append_bytes(&prefix);
            host.append_bytes(&root);
            host.append_bytes(&prefix[..1]);
            host.append_field_element(&value);
            let want = host.sample_field_element();
            let want_state = host.state();

            let mut b = builder();
            let arena = b.declare_arena(3);
            let r0 = b.hint_word(arena, 0);
            let r1 = b.hint_word(arena, 1);
            let v = b.hint_word(arena, 2).as_ext();
            let mut t = WhirTranscript::for_builder(&mut b);
            t.absorb_const_bytes(&prefix);
            t.absorb_root(&mut b, WrapDigest::from_pair(r0, r1));
            t.absorb_const_bytes(&prefix[..1]);
            t.absorb_ext(&mut b, v);
            let e = t.sample_ext(&mut b);
            let st = t.state_digest(&mut b);
            b.public(e.as_cell());
            publish_digest(&mut b, st);
            let program = compile(b.finish());
            validate(&program).expect("admissible");
            let [w0, w1] = digest_words(&root);
            let out = run_program(&program, &[vec![w0, w1, ext_word(&value)]]);
            assert_eq!(
                word_as_ext(&out[0]),
                Some(want),
                "{lead} lead bytes: the draw"
            );
            assert_eq!(
                digest_bytes(&out[1..3]),
                want_state,
                "{lead} lead bytes: the state"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// The grind: `is_valid_nonce::<Blake3Std>` over the transcript state.
// ---------------------------------------------------------------------------

/// A program that absorbs `seed_felts` hinted felts, grinds at `bits` with a
/// hinted nonce, and publishes the transcript state afterwards.
fn grind_program(seed_felts: usize, bits: u8) -> LfmProgram {
    let mut b = builder();
    let arena = b.declare_arena((seed_felts + 1) as u32);
    let felts: Vec<Felt> = (0..seed_felts)
        .map(|index| b.hint_felt(arena, index as u32))
        .collect();
    let nonce = b.hint_felt(arena, seed_felts as u32);
    let mut transcript = WhirTranscript::for_builder(&mut b);
    transcript.absorb_felts(&mut b, &felts);
    emit_grind_check(&mut b, &mut transcript, bits, nonce);
    let after = transcript.state_digest(&mut b);
    publish_digest(&mut b, after);
    let program = compile(b.finish());
    validate(&program).expect("admissible");
    program
}

/// ★ THE SAMPLER'S DOUBLE MISS. A BLAKE3 coordinate draws two 64-bit
/// candidates and keeps the first below `p`; when both miss (≈ 2⁻⁶⁴, which no
/// squeeze reaches in a test) the host's field rejects the fallback and draws
/// another pair, while the machine has no execution. Driven here with hinted
/// candidates through `felt_of_candidates`, the selection `sample_felt` emits
/// for every byte-arm draw: at each pattern but both-miss the machine's felt
/// is the host's; at both-miss it has no execution and the host takes the
/// next pair's. The host is `GoldilocksField::sample_field_element_from` over
/// the fixed two-candidate schedule (`candidate_under_fixed_schedule`,
/// `pub(crate)` in `crypto` and transcribed verbatim).
#[test]
fn a_double_miss_has_no_execution_and_every_other_draw_is_the_hosts() {
    use super::transcript_replay::{Candidate, felt_of_candidates};
    use super::word::base_word;
    use crate::tables::types::GoldilocksField;
    use math::field::traits::{HasDefaultTranscript, IsPrimeField};
    const P: u64 = 0xFFFF_FFFF_0000_0001;
    fn fixed(n: usize, next: &mut impl FnMut() -> u64) -> u64 {
        let mut chosen: Option<u64> = None;
        let mut last = 0u64;
        for _ in 0..n {
            let candidate = next();
            last = candidate;
            if chosen.is_none() && GoldilocksField::candidate_in_range(candidate) {
                chosen = Some(candidate);
            }
        }
        chosen.unwrap_or(last)
    }
    // The host: the pair, then (only on a double miss) a second pair.
    let host = |pair: [u64; 2], redraw: [u64; 2]| -> (u64, usize) {
        let stream = [pair[0], pair[1], redraw[0], redraw[1]];
        let mut used = 0usize;
        let mut next = || {
            let c = stream[used];
            used += 1;
            c
        };
        let value = GoldilocksField::sample_field_element_from(|| fixed(2, &mut next));
        (GoldilocksField::canonical(value.value()), used)
    };
    let machine = |pair: [u64; 2]| -> Option<u64> {
        let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Blake3);
        let arena = b.declare_arena(4);
        let candidates: Vec<Candidate> = (0..2)
            .map(|i| Candidate {
                lo: b.hint_felt(arena, 2 * i),
                hi: b.hint_felt(arena, 2 * i + 1),
            })
            .collect();
        let felt = felt_of_candidates(&mut b, &candidates);
        b.public(felt.as_cell());
        let program = compile(b.finish());
        validate(&program).expect("admissible");
        let words: Vec<LfmWord> = pair
            .iter()
            .flat_map(|c| [*c & 0xFFFF_FFFF, *c >> 32])
            .map(|half| base_word(FE::from(half)))
            .collect();
        execute(&program, &[words], &crate::hash_pin::BLOCK_HASHER)
            .ok()
            .map(|e| GoldilocksField::canonical(e.public_words[0].1[0].value()))
    };
    let redraw = [P + 99, 42];
    for (pair, what) in [
        ([7, 9], "both in range: the first"),
        ([P + 5, 9], "the first misses: the second"),
        ([7, P + 5], "the second misses: the first"),
        ([P - 1, P], "the boundary: p - 1 is in range"),
        ([P, P - 1], "p misses: the second"),
        ([u64::MAX, 0], "the largest misses: zero"),
    ] {
        let (want, used) = host(pair, redraw);
        assert_eq!(used, 2, "{what}: the host draws one pair");
        assert_eq!(
            machine(pair),
            Some(want),
            "{what}: the machine's felt is the host's"
        );
    }
    for pair in [[P + 5, P + 11], [P, P], [u64::MAX, P]] {
        let (want, used) = host(pair, redraw);
        assert_eq!(
            (want, used),
            (42, 4),
            "{pair:x?}: the host draws a second pair and takes its first in range"
        );
        assert_eq!(
            machine(pair),
            None,
            "{pair:x?}: both candidates miss, so the machine has no execution"
        );
    }
}

/// ★ THE GRIND GATE AND ITS TAMPER ARM. At several factors the machine accepts
/// the host's own smallest valid nonce, leaves the host's transcript after
/// absorbing it, and refuses to execute on every nonce
/// `is_valid_nonce::<Blake3Std>` rejects.
#[test]
fn the_blake3_grind_accepts_what_the_host_accepts_and_refuses_the_rest() {
    under_both(|_lowering| {
        let seed_felts = 5usize;
        let values: Vec<FE> = (0..seed_felts)
            .map(|i| FE::from(0xFFFF_FFFF_0000_0000u64 - 977 * i as u64))
            .collect();
        for bits in [1u8, 7, 8, 13] {
            let mut host = HostTranscript::new(&[]);
            for value in &values {
                host.append_bytes(&value.canonical().to_be_bytes());
            }
            let seed = host.state();
            let nonce = crypto::grinding::generate_nonce_smallest::<Blake3Std>(&seed, bits)
                .expect("a nonce exists at these factors");
            assert!(crypto::grinding::is_valid_nonce::<Blake3Std>(
                &seed, nonce, bits
            ));

            let program = grind_program(seed_felts, bits);
            let (blake3, other) = hash_rows(&program);
            assert!(blake3 > 0);
            assert_eq!(other, 0, "the grind hashes with BLAKE3 alone");
            let mut arena: Vec<LfmWord> = values
                .iter()
                .map(|v| [*v, FE::zero(), FE::zero(), FE::zero()])
                .collect();
            arena.push([FE::from(nonce), FE::zero(), FE::zero(), FE::zero()]);
            let words = run_program(&program, &[arena.clone()]);
            host.append_bytes(&nonce.to_be_bytes());
            assert_eq!(
                digest_bytes(&words[..2]),
                host.state(),
                "{bits} bits: after the grind the machine's transcript is the host's"
            );

            let mut refused = 0usize;
            for candidate in 0..48u64 {
                if crypto::grinding::is_valid_nonce::<Blake3Std>(&seed, candidate, bits) {
                    continue;
                }
                let mut forged = arena.clone();
                forged[seed_felts] = [FE::from(candidate), FE::zero(), FE::zero(), FE::zero()];
                assert!(
                    execute(&program, &[forged], &crate::hash_pin::BLOCK_HASHER).is_err(),
                    "{bits} bits, nonce {candidate}: the host rejects it, so must the machine"
                );
                refused += 1;
            }
            if bits >= 7 {
                assert!(
                    refused >= 40,
                    "{bits} bits: only {refused} rejects exercised"
                );
            }
            println!("BLAKE3 grind at {bits:>2} bits: nonce {nonce} accepted, {refused} refused");
        }
    });
}

/// `bits == 0` emits nothing at all, as on the host (no state read, no
/// nonce absorb): the program equals one with no grind call.
#[test]
fn a_zero_bit_blake3_grind_touches_nothing() {
    under_both(|_lowering| {
        let with_zero = grind_program(4, 0);
        let mut b = builder();
        let arena = b.declare_arena(5);
        let felts: Vec<Felt> = (0..4).map(|index| b.hint_felt(arena, index)).collect();
        let _unused = b.hint_felt(arena, 4);
        let mut transcript = WhirTranscript::for_builder(&mut b);
        transcript.absorb_felts(&mut b, &felts);
        let state = transcript.state_digest(&mut b);
        publish_digest(&mut b, state);
        let without = compile(b.finish());
        assert_eq!(with_zero.instrs.len(), without.instrs.len());
    });
}

/// The grind's state seed is the transcript's, so an absorb the host did not
/// make before the grind refuses the host's nonce (the seed moved).
#[test]
fn a_grind_over_another_state_refuses_the_hosts_nonce() {
    under_both(|_lowering| {
        let bits = 10u8;
        let value = FE::from(0x1234_5678u64);
        let mut host = HostTranscript::new(&[]);
        host.append_bytes(&value.canonical().to_be_bytes());
        let seed = host.state();
        let nonce = crypto::grinding::generate_nonce_smallest::<Blake3Std>(&seed, bits)
            .expect("a nonce at ten bits");

        let mut b = builder();
        let arena = b.declare_arena(2);
        let felt = b.hint_felt(arena, 0);
        let n = b.hint_felt(arena, 1);
        let mut t = WhirTranscript::for_builder(&mut b);
        t.absorb_felts(&mut b, &[felt]);
        // The extra absorb: four constant bytes the host never appended (whole
        // `u32` halves, so the nonce after them stays on the byte arm's 4-byte
        // grid).
        t.absorb_const_bytes(&[0; 4]);
        emit_grind_check(&mut b, &mut t, bits, n);
        let program = compile(b.finish());
        let arena_words = vec![
            [value, FE::zero(), FE::zero(), FE::zero()],
            [FE::from(nonce), FE::zero(), FE::zero(), FE::zero()],
        ];
        // ≈ 2^-10 that the host's nonce is also valid for the other seed; this
        // seed and value are fixed, so the outcome is deterministic.
        assert!(
            execute(&program, &[arena_words], &crate::hash_pin::BLOCK_HASHER).is_err(),
            "the nonce was ground for another state"
        );
    });
}

// ---------------------------------------------------------------------------
// One chain: `whir_chain::verify` over `Blake3Whir`, emitted and executed.
// ---------------------------------------------------------------------------

mod chain {
    use multilinear::mle::Mle;
    use multilinear::whir::Domain;
    use multilinear::whir_chain::{
        CapPolicy, ChainConfig, ChainFormat, ChainProof, FirstFold, GrindBits, LeafOrder,
        RoundOpenings, WhirFolds, commit, prove, verify,
    };
    use multilinear::whir_hash::Blake3Whir;

    use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

    use super::super::builder::{Ext, LfmBuilder};
    use super::super::compiler::{LfmProgram, compile};
    use super::super::edsl::{WrapDigest, WrapHash};
    use super::super::executor::execute;
    use super::super::validator::validate;
    use super::super::whir_chain::{
        ChainShape, RoundStorage, byte_digest_words, emit_verify_weighted, push_round_words,
        round_words, try_push_round_words,
    };
    use super::super::whir_open::RootLanes;
    use super::super::whir_poly::emit_eq_eval;
    use super::super::whir_transcript::WhirTranscript;
    use super::super::word::{LfmWord, ext_word};
    use super::{HostTranscript, builder, hash_rows};

    type F = GoldilocksField;
    type E = GoldilocksExtension;

    pub(super) fn pseudo_mle(num_vars: usize, seed: u64) -> Mle<F> {
        let vals = (0..1usize << num_vars)
            .map(|i| {
                let mixed = (i as u64)
                    .wrapping_mul(0x2545_F491_4F6C_DD1D)
                    .wrapping_add(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
                // Some values near the top of the field, whose big-endian bytes
                // start 0xFF FF FF FF: a lane mix-up shows there first.
                if i % 7 == 3 {
                    FE::from(0xFFFF_FFFF_0000_0000u64 - (mixed >> 40))
                } else {
                    FE::from(mixed >> 13)
                }
            })
            .collect();
        Mle::new(vals).expect("a cube")
    }

    fn point(num_vars: usize) -> Vec<FEE> {
        (0..num_vars)
            .map(|i| {
                FEE::new([
                    FE::from(101 + i as u64),
                    FE::from(7 * i as u64 + 3),
                    FE::from(i as u64 + 1),
                ])
            })
            .collect()
    }

    /// A small binary-tree chain config: blowup 4, folds of `log_folding`, the
    /// binary cap `cap`, round-0 leaves in `leaves`' order, round 0 folding
    /// `first` when given.
    pub(super) fn config(
        num_queries: usize,
        grind: u8,
        cap: CapPolicy,
        leaves: LeafOrder,
        first: Option<usize>,
    ) -> ChainConfig {
        ChainConfig {
            log_blowup: 2,
            log_folding: 2,
            num_queries,
            grind: GrindBits::uniform(grind),
            format: ChainFormat {
                cap,
                leaves,
                folds: first.map_or(WhirFolds::Uniform, |k| {
                    WhirFolds::First(FirstFold::new(k).expect("a tested first fold"))
                }),
                ..ChainFormat::DEFAULT
            },
        }
    }

    pub(super) struct Fixture {
        cfg: ChainConfig,
        pub(super) proof: ChainProof<F, E>,
        z: Vec<FEE>,
        y: FEE,
        root: [u8; 32],
        domain: Domain<F>,
        pub(super) shape: ChainShape,
    }

    impl Fixture {
        /// This fixture's shape under the per-query check.
        fn shape_v1(&self) -> ChainShape {
            ChainShape {
                query: super::super::whir_chain::LeafQuery::V1,
                ..self.shape.clone()
            }
        }
    }

    /// One chain proved over `Blake3Whir`, accepted by the host.
    pub(super) fn fixture(num_vars: usize, cfg: ChainConfig) -> Fixture {
        let f = pseudo_mle(num_vars, 11);
        let z = point(num_vars);
        let y = f.evaluate_in::<E>(&z).expect("f takes its own point");
        let (commitment, domain) =
            commit::<F, Blake3Whir>(&f, &cfg, true).expect("the polynomial commits");
        let mut proving = HostTranscript::new(&[]);
        let proof = prove::<F, E, _, Blake3Whir>(&f, &z, &commitment, &domain, &cfg, &mut proving)
            .expect("the chain proves");
        let root = commitment.root();
        let fx = Fixture {
            cfg,
            proof,
            z,
            y,
            root,
            domain,
            shape: ChainShape::for_hash(&cfg, num_vars, WrapHash::Blake3),
        };
        assert!(
            host_accepts(&fx, &fx.proof),
            "the host accepts its own proof"
        );
        fx
    }

    pub(super) fn host_accepts(f: &Fixture, proof: &ChainProof<F, E>) -> bool {
        let mut t = HostTranscript::new(&[]);
        verify::<F, E, _, Blake3Whir>(proof, &f.root, &f.z, f.y, &f.domain, &f.cfg, &mut t).is_ok()
    }

    /// `z`, `y`, the root (two words), the final value, then the rounds.
    pub(super) fn program(shape: &ChainShape) -> LfmProgram {
        program_with(builder(), shape)
    }

    /// [`program`] on the builder `b`.
    fn program_with(mut b: LfmBuilder, shape: &ChainShape) -> LfmProgram {
        let head = (shape.num_vars + 4) as u32;
        let total = head
            + (0..shape.rounds())
                .map(|r| round_words(shape, r))
                .sum::<u32>();
        let arena = b.declare_arena(total);
        let mut transcript = WhirTranscript::for_builder(&mut b);
        let z: Vec<Ext> = (0..shape.num_vars)
            .map(|i| b.hint_word(arena, i as u32).as_ext())
            .collect();
        let n = shape.num_vars as u32;
        let y = b.hint_word(arena, n).as_ext();
        let r0 = b.hint_word(arena, n + 1);
        let r1 = b.hint_word(arena, n + 2);
        let root = RootLanes::unpack(&mut b, WrapDigest::from_pair(r0, r1));
        let final_value = b.hint_word(arena, n + 3).as_ext();
        let storage = RoundStorage::hint(&mut b, arena, head, shape);
        let (current, next) = storage.openings();
        let wires = storage.wires(&current, &next);
        let domain = Domain::<F>::new(shape.domain_log[0]).expect("the first domain");
        emit_verify_weighted(
            &mut b,
            &mut transcript,
            &wires,
            root,
            final_value,
            y,
            shape,
            &domain,
            |b, alphas| emit_eq_eval(b, &z, alphas),
        );
        let program = compile(b.finish());
        validate(&program).expect("the BLAKE3 chain leg is admissible");
        program
    }

    pub(super) fn arena(f: &Fixture, proof: &ChainProof<F, E>) -> Vec<LfmWord> {
        let mut words: Vec<LfmWord> = f.z.iter().map(ext_word).collect();
        words.push(ext_word(&f.y));
        words.extend(byte_digest_words(&f.root));
        words.push(ext_word(&proof.final_value));
        push_round_words(&mut words, &f.shape, proof);
        words
    }

    fn runs(program: &LfmProgram, words: Vec<LfmWord>) -> bool {
        execute(program, &[words], &crate::hash_pin::BLOCK_HASHER).is_ok()
    }

    /// The shapes: odd and even depths, one to three rounds, natural and coset
    /// round-0 leaves, uncapped, a fixed binary cap and the production's
    /// automatic one, grind off and on, a first fold of 6.
    pub(super) fn cases() -> Vec<(usize, ChainConfig)> {
        let mut cases = Vec::new();
        for leaves in [LeafOrder::Natural, LeafOrder::Coset] {
            for cap in [CapPolicy::Off, CapPolicy::Fixed(2), CapPolicy::Auto] {
                for (num_vars, num_queries, grind, first) in [
                    (5usize, 3usize, 0u8, None),
                    (6, 4, 0, None),
                    (9, 3, 5, None),
                    (10, 4, 6, Some(6)),
                ] {
                    cases.push((num_vars, config(num_queries, grind, cap, leaves, first)));
                }
            }
        }
        cases
    }

    /// ★ THE CHAIN GATE. The emitted BLAKE3 chain executes on every proof the
    /// host accepts: every challenge feeds a refusal, so execution IS the
    /// stream comparison. It hashes with BLAKE3 rows and nothing else.
    #[test]
    pub(super) fn a_blake3_chain_executes_on_a_proof_the_host_accepts() {
        super::under_both(|_lowering| {
            for (num_vars, cfg) in cases() {
                let f = fixture(num_vars, cfg);
                let program = program(&f.shape);
                let (blake3, other) = hash_rows(&program);
                assert!(blake3 > 0);
                assert_eq!(other, 0, "the chain hashes with BLAKE3 alone");
                execute(
                    &program,
                    &[arena(&f, &f.proof)],
                    &crate::hash_pin::BLOCK_HASHER,
                )
                .unwrap_or_else(|e| {
                    panic!(
                        "S={num_vars} {:?} caps {:?} grind {:?} leaves {:?}: refused an \
                             accepted proof: {e:?}",
                        f.shape.schedule, f.shape.caps, f.shape.grind, f.shape.leaves
                    )
                });
            }
        });
    }

    /// ★ The cost model's compressions are the emitted program's: the
    /// openings (leaf blocks, a parent a walked level, the caps), two a grind,
    /// and the byte sponge's squeezes and `state()` reads — at every shape the
    /// gate runs.
    #[test]
    pub(super) fn a_blake3_chain_emits_its_compression_closed_form() {
        super::under_both(|_lowering| {
            use super::super::whir_chain::chain_perms;
            use super::super::whir_transcript::SpongeEntry;
            for (num_vars, cfg) in cases() {
                let shape = ChainShape::for_hash(&cfg, num_vars, WrapHash::Blake3);
                let program = program(&shape);
                let (emitted, other) = hash_rows(&program);
                assert_eq!(other, 0, "one lowering's rows only");
                let predicted = chain_perms(&shape, SpongeEntry::fresh_for(WrapHash::Blake3));
                assert_eq!(
                    emitted, predicted,
                    "S={num_vars} {:?} caps {:?} grind {:?}",
                    shape.schedule, shape.caps, shape.grind
                );
            }
        });
    }

    fn sibling_flip(proof: &mut ChainProof<F, E>, round: usize, current: bool) -> bool {
        let path = match (&mut proof.rounds[round].openings, current) {
            (RoundOpenings::Base(p) | RoundOpenings::BaseJoint(p), true) => {
                &mut p.current[1].proof.merkle_path
            }
            (RoundOpenings::Base(p) | RoundOpenings::BaseJoint(p), false) => {
                match p.next.get_mut(1) {
                    Some(o) => &mut o.proof.merkle_path,
                    None => return false,
                }
            }
            (RoundOpenings::Extension(p), true) => &mut p.current[1].proof.merkle_path,
            (RoundOpenings::Extension(p), false) => match p.next.get_mut(1) {
                Some(o) => &mut o.proof.merkle_path,
                None => return false,
            },
            (RoundOpenings::BaseGrouped(_), _) => unreachable!("a per-polynomial chain"),
        };
        match path.first_mut() {
            Some(node) => {
                node[31] ^= 0x80;
                true
            }
            None => false,
        }
    }

    /// ★ THE TAMPER ARM. Every forgery the host rejects has no execution: the
    /// final value, a sumcheck evaluation, an out-of-domain value, a grind
    /// nonce, a round-0 base value, a current and a successor Merkle sibling
    /// (its LAST byte, so a digest compared on one word only would miss it),
    /// a cap node, and the root itself.
    #[test]
    pub(super) fn a_tampered_blake3_chain_cannot_execute() {
        super::under_both(|_lowering| {
            for leaves in [LeafOrder::Natural, LeafOrder::Coset] {
                for cap in [CapPolicy::Off, CapPolicy::Fixed(2)] {
                    let f = fixture(10, config(4, 6, cap, leaves, Some(6)));
                    let program = program(&f.shape);
                    assert!(
                        runs(&program, arena(&f, &f.proof)),
                        "the untouched proof must execute, or the arm below proves nothing"
                    );
                    let mut sites: Vec<(&str, ChainProof<F, E>)> = Vec::new();
                    let mut forged = f.proof.clone();
                    forged.final_value += FEE::one();
                    sites.push(("the final value", forged));
                    let mut forged = f.proof.clone();
                    forged.rounds[0].sumcheck[0].evaluations[1] += FEE::one();
                    sites.push(("a sumcheck evaluation", forged));
                    let mut forged = f.proof.clone();
                    if let Some(v) = forged.rounds[0].ood_value.as_mut() {
                        *v += FEE::one();
                    }
                    sites.push(("an out-of-domain value", forged));
                    let mut forged = f.proof.clone();
                    forged.rounds[0].nonces.query ^= 1;
                    sites.push(("a grind nonce", forged));
                    let mut forged = f.proof.clone();
                    match &mut forged.rounds[0].openings {
                        RoundOpenings::Base(p) => p.current[1].values[5] += FE::one(),
                        _ => unreachable!("round 0 opens base blocks"),
                    }
                    sites.push(("a round-0 base value", forged));
                    let mut forged = f.proof.clone();
                    assert!(sibling_flip(&mut forged, 0, true));
                    sites.push(("a current Merkle sibling", forged));
                    let mut forged = f.proof.clone();
                    if (0..forged.rounds.len()).any(|r| sibling_flip(&mut forged, r, false)) {
                        sites.push(("a successor Merkle sibling", forged));
                    }
                    if cap != CapPolicy::Off {
                        let mut forged = f.proof.clone();
                        match &mut forged.rounds[0].openings {
                            RoundOpenings::Base(p) => {
                                let path = &mut p.current[0].proof.merkle_path;
                                let last = path.len() - 1;
                                path[last][31] ^= 1;
                            }
                            _ => unreachable!("round 0 opens base blocks"),
                        }
                        sites.push(("a cap node", forged));
                    }
                    for (name, forged) in &sites {
                        assert!(
                            !host_accepts(&f, forged),
                            "{name} ({cap:?}, {leaves:?}): the host must reject it"
                        );
                        assert!(
                            !runs(&program, arena(&f, forged)),
                            "{name} ({cap:?}, {leaves:?}): the machine must refuse it"
                        );
                    }
                    // A wrong root: every opening of round 0 fails against it.
                    let mut words = arena(&f, &f.proof);
                    let n = f.shape.num_vars;
                    words[n + 2][3] += FE::one();
                    assert!(
                        !runs(&program, words),
                        "a root off in its second word must be refused"
                    );
                }
            }
        });
    }

    /// ★ Under the lean query check a BLAKE3 chain's tail rounds — a tree
    /// wholly in its binary cap with [`TAIL_QUERIES_A_LEAF`] queries a leaf —
    /// are checked once a leaf, as Poseidon1's 4-ary tail is: the honest proof
    /// executes, the compressions are the closed form's (fewer than the
    /// per-query check's), a tail block's value off has no execution, and the
    /// tail's compare reading the neighbouring slot refuses the honest proof.
    ///
    /// [`TAIL_QUERIES_A_LEAF`]: super::super::whir_chain::TAIL_QUERIES_A_LEAF
    #[test]
    pub(super) fn a_lean_blake3_chain_checks_its_tail_once_a_leaf() {
        use super::super::whir_chain::{LeafQuery, chain_perms, tail_faults};
        use super::super::whir_transcript::SpongeEntry;
        super::under_both(|_lowering| {
            // Folds of one after the first, so three small trees lie wholly in
            // their caps at 400 queries: three tail rounds, two of them with a
            // tail successor.
            let mut cfg = config(400, 0, CapPolicy::Fixed(4), LeafOrder::Coset, Some(6));
            cfg.log_folding = 1;
            let f = fixture(9, cfg);
            let lean = ChainShape {
                query: LeafQuery::Lean,
                ..f.shape.clone()
            };
            let tails: Vec<usize> = (0..lean.rounds()).filter(|&r| lean.tail_round(r)).collect();
            assert!(
                tails.len() >= 2,
                "a tail round at {:?} caps {:?}",
                lean.schedule,
                lean.caps
            );
            let lean_fx = Fixture {
                shape: lean.clone(),
                ..f
            };
            let leg = program(&lean);
            let words = arena(&lean_fx, &lean_fx.proof);
            assert!(runs(&leg, words.clone()), "the honest proof executes");
            let (emitted, other) = hash_rows(&leg);
            assert_eq!(other, 0);
            let predicted = chain_perms(&lean, SpongeEntry::fresh_for(WrapHash::Blake3));
            assert_eq!(emitted, predicted, "the closed form, tail and all");
            let v1 = program(&lean_fx.shape_v1());
            assert!(
                hash_rows(&v1).0 > emitted,
                "the tail saves compressions: {} against {emitted}",
                hash_rows(&v1).0
            );
            // A tail block's value: the arena's last words are the last tail
            // round's blocks, so the last word is one of its values.
            let mut forged = words.clone();
            let last = forged.len() - 1;
            forged[last][0] += FE::one();
            assert!(!runs(&leg, forged), "a tail value off");
            let slot_off = tail_faults::with_slot_off(|| program(&lean));
            assert!(
                !runs(&slot_off, words),
                "the tail's compare reads the slot its position names"
            );
            println!(
                "BLAKE3 TAIL: rounds {tails:?} of {:?} · {emitted} compressions (per query {})",
                lean.schedule,
                hash_rows(&v1).0
            );
        });
    }

    /// ★ THE FINAL-POLYNOMIAL GATE. A BLAKE3 chain under `FinalPoly::Vars`
    /// (the final round commits no successor, sends its folded polynomial, and
    /// the rounds after it are sumchecks alone) executes on every proof the
    /// host accepts, at three final rounds; its compressions are the closed
    /// form's and fewer than the same chain's without the final polynomial;
    /// and a sent value, a tail sumcheck evaluation, the final value and a
    /// final-round block value each tampered are refused by the host and have
    /// no execution.
    #[test]
    pub(super) fn a_blake3_chain_with_a_final_polynomial_executes_and_refuses_its_tampers() {
        use super::super::whir_chain::chain_perms;
        use super::super::whir_transcript::SpongeEntry;
        use multilinear::whir_chain::FinalPoly;
        super::under_both(|_lowering| {
            for (num_vars, v) in [(10usize, 2u8), (12, 4), (12, 2)] {
                let mut cfg = config(5, 4, CapPolicy::Fixed(2), LeafOrder::Coset, Some(6));
                cfg.format.final_poly = FinalPoly::vars(v).expect("a final width");
                let (final_r, rem) = cfg.final_round(num_vars).expect("a final round");
                let f = fixture(num_vars, cfg);
                assert_eq!(f.shape.final_at, Some((final_r, rem)));
                let leg = program(&f.shape);
                execute(&leg, &[arena(&f, &f.proof)], &crate::hash_pin::BLOCK_HASHER)
                    .unwrap_or_else(|e| {
                        panic!("S={num_vars} v={v}: refused an accepted proof: {e:?}")
                    });
                let (emitted, other) = hash_rows(&leg);
                assert_eq!(other, 0);
                assert_eq!(
                    emitted,
                    chain_perms(&f.shape, SpongeEntry::fresh_for(WrapHash::Blake3)),
                    "S={num_vars} v={v}: the closed form"
                );
                let mut plain_cfg = cfg;
                plain_cfg.format.final_poly = FinalPoly::Off;
                let plain = ChainShape::for_hash(&plain_cfg, num_vars, WrapHash::Blake3);
                assert!(
                    emitted < chain_perms(&plain, SpongeEntry::fresh_for(WrapHash::Blake3)),
                    "the final polynomial saves compressions"
                );
                let mut sites: Vec<(&str, ChainProof<F, E>)> = Vec::new();
                let mut forged = f.proof.clone();
                match &mut forged.rounds[final_r].openings {
                    RoundOpenings::Extension(p) => p.next[0].values[1] += FEE::one(),
                    _ => unreachable!("the final round is extension-valued"),
                }
                sites.push(("a sent value", forged));
                let mut forged = f.proof.clone();
                forged.final_value += FEE::one();
                sites.push(("the final value", forged));
                let mut forged = f.proof.clone();
                match &mut forged.rounds[final_r].openings {
                    RoundOpenings::Extension(p) => p.current[0].values[0] += FEE::one(),
                    _ => unreachable!("the final round is extension-valued"),
                }
                sites.push(("a final-round block value", forged));
                if final_r + 1 < f.proof.rounds.len() {
                    let mut forged = f.proof.clone();
                    forged.rounds[final_r + 1].sumcheck[0].evaluations[0] += FEE::one();
                    sites.push(("a tail sumcheck evaluation", forged));
                }
                // A sent polynomial one value short or long: refused by the
                // arena's layout before the leaf runs.
                for grow in [false, true] {
                    let mut forged = f.proof.clone();
                    match &mut forged.rounds[final_r].openings {
                        RoundOpenings::Extension(p) if grow => p.next[0].values.push(FEE::one()),
                        RoundOpenings::Extension(p) => {
                            p.next[0].values.pop();
                        }
                        _ => unreachable!("the final round is extension-valued"),
                    }
                    let mut words = Vec::new();
                    assert!(
                        try_push_round_words(&mut words, &f.shape, &forged).is_err(),
                        "S={num_vars} v={v}: a sent polynomial of the wrong width (grow {grow})"
                    );
                    sites.push(("a sent polynomial of the wrong width", forged));
                }
                for (name, forged) in &sites {
                    assert!(
                        !host_accepts(&f, forged),
                        "S={num_vars} v={v}: {name}: host"
                    );
                    assert!(
                        !runs(&leg, arena(&f, forged)),
                        "S={num_vars} v={v}: {name} must have no execution"
                    );
                }
                println!(
                    "BLAKE3 FINAL POLY S={num_vars} v={v}: final round {final_r} ({rem} vars) of {:?} \
                     · {emitted} compressions",
                    f.shape.schedule
                );
            }
        });
    }

    /// A final polynomial is emitted under BLAKE3 alone: the same shape on a
    /// Keccak builder (two-word digests, binary trees too) is refused at
    /// emission, so no other hash's final round runs ungated.
    #[test]
    #[should_panic(expected = "the BLAKE3 base's binary trees only")]
    fn a_final_polynomial_under_another_hash_is_refused_at_emission() {
        use multilinear::whir_chain::FinalPoly;
        let mut cfg = config(5, 4, CapPolicy::Fixed(2), LeafOrder::Coset, Some(6));
        cfg.format.final_poly = FinalPoly::vars(2).expect("a final width");
        let shape = ChainShape::for_hash(&cfg, 10, WrapHash::Keccak);
        assert!(shape.final_at.is_some());
        program_with(LfmBuilder::new().with_wrap_hash(WrapHash::Keccak), &shape);
    }

    /// The cap height on one production chain under the BLAKE3 base's format
    /// (a single 2^27 polynomial, the final polynomial on, the lean query
    /// check and lowering), priced in CELLS: compressions at the lean tables'
    /// 2,832 cells each, plus the field chips' real rows times their value
    /// columns (and with their LogUp columns at three felts each). The cap's
    /// mux is `2^c − 1` `Select` rows a digest word an opening, so the field
    /// side grows with the cap while the compressions shrink. Emission only.
    /// Prints; asserts that the forms agree with the emission.
    #[test]
    #[ignore = "census: emits six production-shape chain programs"]
    fn the_production_chain_cells_by_cap() {
        use super::super::whir_chain::{ChipRows, LeafQuery, chain_perms};
        use super::super::whir_transcript::SpongeEntry;
        use crate::block_whir::BlockFormat;
        use crate::hash_pin::{BlockFamily, ProofFamily};
        use crate::zf_format::ZfFormat;
        use stark::proof::options::BaseFormat;
        super::LEAN.with(|c| c.set(true));
        let zf = ZfFormat::DEFAULT.with_base(BaseFormat::B3_WHIR);
        let format = BlockFormat {
            zf,
            ..BlockFormat::production()
        };
        let n = zf.whir_stack.get();
        let mut config = format.chain_config(&[(1, n)]);
        config.format.group = multilinear::whir_chain::GroupFold::PerPoly;
        config.format.round0 = multilinear::whir_chain::Round0Trees::Each;
        use super::super::merkle_cap::{CAP_MUX_OVERRIDE, CapMux};
        for (packed, cap) in [CapMux::Plain, CapMux::Packed, CapMux::Indexed]
            .into_iter()
            .flat_map(|p| (3u8..=8).map(move |c| (p, c)))
        {
            CAP_MUX_OVERRIDE.with(|c| c.set(Some(packed)));
            let mut c = config;
            c.format.cap = CapPolicy::Fixed(cap);
            let shape = ChainShape {
                query: LeafQuery::Lean,
                ..ChainShape::for_hash(&c, n, WrapHash::Blake3)
            };
            let leg = program(&shape);
            let (compressions, other) = hash_rows(&leg);
            assert_eq!(other, 0);
            assert_eq!(
                compressions,
                chain_perms(&shape, SpongeEntry::fresh_for(WrapHash::Blake3))
            );
            let chips = super::super::airs::lfm_chip_census_with_hasher(
                &leg,
                leg.hasher(<BlockFamily as ProofFamily>::SOCKET),
            );
            // The lean program's one-output selects are LEAN_MUX rows, outside
            // the chip census: counted with the field side and the selects.
            let muxes = super::lean_mux_cells(&leg);
            let field: u64 = chips
                .iter()
                .map(|c| c.real_rows * c.main_cols as u64)
                .sum::<u64>()
                + muxes.real_main;
            let field_aux: u64 = chips
                .iter()
                .map(|c| c.real_rows * (c.main_cols + 3 * c.aux_cols) as u64)
                .sum::<u64>()
                + muxes.real_main
                + 3 * muxes.real_aux;
            let select: u64 = chips
                .iter()
                .filter(|c| c.name.contains("SELECT"))
                .map(|c| c.real_rows * c.main_cols as u64)
                .sum::<u64>()
                + muxes.real_main;
            let hash = compressions as u64 * 2_832;
            println!(
                "CAP CELLS {packed:?} c={cap}: caps {:?} · {compressions} compressions ({hash} lean cells) · field main \
                 {field} (select {select}; with LogUp {field_aux}) · total {} (with LogUp {}) · rows {:?}",
                shape.caps,
                hash + field,
                hash + field_aux,
                ChipRows::of(&leg)
            );
        }
        CAP_MUX_OVERRIDE.with(|c| c.set(None));
        super::LEAN.with(|c| c.set(false));
    }

    /// The BLAKE3 base's final polynomial on one production chain (a single
    /// 2^27 polynomial at the base's format, under the leaf's lean query
    /// check), emitted with and without it: the instructions and compressions
    /// each program takes. Emission only (no proof: the shape alone fixes the
    /// program). Prints; asserts that the final polynomial saves compressions.
    #[test]
    #[ignore = "census: emits two production-shape chain programs"]
    fn the_production_final_round_costs_its_instructions() {
        use super::super::whir_chain::{LeafQuery, chain_perms};
        use super::super::whir_transcript::SpongeEntry;
        use crate::block_whir::BlockFormat;
        use crate::zf_format::ZfFormat;
        use multilinear::whir_chain::FinalPoly;
        use stark::proof::options::BaseFormat;
        super::under_both(|lowering| {
            let zf = ZfFormat::DEFAULT.with_base(BaseFormat::B3_WHIR);
            let format = BlockFormat {
                zf,
                ..BlockFormat::production()
            };
            let n = zf.whir_stack.get();
            let mut config = format.chain_config(&[(1, n)]);
            // One polynomial, its own chain: the rounds the final polynomial
            // moves (3 and after) are a run's whatever its width.
            config.format.group = multilinear::whir_chain::GroupFold::PerPoly;
            config.format.round0 = multilinear::whir_chain::Round0Trees::Each;
            let mut counts = Vec::new();
            for final_poly in [config.format.final_poly, FinalPoly::Off] {
                let mut c = config;
                c.format.final_poly = final_poly;
                let shape = ChainShape {
                    query: LeafQuery::Lean,
                    ..ChainShape::for_hash(&c, n, WrapHash::Blake3)
                };
                let leg = program(&shape);
                let (compressions, other) = hash_rows(&leg);
                assert_eq!(other, 0);
                assert_eq!(
                    compressions,
                    chain_perms(&shape, SpongeEntry::fresh_for(WrapHash::Blake3))
                );
                println!(
                    "FINAL ROUND CENSUS {lowering} {final_poly:?}: final {:?} · {} instructions \
                     · {compressions} compressions",
                    shape.final_at,
                    leg.instrs.len()
                );
                counts.push((leg.instrs.len(), compressions));
            }
            let (with, without) = (counts[0], counts[1]);
            println!(
                "FINAL ROUND CENSUS {lowering}: {:+} instructions · {:+} compressions",
                with.0 as i64 - without.0 as i64,
                with.1 as i64 - without.1 as i64
            );
            assert!(
                with.1 < without.1,
                "the final polynomial saves compressions"
            );
        });
    }

    /// A round-0 value hinted with a nonzero upper lane is refused before any
    /// hash: the leaf's `BitDec` reads it as a base word, so the lift into the
    /// fold is pinned (whir_open's obligation, under BLAKE3).
    #[test]
    fn a_round0_value_with_an_upper_lane_is_refused() {
        super::under_both(|_lowering| {
            let f = fixture(6, config(3, 0, CapPolicy::Off, LeafOrder::Natural, None));
            let program = program(&f.shape);
            let words = arena(&f, &f.proof);
            // Round 0's first block value: after the head, the round's sumcheck
            // words (two a message) and its carried nonces.
            let r0 = f.shape.num_vars + 4;
            let nonces = f.shape.carries(0).iter().filter(|&&c| c).count();
            let at = r0 + 2 * f.shape.messages(0) + nonces;
            assert_eq!(
                words[at][1..],
                [FE::zero(); 3],
                "the slot is a base value's word"
            );
            let mut forged = words.clone();
            forged[at][2] = FE::one();
            let err = execute(&program, &[forged], &crate::hash_pin::BLOCK_HASHER)
                .expect_err("a lifted lane must be refused");
            assert!(
                matches!(err, super::super::executor::LfmExecError::NotBaseWord(_)),
                "refused as a non-base word, got {err:?}"
            );
        });
    }

    /// The coset program and the natural one refuse each other's proofs (and
    /// so does the host): round 0's walks are wired to the order.
    #[test]
    fn a_coset_blake3_program_refuses_a_natural_proof() {
        super::under_both(|_lowering| {
            let coset = config(9, 0, CapPolicy::Off, LeafOrder::Coset, Some(6));
            let natural = config(9, 0, CapPolicy::Off, LeafOrder::Natural, Some(6));
            let made_coset = fixture(11, coset);
            let made_natural = fixture(11, natural);
            let coset_program = program(&made_coset.shape);
            let natural_program = program(&made_natural.shape);
            assert!(!runs(
                &coset_program,
                arena(&made_natural, &made_natural.proof)
            ));
            assert!(!runs(
                &natural_program,
                arena(&made_coset, &made_coset.proof)
            ));
        });
    }
}

// ---------------------------------------------------------------------------
// The hashes one at a time: leaf, joint leaf, parent, against the host.
// ---------------------------------------------------------------------------

mod kats {
    use crypto::merkle_tree::backends::blake3_joint::joint_leaf;
    use crypto::merkle_tree::backends::types::BatchBlake3StdBackend;
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;

    use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

    use super::super::builder::Felt;
    use super::super::compiler::compile;
    use super::super::edsl::{self, WrapDigest};
    use super::super::whir_chain::byte_digest_words;
    use super::super::whir_open::{BlockValues, emit_blake3_joint_leaf, emit_block_leaf};
    use super::super::word::{LfmWord, ext_word};
    use super::{builder, digest_bytes, publish_digest, run_program};

    fn felts(n: usize, seed: u64) -> Vec<FE> {
        (0..n as u64)
            .map(|i| {
                if i % 5 == 2 {
                    FE::from(0xFFFF_FFFF_0000_0000u64 - i - seed)
                } else {
                    FE::from(
                        (i + seed)
                            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                            .wrapping_add(13),
                    )
                }
            })
            .collect()
    }

    /// One digest out of a program that hints `words` and emits `emit` over
    /// them.
    fn digest_of(
        words: Vec<LfmWord>,
        emit: impl FnOnce(
            &mut super::super::builder::LfmBuilder,
            &[super::super::builder::Cell],
        ) -> WrapDigest,
    ) -> [u8; 32] {
        let mut b = builder();
        let arena = b.declare_arena(words.len() as u32);
        let cells: Vec<_> = (0..words.len())
            .map(|i| b.hint_word(arena, i as u32))
            .collect();
        let d = emit(&mut b, &cells);
        publish_digest(&mut b, d);
        let program = compile(b.finish());
        let out = run_program(&program, &[words]);
        digest_bytes(&out)
    }

    /// ★ The per-polynomial leaf is `BatchBlake3StdBackend::hash_data`, for a
    /// base block of every width the chains open (2 to 64 values, and an odd
    /// 3 for a partial block) and an extension block (3 felts a value, the
    /// two-wide tail's 48 bytes in one partial block).
    #[test]
    fn the_leaf_is_the_hosts_blake3_std_leaf() {
        super::under_both(|_lowering| {
            for n in [2usize, 3, 8, 16, 21, 64] {
                let values = felts(n, n as u64);
                let want =
                    <BatchBlake3StdBackend<GoldilocksField> as IsMerkleTreeBackend>::hash_data(
                        &values,
                    );
                let got = digest_of(
                    values
                        .iter()
                        .map(|v| [*v, FE::zero(), FE::zero(), FE::zero()])
                        .collect(),
                    |b, cells| {
                        let fs: Vec<Felt> = cells.iter().map(|c| Felt(c.addr())).collect();
                        emit_block_leaf(b, BlockValues::Base(&fs))
                    },
                );
                assert_eq!(got, want, "a base block of {n} values");
            }
            for n in [1usize, 2, 4, 16] {
                let values: Vec<FEE> = (0..n)
                    .map(|i| {
                        let f = felts(3, 100 + i as u64);
                        FEE::new([f[0], f[1], f[2]])
                    })
                    .collect();
                let want =
                    <BatchBlake3StdBackend<GoldilocksExtension> as IsMerkleTreeBackend>::hash_data(
                        &values,
                    );
                let got = digest_of(values.iter().map(ext_word).collect(), |b, cells| {
                    let es: Vec<_> = cells.iter().map(|c| c.as_ext()).collect();
                    emit_block_leaf(b, BlockValues::Ext(&es))
                });
                assert_eq!(got, want, "an extension block of {n} values");
            }
        });
    }

    /// ★ The joint leaf, resumed from the prefix's midstate, is
    /// `blake3_joint::joint_leaf` at every run width (2, 3, 4) and at the
    /// block sizes the tests and the production fold use (8, 16, 64 values),
    /// and equals the whole chain over `prefix ‖ values` from the IV.
    #[test]
    fn the_joint_leaf_from_its_midstate_is_the_hosts() {
        super::under_both(|_lowering| {
            use crypto::merkle_tree::backends::blake3_joint::{
                JOINT_LEAF_PREFIX_LEN, joint_leaf_prefix,
            };
            for blocks in [2usize, 3, 4] {
                for per_block in [8usize, 16, 64] {
                    let values = felts(blocks * per_block, (blocks * 100 + per_block) as u64);
                    let want = joint_leaf(&values, blocks).expect("an even split");
                    let words: Vec<LfmWord> = values
                        .iter()
                        .map(|v| [*v, FE::zero(), FE::zero(), FE::zero()])
                        .collect();
                    let got = digest_of(words.clone(), |b, cells| {
                        let fs: Vec<Felt> = cells.iter().map(|c| Felt(c.addr())).collect();
                        emit_blake3_joint_leaf(b, &fs, blocks)
                    });
                    assert_eq!(got, want, "{blocks} blocks of {per_block}");
                    // The same message chained from the IV, prefix as constants.
                    let prefix = joint_leaf_prefix(blocks as u64, per_block as u64, 8);
                    let whole = digest_of(words, |b, cells| {
                        let fs: Vec<Felt> = cells.iter().map(|c| Felt(c.addr())).collect();
                        let mut t = super::super::transcript_replay::ByteString::new();
                        t.push_const(&prefix);
                        let (stream, len) = edsl::leaf_stream(b, &fs);
                        assert_eq!(len + JOINT_LEAF_PREFIX_LEN, prefix.len() + 8 * fs.len());
                        t.push_halves(&stream);
                        t.wrap_hash(b, edsl::ByteWrapHash::Blake3)
                    });
                    assert_eq!(whole, want, "{blocks} blocks of {per_block}, from the IV");
                }
            }
        });
    }

    /// The joint leaf binds its block count through the midstate: the same
    /// values hashed as another run width give another digest, and only the
    /// shape's own count matches the host's.
    #[test]
    fn the_joint_leaf_binds_its_block_count() {
        super::under_both(|_lowering| {
            let values = felts(4 * 16, 7);
            let words: Vec<LfmWord> = values
                .iter()
                .map(|v| [*v, FE::zero(), FE::zero(), FE::zero()])
                .collect();
            let want = joint_leaf(&values, 4).expect("an even split");
            for blocks in [2usize, 4] {
                let got = digest_of(words.clone(), |b, cells| {
                    let fs: Vec<Felt> = cells.iter().map(|c| Felt(c.addr())).collect();
                    emit_blake3_joint_leaf(b, &fs, blocks)
                });
                assert_eq!(got == want, blocks == 4, "{blocks} blocks");
            }
        });
    }

    /// ★ The final polynomial's two values, against the host's: its encoding
    /// at a power point (`Mle::evaluate` at `ood_point(y, n)`, what
    /// `verify_final_against` compares a fold with) through the Möbius
    /// coefficients and Horner, and its value at a point (`Mle::evaluate_at`,
    /// the tail's `f(α)`), at every width the gate's chains send and wider.
    #[test]
    fn the_final_polynomials_encoding_and_value_are_the_hosts() {
        use super::super::whir_chain::{emit_mle_eval, emit_power_coefficients};
        use crate::tables::types::GoldilocksExtension;
        use multilinear::mle::Mle;
        for n in 1usize..=5 {
            let values: Vec<FEE> = (0..1u64 << n)
                .map(|i| {
                    let f = felts(3, 31 * i + n as u64);
                    FEE::new([f[0], f[1], f[2]])
                })
                .collect();
            let y = FE::from(0x1234_5678_9abc_u64 + n as u64);
            let point: Vec<FEE> = (0..n)
                .map(|j| {
                    let f = felts(3, 900 + j as u64);
                    FEE::new([f[0], f[1], f[2]])
                })
                .collect();
            let host = Mle::<GoldilocksExtension>::new(values.clone()).expect("a cube");
            let want_encoding = host
                .evaluate(&multilinear::whir_chain::ood_point(&y.to_extension(), n))
                .expect("the encoding");
            let want_value = host.evaluate(&point).expect("the value");

            let mut b = builder();
            let arena = b.declare_arena((values.len() + n + 1) as u32);
            let vs: Vec<_> = (0..values.len())
                .map(|i| b.hint_word(arena, i as u32).as_ext())
                .collect();
            let ps: Vec<_> = (0..n)
                .map(|j| b.hint_word(arena, (values.len() + j) as u32).as_ext())
                .collect();
            let yc = b.hint_felt(arena, (values.len() + n) as u32);
            let coefficients = emit_power_coefficients(&mut b, &vs, n);
            let encoding = super::super::edsl::horner_ext(&mut b, yc.as_ext(), &coefficients);
            let value = emit_mle_eval(&mut b, &vs, &ps);
            b.public(encoding.as_cell());
            b.public(value.as_cell());
            let program = compile(b.finish());
            let mut words: Vec<LfmWord> = values.iter().map(ext_word).collect();
            words.extend(point.iter().map(ext_word));
            words.push([y, FE::zero(), FE::zero(), FE::zero()]);
            let out = run_program(&program, &[words]);
            assert_eq!(
                super::super::word::word_as_ext(&out[0]),
                Some(want_encoding),
                "{n}: encoding"
            );
            assert_eq!(
                super::super::word::word_as_ext(&out[1]),
                Some(want_value),
                "{n}: value"
            );
        }
    }

    /// ★ A parent is `hash_new_parent(left, right)`: one compression of the
    /// two digests' 64 bytes, in order.
    #[test]
    fn the_parent_is_the_hosts() {
        super::under_both(|_lowering| {
            let left: [u8; 32] = core::array::from_fn(|i| (i as u8).wrapping_mul(13) ^ 0x5A);
            let right: [u8; 32] = core::array::from_fn(|i| 0xC3 ^ (i as u8).wrapping_mul(7));
            let want =
                <BatchBlake3StdBackend<GoldilocksField> as IsMerkleTreeBackend>::hash_new_parent(
                    &left, &right,
                );
            let mut words: Vec<LfmWord> = byte_digest_words(&left).to_vec();
            words.extend(byte_digest_words(&right));
            let got = digest_of(words, |b, cells| {
                edsl::wrap_hash_pair(
                    b,
                    WrapDigest::from_pair(cells[0], cells[1]),
                    WrapDigest::from_pair(cells[2], cells[3]),
                )
            });
            assert_eq!(got, want);
        });
    }
}

// ---------------------------------------------------------------------------
// A group's stacked opening: `stacked_eval::verify` over `Blake3Whir`, one
// chain a run (`GroupFold::Indexed`), a tree a polynomial or one a run.
// ---------------------------------------------------------------------------

mod stacked {
    use multilinear::mle::Mle;
    use multilinear::stacked_eval::{Claimed, StackedCommitment, StackedProof};
    use multilinear::stacking::StackedLayout;
    use multilinear::whir::Domain;
    use multilinear::whir_chain::{
        CapPolicy, ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, LeafOrder,
        Round0Trees, RoundOpenings, WhirFolds,
    };
    use multilinear::whir_commit::Commitment;
    use multilinear::whir_hash::Blake3Whir;

    use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField};

    use super::super::builder::{Cell, Ext};
    use super::super::compiler::{LfmProgram, compile};
    use super::super::edsl::WrapHash;
    use super::super::executor::execute;
    use super::super::validator::validate;
    use super::super::whir_chain::{ChainShape, RoundStorage, byte_digest_words, push_round_words};
    use super::super::whir_stacked::{OodWires, StackedChainWires, emit_stacked_verify_chains};
    use super::super::whir_transcript::WhirTranscript;
    use super::super::word::{LfmWord, ext_word, word_as_ext};
    use super::{HostTranscript, builder, hash_rows};

    type F = GoldilocksField;
    type E = GoldilocksExtension;

    #[derive(Clone)]
    pub(super) struct Case {
        heights: Vec<usize>,
        n_stack: usize,
        first: Option<usize>,
        cap: CapPolicy,
        leaves: LeafOrder,
        round0: Round0Trees,
        grind: u8,
        /// The WHIR round form at `(shrink, security bits)`, its counts
        /// sized for the bits; `None` is today's form at four queries.
        whir: Option<(usize, u8)>,
    }

    fn config(case: &Case) -> ChainConfig {
        if let Some((shrink, bits)) = case.whir {
            let mut cfg = ChainConfig::with_security_folds(
                2,
                4,
                case.first.map_or(WhirFolds::Uniform, |k| {
                    WhirFolds::First(FirstFold::new(k).expect("a tested first fold"))
                }),
                case.n_stack,
                bits,
                GrindBits::uniform(case.grind),
            );
            cfg.format.group = GroupFold::Indexed;
            cfg.format.leaves = case.leaves;
            cfg.format.round0 = case.round0;
            cfg.format.cap = case.cap;
            return cfg
                .with_whir_rounds(shrink, bits, case.n_stack)
                .expect("a tested shrink");
        }
        ChainConfig {
            log_blowup: 2,
            log_folding: 4,
            num_queries: 4,
            grind: GrindBits::uniform(case.grind),
            format: ChainFormat {
                folds: case.first.map_or(WhirFolds::Uniform, |k| {
                    WhirFolds::First(FirstFold::new(k).expect("a tested first fold"))
                }),
                group: GroupFold::Indexed,
                leaves: case.leaves,
                round0: case.round0,
                cap: case.cap,
                ..ChainFormat::DEFAULT
            },
        }
    }

    fn fee(v: u64) -> FEE {
        FEE::new([
            FE::from(v.wrapping_mul(6364136223846793005) >> 11),
            FE::from(v ^ 0xA5A5),
            FE::from(v.wrapping_add(7)),
        ])
    }

    pub(super) struct Fixture {
        cfg: ChainConfig,
        layout: StackedLayout,
        points: Vec<Vec<FEE>>,
        values: Vec<FEE>,
        roots: Vec<Commitment>,
        proof: StackedProof<F, E>,
        domain: Domain<F>,
        shape: ChainShape,
        /// The commit-time answers the opening carries, at `(z0, answers)`:
        /// `None` for an opening without them.
        ood: Option<(FEE, Vec<FEE>)>,
    }

    pub(super) fn build(case: &Case) -> Fixture {
        build_with(case, None)
    }

    /// [`build`], the opening carrying commit-time answers at a fixed `z0`
    /// when `answer_delta` is given: polynomial 0's answer moved by it, then
    /// stated and proved as given, so a nonzero delta is a false answer the
    /// prover's own chain carries.
    pub(super) fn build_with(case: &Case, answer_delta: Option<FEE>) -> Fixture {
        let cfg = config(case);
        let layout = StackedLayout::build(&case.heights, case.n_stack).expect("the layout packs");
        let columns: Vec<Mle<F>> = case
            .heights
            .iter()
            .enumerate()
            .map(|(c, &h)| super::chain::pseudo_mle(h, 17 + 13 * c as u64))
            .collect();
        let borrowed = multilinear::stacking::borrow(&columns);
        let points: Vec<Vec<FEE>> = case
            .heights
            .iter()
            .enumerate()
            .map(|(c, &h)| (0..h).map(|i| fee(0x5eed + (c * 37 + i) as u64)).collect())
            .collect();
        let values: Vec<FEE> = columns
            .iter()
            .zip(&points)
            .map(|(c, p)| c.evaluate_in::<E>(p).expect("a column takes its own point"))
            .collect();
        let stacked =
            StackedCommitment::<F, Blake3Whir>::commit(layout.clone(), &borrowed, None, &cfg)
                .expect("the group commits");
        let ood = answer_delta.map(|delta| {
            let z0 = fee(0x0bad_5eed);
            let (mut claims, _) =
                multilinear::stacked_eval::ood_claims::<F, E, _>(&layout, &borrowed, None, &z0)
                    .expect("the answers");
            claims.answers[0] += delta;
            claims
        });
        let proof = match &ood {
            None => multilinear::stacked_eval::prove::<F, E, _, Blake3Whir, _>(
                &stacked,
                &borrowed,
                None,
                &Claimed::PerColumn(&points),
                &values,
                &cfg,
                &mut HostTranscript::new(&[]),
            ),
            Some(claims) => multilinear::stacked_eval::prove_mapped_ood::<F, E, _, Blake3Whir, _>(
                &stacked,
                &borrowed,
                None,
                &Claimed::PerColumn(&points),
                &values,
                Some(claims),
                &cfg,
                &mut HostTranscript::new(&[]),
            ),
        }
        .expect("the group proves");
        let f = Fixture {
            shape: ChainShape::for_hash(&cfg, layout.n_stack(), WrapHash::Blake3),
            cfg,
            layout,
            points,
            values,
            roots: stacked.roots(),
            proof,
            domain: stacked.domain().clone(),
            ood: ood.map(|claims| (claims.z0, claims.answers)),
        };
        if answer_delta.is_none_or(|d| d == FEE::zero()) {
            assert!(host_accepts(&f, &f.proof), "the host accepts its own proof");
        }
        f
    }

    fn host_accepts(f: &Fixture, proof: &StackedProof<F, E>) -> bool {
        if let Some((z0, answers)) = &f.ood {
            return multilinear::stacked_eval::verify_ood::<F, E, _, Blake3Whir>(
                proof,
                &f.layout,
                &f.roots,
                &Claimed::PerColumn(&f.points),
                &f.values,
                Some(multilinear::stacked_eval::CommitOod { z0, answers }),
                &f.domain,
                &f.cfg,
                &mut HostTranscript::new(&[]),
            )
            .is_ok();
        }
        multilinear::stacked_eval::verify::<F, E, _, Blake3Whir>(
            proof,
            &f.layout,
            &f.roots,
            &Claimed::PerColumn(&f.points),
            &f.values,
            &f.domain,
            &f.cfg,
            &mut HostTranscript::new(&[]),
        )
        .is_ok()
    }

    /// The trees of each run: one a polynomial, or the run's one joint tree.
    fn run_trees(f: &Fixture) -> Vec<std::ops::Range<usize>> {
        let trees = multilinear::stacked_eval::round0_trees(f.layout.num_polys(), &f.cfg);
        f.shape
            .chains(f.layout.num_polys())
            .iter()
            .map(|run| {
                let first = trees
                    .iter()
                    .position(|t| t.start == run.start)
                    .expect("a tree starts the run");
                let last = trees
                    .iter()
                    .position(|t| t.end == run.end)
                    .expect("a tree ends the run");
                first..last + 1
            })
            .collect()
    }

    /// Each column's point, the values, then per run its roots (two words
    /// each), its final value and its rounds. Publishes the batching draw.
    pub(super) fn program(f: &Fixture) -> LfmProgram {
        let mut b = builder();
        let runs = f.shape.chains(f.layout.num_polys());
        let trees = run_trees(f);
        let mut total: u32 =
            f.points.iter().map(|p| p.len() as u32).sum::<u32>() + f.values.len() as u32;
        let mut chain_at = Vec::new();
        for (run, t) in runs.iter().zip(&trees) {
            chain_at.push(total);
            total += 2 * t.len() as u32 + 1 + RoundStorage::words(&f.shape.run(run.len()));
        }
        // The answers' point (the stack's coordinates) and the answers, after
        // the chains.
        let ood_at = total;
        if f.ood.is_some() {
            total += (f.layout.n_stack() + f.layout.num_polys()) as u32;
        }
        let arena = b.declare_arena(total);
        let mut transcript = WhirTranscript::for_builder(&mut b);
        let mut next = 0u32;
        let points: Vec<Vec<Ext>> = f
            .points
            .iter()
            .map(|p| {
                p.iter()
                    .map(|_| {
                        let w = b.hint_word(arena, next).as_ext();
                        next += 1;
                        w
                    })
                    .collect()
            })
            .collect();
        let values: Vec<Ext> = (0..f.values.len() as u32)
            .map(|_| {
                let w = b.hint_word(arena, next).as_ext();
                next += 1;
                w
            })
            .collect();
        let mut roots: Vec<Vec<Cell>> = Vec::new();
        let mut finals: Vec<Ext> = Vec::new();
        let mut storage: Vec<RoundStorage> = Vec::new();
        for ((run, t), &base) in runs.iter().zip(&trees).zip(&chain_at) {
            let n = 2 * t.len() as u32;
            roots.push((0..n).map(|i| b.hint_word(arena, base + i)).collect());
            finals.push(b.hint_word(arena, base + n).as_ext());
            storage.push(RoundStorage::hint(
                &mut b,
                arena,
                base + n + 1,
                &f.shape.run(run.len()),
            ));
        }
        let openings: Vec<_> = storage.iter().map(RoundStorage::openings).collect();
        let rounds: Vec<Vec<_>> = storage
            .iter()
            .zip(&openings)
            .map(|(chain, (current, next))| chain.wires(current, next))
            .collect();
        let chains: Vec<StackedChainWires<'_>> = (0..runs.len())
            .map(|i| StackedChainWires {
                rounds: &rounds[i],
                roots: &roots[i],
                final_value: finals[i],
            })
            .collect();
        let claimed: Vec<&[Ext]> = points.iter().map(Vec::as_slice).collect();
        let hinted = |b: &mut super::super::builder::LfmBuilder, from: u32, n: usize| -> Vec<Ext> {
            (0..n as u32)
                .map(|i| b.hint_word(arena, from + i).as_ext())
                .collect()
        };
        let ood_wires = f.ood.as_ref().map(|_| {
            let point = hinted(&mut b, ood_at, f.layout.n_stack());
            let answers = hinted(
                &mut b,
                ood_at + f.layout.n_stack() as u32,
                f.layout.num_polys(),
            );
            (point, answers)
        });
        let gamma = emit_stacked_verify_chains(
            &mut b,
            &mut transcript,
            &f.layout,
            &chains,
            &claimed,
            &values,
            ood_wires
                .as_ref()
                .map(|(point, answers)| OodWires { point, answers }),
            &f.shape,
            &f.domain,
        );
        b.public(gamma.as_cell());
        let program = compile(b.finish());
        validate(&program).expect("the BLAKE3 grouped leg is admissible");
        program
    }

    fn arena(f: &Fixture, proof: &StackedProof<F, E>) -> Vec<LfmWord> {
        let mut words: Vec<LfmWord> = f.points.iter().flatten().map(ext_word).collect();
        words.extend(f.values.iter().map(ext_word));
        let runs = f.shape.chains(f.layout.num_polys());
        for ((chain, run), t) in proof.polys.iter().zip(&runs).zip(run_trees(f)) {
            for root in &f.roots[t] {
                words.extend(byte_digest_words(root));
            }
            words.push(ext_word(&chain.final_value));
            push_round_words(&mut words, &f.shape.run(run.len()), chain);
        }
        if let Some((z0, answers)) = &f.ood {
            let point = multilinear::whir_chain::ood_point(z0, f.layout.n_stack());
            words.extend(point.iter().map(ext_word));
            words.extend(answers.iter().map(ext_word));
        }
        words
    }

    fn runs(program: &LfmProgram, words: Vec<LfmWord>) -> Result<FEE, String> {
        let exec = execute(program, &[words], &crate::hash_pin::BLOCK_HASHER)
            .map_err(|e| format!("{e:?}"))?;
        word_as_ext(&exec.public_words[0].1).ok_or_else(|| "no published draw".to_string())
    }

    /// The host's batching draw: the stacked verifier's first sample after
    /// absorbing the claimed values, replayed on the host transcript.
    fn host_gamma(f: &Fixture) -> FEE {
        use crypto::fiat_shamir::is_transcript::IsTranscript;
        let mut t = HostTranscript::new(&[]);
        for v in &f.values {
            t.append_field_element(v);
        }
        t.sample_field_element()
    }

    pub(super) fn cases() -> Vec<(&'static str, Case)> {
        let base = |heights: &[usize], n_stack| Case {
            heights: heights.to_vec(),
            n_stack,
            first: Some(6),
            cap: CapPolicy::Off,
            leaves: LeafOrder::Natural,
            round0: Round0Trees::Each,
            grind: 4,
            whir: None,
        };
        let mut cases = Vec::new();
        for round0 in [Round0Trees::Each, Round0Trees::One] {
            for leaves in [LeafOrder::Natural, LeafOrder::Coset] {
                for cap in [CapPolicy::Off, CapPolicy::Fixed(2), CapPolicy::Auto] {
                    cases.push((
                        "three polys",
                        Case {
                            round0,
                            leaves,
                            cap,
                            ..base(&[9; 3], 9)
                        },
                    ));
                }
                cases.push((
                    "two polys",
                    Case {
                        round0,
                        leaves,
                        ..base(&[9; 2], 9)
                    },
                ));
                cases.push((
                    "four polys",
                    Case {
                        round0,
                        leaves,
                        ..base(&[9; 4], 9)
                    },
                ));
                cases.push((
                    "five polys split [4, 1]",
                    Case {
                        round0,
                        leaves,
                        ..base(&[8; 5], 8)
                    },
                ));
                cases.push((
                    "mixed heights",
                    Case {
                        round0,
                        leaves,
                        ..base(&[8, 5, 8, 3, 7, 8], 8)
                    },
                ));
            }
        }
        cases
    }

    /// ★ THE GROUPED GATE. Under one tree a polynomial and one a run, natural
    /// and coset leaves, three caps, runs of two to four and a split, the
    /// emitted leg executes on every proof the host accepts, its batching draw
    /// is the host's, and it hashes with BLAKE3 alone.
    #[test]
    pub(super) fn a_blake3_group_opening_executes_on_proofs_the_host_accepts() {
        super::under_both(|_lowering| {
            for (name, case) in cases() {
                let f = build(&case);
                let joint = f
                    .shape
                    .chains(f.layout.num_polys())
                    .iter()
                    .any(|run| f.shape.run(run.len()).joint());
                assert_eq!(
                    joint,
                    case.round0 == Round0Trees::One,
                    "{name}: a joint run exactly under one tree a run"
                );
                let program = program(&f);
                let (blake3, other) = hash_rows(&program);
                assert!(blake3 > 0);
                assert_eq!(other, 0, "{name}: BLAKE3 alone");
                let gamma = runs(&program, arena(&f, &f.proof)).unwrap_or_else(|e| {
                    panic!(
                        "{name} ({:?}, {:?}, {:?}): refused an accepted proof: {e}",
                        case.round0, case.leaves, case.cap
                    )
                });
                assert_eq!(gamma, host_gamma(&f), "{name}: the batching draw");
            }
        });
    }

    /// ★ THE JOINT TAMPER ARM. One tree a run, natural and coset, uncapped and
    /// capped: a value off in any block of a joint leaf, two blocks swapped,
    /// a joint path node off (its last byte), and a root off in its second
    /// word are refused by the host and have no execution.
    #[test]
    pub(super) fn a_tampered_joint_blake3_leg_cannot_execute() {
        super::under_both(|_lowering| {
            for leaves in [LeafOrder::Natural, LeafOrder::Coset] {
                for cap in [CapPolicy::Off, CapPolicy::Fixed(2)] {
                    let case = Case {
                        heights: vec![9; 3],
                        n_stack: 9,
                        first: Some(6),
                        cap,
                        leaves,
                        round0: Round0Trees::One,
                        grind: 0,
                        whir: None,
                    };
                    let f = build(&case);
                    let program = program(&f);
                    runs(&program, arena(&f, &f.proof)).expect("the control executes");
                    let width = f.shape.run(3).current_block(0);
                    assert_eq!(width, 3 * 64);
                    let mut forgeries = Vec::new();
                    for t in [0usize, 63, 64, 64 + 5, width - 1] {
                        let mut forged = f.proof.clone();
                        match &mut forged.polys[0].rounds[0].openings {
                            RoundOpenings::BaseJoint(p) => p.current[1].values[t] += FE::one(),
                            _ => panic!("chain 0 is joint"),
                        }
                        forgeries.push((format!("value {t}"), forged));
                    }
                    let mut swapped = f.proof.clone();
                    match &mut swapped.polys[0].rounds[0].openings {
                        RoundOpenings::BaseJoint(p) => {
                            for opening in &mut p.current {
                                let (a, b) = opening.values.split_at_mut(64);
                                a.swap_with_slice(&mut b[..64]);
                            }
                        }
                        _ => panic!("chain 0 is joint"),
                    }
                    forgeries.push(("blocks 0 and 1 swapped".to_string(), swapped));
                    let mut node = f.proof.clone();
                    match &mut node.polys[0].rounds[0].openings {
                        RoundOpenings::BaseJoint(p) => p.current[2].proof.merkle_path[0][31] ^= 1,
                        _ => panic!("chain 0 is joint"),
                    }
                    forgeries.push(("a path node".to_string(), node));
                    for (what, forged) in &forgeries {
                        assert!(
                            !host_accepts(&f, forged),
                            "{what} ({leaves:?}, {cap:?}): host"
                        );
                        assert!(
                            runs(&program, arena(&f, forged)).is_err(),
                            "{what} ({leaves:?}, {cap:?}): must have no execution"
                        );
                    }
                    let mut words = arena(&f, &f.proof);
                    let root_at = f.points.iter().map(Vec::len).sum::<usize>() + f.values.len();
                    words[root_at + 1][2] += FE::one();
                    assert!(
                        runs(&program, words).is_err(),
                        "a joint root off in its second word"
                    );
                }
            }
        });
    }

    /// ★ The commit-time answers under BLAKE3 and one tree a run, in-guest: an
    /// opening that carries its answers (the point hinted) executes on the
    /// host's honest proof, under both lowerings, today's form and the round
    /// form; and a false answer that the prover states and its joint chain
    /// carries — polynomial 0, in a run of two or more — has no execution, as
    /// the host refuses it.
    #[test]
    fn a_joint_blake3_opening_binds_its_answers() {
        let mut picked: Vec<(&'static str, Case)> = cases()
            .into_iter()
            .filter(|(_, c)| c.round0 == Round0Trees::One && c.cap == CapPolicy::Off)
            .collect();
        picked.extend(
            round_form_cases()
                .into_iter()
                .filter(|(n, c)| *n == "three polys, shrink 2" && c.round0 == Round0Trees::One),
        );
        assert!(
            picked.iter().any(|(_, c)| c.whir.is_some()),
            "a round-form case"
        );
        super::under_both(|lowering| {
            for (name, case) in &picked {
                let honest = build_with(case, Some(FEE::zero()));
                let first = honest.shape.chains(honest.layout.num_polys())[0].clone();
                assert!(
                    first.start == 0 && honest.shape.run(first.len()).joint(),
                    "{name}: polynomial 0 lies in a joint run"
                );
                runs(&program(&honest), arena(&honest, &honest.proof)).unwrap_or_else(|e| {
                    panic!("{lowering} {name}: the honest opening with answers refused: {e}")
                });
                let forged = build_with(case, Some(FEE::one()));
                assert!(
                    !host_accepts(&forged, &forged.proof),
                    "{name}: a false answer its chain carries, host"
                );
                assert!(
                    runs(&program(&forged), arena(&forged, &forged.proof)).is_err(),
                    "{lowering} {name}: a false answer its joint chain carries executed"
                );
            }
        });
    }

    /// A joint leg and a per-tree leg refuse each other's proofs.
    #[test]
    fn a_joint_blake3_leg_and_a_per_tree_one_refuse_each_others_proofs() {
        super::under_both(|_lowering| {
            let at = |round0| Case {
                heights: vec![9; 3],
                n_stack: 9,
                first: Some(6),
                cap: CapPolicy::Off,
                leaves: LeafOrder::Coset,
                round0,
                grind: 0,
                whir: None,
            };
            let each = build(&at(Round0Trees::Each));
            let one = build(&at(Round0Trees::One));
            let (each_program, one_program) = (program(&each), program(&one));
            assert!(runs(&one_program, arena(&each, &each.proof)).is_err());
            assert!(runs(&each_program, arena(&one, &one.proof)).is_err());
        });
    }

    /// The group openings the round-form gates run: every case above under
    /// the round form at shrink 3 (and the three-polynomial runs at shrink 2),
    /// their counts sized for 20 bits.
    fn round_form_cases() -> Vec<(&'static str, Case)> {
        let mut out = Vec::new();
        for (name, case) in cases() {
            if name == "three polys" && case.cap == CapPolicy::Off {
                out.push((
                    "three polys, shrink 2",
                    Case {
                        whir: Some((2, 20)),
                        ..case.clone()
                    },
                ));
            }
            out.push((
                name,
                Case {
                    whir: Some((3, 20)),
                    ..case
                },
            ));
        }
        out
    }

    /// ★ THE ROUND-FORM GROUPED GATE. Under the round form — round 0 one tree
    /// a polynomial or one a run, its every tree (or its joint leaf) opened at
    /// each drawn position and the blocks combined and folded into a claim on
    /// the successor — the emitted leg executes on every proof the host
    /// accepts, under both lowerings; its batching draw is the host's; and it
    /// hashes with BLAKE3 alone, fewer compressions than today's form.
    #[test]
    fn a_round_form_blake3_group_opening_executes_on_proofs_the_host_accepts() {
        super::under_both(|lowering| {
            for (name, case) in round_form_cases() {
                let f = build(&case);
                assert!(f.shape.whir(), "{name}: the round form");
                let program = program(&f);
                let (blake3, other) = hash_rows(&program);
                assert_eq!(other, 0, "{name}: BLAKE3 alone");
                let gamma = runs(&program, arena(&f, &f.proof)).unwrap_or_else(|e| {
                    panic!(
                        "{lowering} {name} ({:?}, {:?}, {:?}): refused an accepted proof: {e}",
                        case.round0, case.leaves, case.cap
                    )
                });
                assert_eq!(gamma, host_gamma(&f), "{name}: the batching draw");
                println!(
                    "ROUND FORM GROUP {lowering} {name} ({:?}, {:?}, {:?}): queries {:?} · {blake3} \
                     compressions",
                    case.round0, case.leaves, case.cap, f.shape.round_queries
                );
            }
        });
    }

    /// ★ THE ROUND-FORM JOINT TAMPER ARM. One tree a run, natural and coset:
    /// a value off in a joint leaf's block, a joint path node, a later round's
    /// opened value, and an out-of-domain value are refused by the host and
    /// have no execution; a successor block at round 0 is refused by the
    /// arena too.
    #[test]
    fn a_tampered_round_form_joint_blake3_leg_cannot_execute() {
        super::under_both(|lowering| {
            for leaves in [LeafOrder::Natural, LeafOrder::Coset] {
                let case = Case {
                    heights: vec![9; 3],
                    n_stack: 9,
                    first: Some(6),
                    cap: CapPolicy::Fixed(2),
                    leaves,
                    round0: Round0Trees::One,
                    grind: 0,
                    whir: Some((3, 20)),
                };
                let f = build(&case);
                let program = program(&f);
                runs(&program, arena(&f, &f.proof)).expect("the control executes");
                let mut forgeries = Vec::new();
                for t in [0usize, 64 + 5] {
                    let mut forged = f.proof.clone();
                    match &mut forged.polys[0].rounds[0].openings {
                        RoundOpenings::BaseJoint(p) => p.current[1].values[t] += FE::one(),
                        _ => panic!("chain 0 is joint"),
                    }
                    forgeries.push((format!("value {t}"), forged));
                }
                let mut node = f.proof.clone();
                match &mut node.polys[0].rounds[0].openings {
                    RoundOpenings::BaseJoint(p) => p.current[2].proof.merkle_path[0][31] ^= 1,
                    _ => panic!("chain 0 is joint"),
                }
                forgeries.push(("a path node".to_string(), node));
                let mut later = f.proof.clone();
                match &mut later.polys[0].rounds[1].openings {
                    RoundOpenings::Extension(p) => p.current[0].values[1] += FEE::one(),
                    _ => panic!("round 1 is extension-valued"),
                }
                forgeries.push(("round 1's opened value".to_string(), later));
                let mut ood = f.proof.clone();
                if let Some(v) = ood.polys[0].rounds[0].ood_value.as_mut() {
                    *v += FEE::one();
                }
                forgeries.push(("round 0's out-of-domain value".to_string(), ood));
                let mut successor = f.proof.clone();
                let block = match &successor.polys[0].rounds[1].openings {
                    RoundOpenings::Extension(p) => p.current[0].clone(),
                    _ => panic!("round 1 is extension-valued"),
                };
                match &mut successor.polys[0].rounds[0].openings {
                    RoundOpenings::BaseJoint(p) => p.next.push(block),
                    _ => panic!("chain 0 is joint"),
                }
                let mut words = Vec::new();
                assert!(
                    super::super::whir_chain::try_push_round_words(
                        &mut words,
                        &f.shape.run(3),
                        &successor.polys[0]
                    )
                    .is_err(),
                    "a successor block: the arena refuses it"
                );
                forgeries.push(("a successor block at round 0".to_string(), successor));
                for (what, forged) in &forgeries {
                    assert!(!host_accepts(&f, forged), "{what} ({leaves:?}): host");
                    assert!(
                        runs(&program, arena(&f, forged)).is_err(),
                        "{lowering} {what} ({leaves:?}): must have no execution"
                    );
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------
// The two lowerings side by side.
// ---------------------------------------------------------------------------

/// ★ What the lean lowering changes is the chip and the byte work, never the
/// compressions: a chain emits the same number of compressions either way,
/// all of them `B3l` rows and none `LFM_BLAKE3` rows under the lean builder,
/// and its leaves take no `BitDec` a value (the head renders the bytes), so the
/// lean program has fewer `BitDec` rows by at least the leaf values hashed.
#[test]
fn the_lean_lowering_moves_the_chip_and_the_byte_work_only() {
    use super::whir_chain::ChipRows;
    use multilinear::whir_chain::{CapPolicy, LeafOrder};
    let cfg = chain::config(4, 6, CapPolicy::Fixed(2), LeafOrder::Coset, Some(6));
    let shape = super::whir_chain::ChainShape::for_hash(&cfg, 10, WrapHash::Blake3);
    let mut programs = Vec::new();
    under_both(|_| programs.push(chain::program(&shape)));
    let [plain, lean] = [&programs[0], &programs[1]];
    let count = |p: &LfmProgram, f: fn(&Instr) -> bool| p.instrs.iter().filter(|i| f(i)).count();
    let plain_rows = count(plain, |i| matches!(i, Instr::Blake3(_)));
    let lean_rows = count(lean, |i| matches!(i, Instr::B3l(_)));
    assert_eq!(plain_rows, lean_rows, "the same compressions");
    assert_eq!(count(lean, |i| matches!(i, Instr::Blake3(_))), 0);
    assert_eq!(count(plain, |i| matches!(i, Instr::B3l(_))), 0);
    let (p, l) = (ChipRows::of(plain), ChipRows::of(lean));
    // Round 0's base values alone: Q queries of a 64-value block.
    let round0_values = shape.num_queries * 64;
    assert!(
        p.bitdec >= l.bitdec + round0_values,
        "lean leaves render no value with a BitDec: {} against {}",
        l.bitdec,
        p.bitdec
    );
    assert!(lean.instrs.len() < plain.instrs.len());
    println!(
        "chain S=10: {plain_rows} compressions; instructions {} (LFM_BLAKE3) vs {} (lean); \
         BitDec {} vs {}",
        plain.instrs.len(),
        lean.instrs.len(),
        p.bitdec,
        l.bitdec
    );
}

/// The production chain's compressions under BLAKE3, by closed form (the cost
/// model the chain gates hold to the emission): a run of `P'` stacked
/// polynomials of the default stack, one tree a run (joint leaves) and one a
/// polynomial, against the same run's Poseidon1 permutations. Prints, asserts
/// only the forms' internal consistency.
#[test]
fn the_production_blake3_run_costs_its_closed_form() {
    use super::whir_chain::{
        ChainShape, chain_cap_perms, chain_grind_perms, chain_opening_perms, chain_perms,
    };
    use super::whir_transcript::SpongeEntry;
    use crate::zf_format::ZfFormat;
    use multilinear::whir_chain::{GroupFold, Round0Trees};
    let zf = ZfFormat::DEFAULT;
    let n = zf.whir_stack.get();
    let mut config = crate::multilinear_prove::chain_config_under(&zf, &[(3, n)]);
    config.format.group = GroupFold::Indexed;
    config.format.leaves = zf.whir_leaves;
    for trees in [Round0Trees::One, Round0Trees::Each] {
        config.format.round0 = trees;
        for run in [1usize, 2, 3] {
            let b3 = ChainShape::for_hash(&config, n, WrapHash::Blake3).run(run);
            let p1 = ChainShape::for_hash(&config, n, WrapHash::Poseidon1).run(run);
            let b3_total = chain_perms(&b3, SpongeEntry::fresh_for(WrapHash::Blake3));
            let p1_total = chain_perms(&p1, SpongeEntry::fresh_for(WrapHash::Poseidon1));
            assert!(chain_opening_perms(&b3) <= b3_total);
            println!(
                "PRODUCTION RUN {trees:?} x{run}: stack 2^{n} · schedule {:?} · Q {} · caps {:?} · \
                 BLAKE3 compressions {b3_total} (openings {} incl. caps {}, grinds {}) · Poseidon1 \
                 permutations {p1_total} (openings {})",
                b3.schedule,
                b3.num_queries,
                b3.caps,
                chain_opening_perms(&b3),
                chain_cap_perms(&b3),
                chain_grind_perms(&b3),
                chain_opening_perms(&p1),
            );
        }
    }
}

/// ★ The BLAKE3 base's own format, by closed form: its block format's chain
/// config (the binary cap [`crate::block_whir::B3_BASE_CAP`] and the final
/// polynomial [`crate::block_whir::B3_BASE_FINAL_VARS`], every other setting
/// production's) under the leaf's lean query check. A run of three 2^27
/// polynomials, one tree a run: the final round is round 3, so rounds 4 to 6
/// open nothing. Against the same config without the final polynomial (the
/// tail rounds 5 and 6 then checked once a leaf: 30,962) and today's 41,109
/// (cap 3, every round per query).
#[test]
fn the_blake3_base_format_run_costs_its_closed_form() {
    use super::whir_chain::{
        ChainShape, LeafQuery, chain_cap_perms, chain_grind_perms, chain_opening_perms, chain_perms,
    };
    use super::whir_transcript::SpongeEntry;
    use crate::block_whir::{B3_BASE_CAP, B3_BASE_FINAL_VARS, BlockFormat};
    use crate::zf_format::ZfFormat;
    use multilinear::whir_chain::FinalPoly;
    use stark::proof::options::BaseFormat;
    let zf = ZfFormat::DEFAULT.with_base(BaseFormat::B3_WHIR);
    let format = BlockFormat {
        zf,
        ..BlockFormat::production()
    };
    let n = zf.whir_stack.get();
    let config = format.chain_config(&[(3, n)]);
    assert_eq!(
        config.format.cap,
        multilinear::whir_chain::CapPolicy::Fixed(B3_BASE_CAP)
    );
    assert_eq!(
        config.format.final_poly,
        FinalPoly::Vars(B3_BASE_FINAL_VARS)
    );
    assert_eq!(config.final_round(n), Some((3, 9)), "round 3, 2^9 values");
    let lean = |config: &multilinear::whir_chain::ChainConfig| {
        ChainShape {
            query: LeafQuery::Lean,
            ..ChainShape::for_hash(config, n, WrapHash::Blake3)
        }
        .run(3)
    };
    let shape = lean(&config);
    let mut unfinal = config;
    unfinal.format.final_poly = FinalPoly::Off;
    let without = lean(&unfinal);
    let tails: Vec<usize> = (0..without.rounds())
        .filter(|&r| without.tail_round(r))
        .collect();
    assert_eq!(tails, vec![5, 6], "rounds 5 and 6 lie wholly in their caps");
    assert!(
        (0..shape.rounds()).all(|r| !shape.tail_round(r)),
        "no tail round past a final polynomial"
    );
    let total = chain_perms(&shape, SpongeEntry::fresh_for(WrapHash::Blake3));
    let without_total = chain_perms(&without, SpongeEntry::fresh_for(WrapHash::Blake3));
    let per_query = chain_perms(
        &ChainShape {
            query: LeafQuery::V1,
            ..without.clone()
        },
        SpongeEntry::fresh_for(WrapHash::Blake3),
    );
    let sponge = |s: &ChainShape| {
        chain_perms(s, SpongeEntry::fresh_for(WrapHash::Blake3))
            - chain_opening_perms(s)
            - chain_grind_perms(s)
    };
    println!(
        "BLAKE3 BASE FORMAT RUN x3: caps {:?} · final round {:?} · {total} compressions (openings \
         {} incl. caps {}, grinds {}, transcript {}) · without the final polynomial {without_total} \
         (openings {} incl. caps {}, grinds {}, transcript {}; tail rounds {tails:?}) · without the \
         tail too {per_query} · today's 41109",
        shape.caps,
        shape.final_at,
        chain_opening_perms(&shape),
        chain_cap_perms(&shape),
        chain_grind_perms(&shape),
        sponge(&shape),
        chain_opening_perms(&without),
        chain_cap_perms(&without),
        chain_grind_perms(&without),
        sponge(&without),
    );
    // The final polynomial alone against today's chain (the binary default
    // cap, every round per query): the lever measured on its own control.
    let mut today = unfinal;
    today.format.cap = crate::multilinear_prove::chain_config_under(&zf, &[(3, n)])
        .format
        .cap;
    let today_shape = ChainShape {
        query: LeafQuery::V1,
        ..ChainShape::for_hash(&today, n, WrapHash::Blake3)
    }
    .run(3);
    let mut alone = today;
    alone.format.final_poly = FinalPoly::Vars(B3_BASE_FINAL_VARS);
    let alone_shape = ChainShape {
        query: LeafQuery::V1,
        ..ChainShape::for_hash(&alone, n, WrapHash::Blake3)
    }
    .run(3);
    let today_total = chain_perms(&today_shape, SpongeEntry::fresh_for(WrapHash::Blake3));
    let alone_total = chain_perms(&alone_shape, SpongeEntry::fresh_for(WrapHash::Blake3));
    println!(
        "BLAKE3 FINAL POLYNOMIAL ALONE: today's {today_total} (caps {:?}) -> {alone_total} ({:+}); on the base's \
         cap and tail: {without_total} -> {total} ({:+})",
        today_shape.caps,
        alone_total as i64 - today_total as i64,
        total as i64 - without_total as i64
    );
    // Four polynomials a group (`B3_BASE_GROUP_POLYS`): one run of four.
    let four = ChainShape {
        query: LeafQuery::Lean,
        ..ChainShape::for_hash(&format.chain_config(&[(4, n)]), n, WrapHash::Blake3)
    }
    .run(4);
    let four_total = chain_perms(&four, SpongeEntry::fresh_for(WrapHash::Blake3));
    println!(
        "BLAKE3 BASE FORMAT RUN x4: {four_total} compressions ({} a polynomial) against the run of three's \
         {total} ({} a polynomial): {:+.1} % a group, {:+.1} % a polynomial",
        four_total / 4,
        total / 3,
        100.0 * (four_total as f64 / total as f64 - 1.0),
        100.0 * ((four_total as f64 / 4.0) / (total as f64 / 3.0) - 1.0)
    );
    assert!(
        four_total > total && four_total * 3 < total * 4,
        "a run of four costs less a polynomial"
    );
    assert_eq!(today_total, 41_109, "today's chain");
    assert_eq!(
        without_total, 30_962,
        "the format before its final polynomial"
    );
    assert!(total < without_total && without_total < per_query && per_query < 41_109);
}

/// The levers on the production BLAKE3 run's compressions, by closed form: the
/// default (one run of three 2^27 polynomials, one tree a run), then one
/// setting moved at a time — the query grind, the blowup, the first fold, the
/// later folds, the binary cap, the run width — each with its query count and
/// its compressions, and the default's per-round split (current tree and
/// successor, a query). Prints; asserts nothing but that the forms run.
#[test]
#[ignore = "census: prints the levers' closed forms"]
fn the_production_blake3_run_levers_by_closed_form() {
    use super::whir_chain::{ChainShape, chain_perms};
    use super::whir_transcript::SpongeEntry;
    use crate::zf_format::ZfFormat;
    use multilinear::whir_chain::{
        CapPolicy, ChainConfig, FirstFold, GrindBits, GroupFold, Round0Trees, WhirFolds,
    };
    let zf = ZfFormat::DEFAULT;
    let n = zf.whir_stack.get();
    let build = |log_blowup: usize, log_folding: usize, k0: usize, grind: u8, cap: Option<u8>| {
        let base = ChainConfig::with_security_folds(
            log_blowup,
            log_folding,
            WhirFolds::First(FirstFold::new(k0).expect("a first fold")),
            n,
            128,
            GrindBits {
                folding: 0,
                ood: 0,
                query: grind,
            },
        );
        let mut config = zf.chain(base);
        // The format's own folds would replace the lever's: the query count
        // above was derived under the lever's, so they are put back.
        config.format.folds = base.format.folds;
        config.format.group = GroupFold::Indexed;
        config.format.leaves = zf.whir_leaves;
        config.format.round0 = Round0Trees::One;
        if let Some(c) = cap {
            config.format.cap = CapPolicy::Fixed(c);
        }
        config
    };
    let run = |config: &ChainConfig, polys: usize| {
        let shape = ChainShape::for_hash(config, n, WrapHash::Blake3).run(polys);
        let total = chain_perms(&shape, SpongeEntry::fresh_for(WrapHash::Blake3));
        (shape, total)
    };
    let (base_shape, base) = run(&build(1, 4, 6, 18, None), 3);
    println!(
        "LEVER base: Q {} · schedule {:?} · caps {:?} · domains {:?} · {base} compressions",
        base_shape.num_queries, base_shape.schedule, base_shape.caps, base_shape.domain_log
    );
    // The default's split, a query: round r's current opening (per tree at
    // round 0) and its successor's.
    for r in 0..base_shape.rounds() {
        let current = base_shape.trees_at(r)
            * base_shape.opening_perms(
                base_shape.current_felts(r),
                base_shape.current_depth(r),
                base_shape.caps[r],
            );
        let successor = base_shape.next_depth(r).map_or(0, |d| {
            base_shape.opening_perms(3 << base_shape.schedule[r + 1], d, base_shape.caps[r + 1])
        });
        println!(
            "LEVER split round {r}: tree depth {} cap {} · current {current} + successor {successor} \
             = {} a query · {} at Q",
            base_shape.current_depth(r),
            base_shape.caps[r],
            current + successor,
            (current + successor) * base_shape.num_queries
        );
    }
    let report = |name: &str, shape: &ChainShape, total: usize| {
        println!(
            "LEVER {name}: Q {} · rounds {} · {total} compressions · {:+} against the default",
            shape.num_queries,
            shape.rounds(),
            total as i64 - base as i64
        );
    };
    for grind in [20u8, 22, 24, 26, 28] {
        let (s, t) = run(&build(1, 4, 6, grind, None), 3);
        report(&format!("query grind {grind} bits"), &s, t);
    }
    for lb in [2usize, 3] {
        let (s, t) = run(&build(lb, 4, 6, 18, None), 3);
        report(&format!("blowup 2^{lb}"), &s, t);
    }
    for k0 in [4usize, 5] {
        let (s, t) = run(&build(1, 4, k0, 18, None), 3);
        report(&format!("first fold {k0}"), &s, t);
    }
    for lf in [3usize, 5, 6] {
        let (s, t) = run(&build(1, lf, 6, 18, None), 3);
        report(&format!("later folds {lf}"), &s, t);
    }
    for c in [4u8, 5, 6, 7, 8] {
        let (s, t) = run(&build(1, 4, 6, 18, Some(c)), 3);
        report(&format!("binary cap {c} (caps {:?})", s.caps), &s, t);
    }
    for polys in [1usize, 2, 4] {
        let (s, t) = run(&build(1, 4, 6, 18, None), polys);
        report(
            &format!("run of {polys} ({} a polynomial)", t / polys),
            &s,
            t,
        );
    }
    let (s, t) = run(&build(1, 4, 6, 24, Some(6)), 3);
    report("query grind 24 + binary cap 6", &s, t);
}

/// Runs `f` with this thread's cap check in `mode` whatever the knob says
/// (`merkle_cap::CAP_MUX_OVERRIDE`), then restores the knob's reading.
fn with_cap_mux(mode: super::merkle_cap::CapMux, f: impl FnOnce()) {
    super::merkle_cap::CAP_MUX_OVERRIDE.with(|c| c.set(Some(mode)));
    f();
    super::merkle_cap::CAP_MUX_OVERRIDE.with(|c| c.set(None));
}

/// ★ THE ROLLBACK MUX GATE. The cap is checked over packed digests by default
/// (`LAMBDA_VM_LFM_CAP_MUX` unset), so every gate above runs packed; this runs
/// every BLAKE3 chain and group gate again with the plain mux
/// (`LAMBDA_VM_LFM_CAP_MUX=plain`, the rollback): honest proofs execute, every
/// tamper has no execution, the compressions are the closed form's, under
/// both lowerings.
#[test]
fn every_blake3_gate_holds_under_the_plain_cap_mux() {
    with_cap_mux(super::merkle_cap::CapMux::Plain, || {
        chain::a_blake3_chain_executes_on_a_proof_the_host_accepts();
        chain::a_blake3_chain_emits_its_compression_closed_form();
        chain::a_tampered_blake3_chain_cannot_execute();
        chain::a_lean_blake3_chain_checks_its_tail_once_a_leaf();
        chain::a_blake3_chain_with_a_final_polynomial_executes_and_refuses_its_tampers();
        stacked::a_blake3_group_opening_executes_on_proofs_the_host_accepts();
        stacked::a_tampered_joint_blake3_leg_cannot_execute();
    });
}

/// The packed mux (the default) muxes one word a node instead of the plain
/// mux's two (about half the cap's `Select` rows), with the same compressions.
#[test]
fn the_packed_cap_mux_halves_the_muxs_select_rows() {
    use super::whir_chain::ChainShape;
    use multilinear::whir_chain::{CapPolicy, LeafOrder};
    let cfg = chain::config(4, 0, CapPolicy::Fixed(4), LeafOrder::Coset, Some(6));
    let shape = ChainShape::for_hash(&cfg, 12, WrapHash::Blake3);
    assert!(
        shape.caps.iter().any(|&c| c >= 3),
        "a cap the mux reads: {:?}",
        shape.caps
    );
    let selects = |p: &LfmProgram| {
        p.instrs
            .iter()
            .filter(|i| matches!(i, Instr::Select { .. } | Instr::Mux { .. }))
            .count()
    };
    let (mut plain, mut packed) = (None, None);
    use super::merkle_cap::CapMux;
    with_cap_mux(CapMux::Plain, || plain = Some(chain::program(&shape)));
    with_cap_mux(CapMux::Packed, || packed = Some(chain::program(&shape)));
    let (plain, packed) = (
        plain.expect("the plain program"),
        packed.expect("the packed program"),
    );
    let default = chain::program(&shape);
    assert_eq!(
        default.instrs.len(),
        packed.instrs.len(),
        "the default is the packed mux"
    );
    assert_eq!(
        hash_rows(&plain),
        hash_rows(&packed),
        "the packing hashes nothing"
    );
    let (s_plain, s_packed) = (selects(&plain), selects(&packed));
    println!(
        "PACKED MUX: caps {:?} · Select rows {s_plain} -> {s_packed}",
        shape.caps
    );
    assert!(s_packed < s_plain, "the packed mux takes fewer Select rows");
}

/// ★ What the packed compare admits, exactly. A height-2 cap of four nodes is
/// authenticated against its host root; an opening at the cap's own level
/// (no walk) compares a hinted digest with node `idx`. Packed, the node
/// itself executes, every single-lane change of it is refused, and the
/// node's documented partner — a `(lo = a, hi = 0)` pair replaced by
/// `(lo = a + 1, hi = 2^32 − 1)`, the one non-injective case of
/// `lo + 2^32·hi mod p` — executes too (the at most 16 packed-equal
/// digests). Plain, the partner is refused.
#[test]
fn the_packed_cap_compare_admits_the_node_and_its_partners_only() {
    use super::merkle_cap::CapCells;
    use super::word::base_word;
    use crate::tables::types::GoldilocksField;
    use crypto::merkle_tree::backends::types::BatchBlake3StdBackend;
    use crypto::merkle_tree::traits::IsMerkleTreeBackend;
    type Std = BatchBlake3StdBackend<GoldilocksField>;
    let mut nodes: Vec<[u8; 32]> = (0..4u8)
        .map(|n| core::array::from_fn(|i| (i as u8).wrapping_mul(29) ^ n.wrapping_mul(0x47) ^ 0x3C))
        .collect();
    // Node 2's first lane pair (lanes 0 and 1: lo, hi) is (a, 0): a partner exists.
    nodes[2][0..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
    nodes[2][4..8].copy_from_slice(&0u32.to_le_bytes());
    let l = Std::hash_new_parent(&nodes[0], &nodes[1]);
    let r = Std::hash_new_parent(&nodes[2], &nodes[3]);
    let root = Std::hash_new_parent(&l, &r);
    use super::merkle_cap::{CAP_MUX_OVERRIDE, CapMux};
    let run = |mode: CapMux, idx: u64, leaf: [u8; 32]| -> bool {
        let mut b = builder();
        // Words: four nodes, the leaf, the root (two words each), the index.
        let arena = b.declare_arena(13);
        let digest = |b: &mut LfmBuilder, at: u32| {
            WrapDigest::from_pair(b.hint_word(arena, at), b.hint_word(arena, at + 1))
        };
        let cap: Vec<WrapDigest> = (0..4).map(|n| digest(&mut b, 2 * n)).collect();
        let leaf_d = digest(&mut b, 8);
        let root_d = digest(&mut b, 10);
        let root_lanes = [b.unpack(root_d[0]), b.unpack(root_d[1])];
        CAP_MUX_OVERRIDE.with(|c| c.set(Some(mode)));
        let cells = CapCells::authenticate(&mut b, &cap, &root_lanes);
        let index = b.hint_felt(arena, 12);
        let bits = b.bit_dec(index, 2);
        cells.verify_path(&mut b, leaf_d, &bits, &[]);
        CAP_MUX_OVERRIDE.with(|c| c.set(None));
        let program = compile(b.finish());
        validate(&program).expect("admissible");
        let mut words: Vec<LfmWord> = Vec::new();
        for node in &nodes {
            words.extend(digest_words(node));
        }
        words.extend(digest_words(&leaf));
        words.extend(digest_words(&root));
        words.push(base_word(FE::from(idx)));
        execute(&program, &[words], &crate::hash_pin::BLOCK_HASHER).is_ok()
    };
    under_both(|_lowering| {
        for packed in [CapMux::Plain, CapMux::Packed, CapMux::Indexed] {
            for idx in 0..4u64 {
                assert!(
                    run(packed, idx, nodes[idx as usize]),
                    "{packed:?}: node {idx} itself"
                );
                for lane in 0..8 {
                    let mut off = nodes[idx as usize];
                    off[4 * lane] ^= 0x01;
                    assert!(
                        !run(packed, idx, off),
                        "{packed:?}: node {idx} with lane {lane} off"
                    );
                }
            }
            assert!(
                !run(packed, 1, nodes[2]),
                "{packed:?}: another node's digest"
            );
            // Node 2's partner: lane pair 0 from (a, 0) to (a + 1, 2^32 − 1).
            let mut partner = nodes[2];
            partner[0..4].copy_from_slice(&0x1234_5679u32.to_le_bytes());
            partner[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
            assert_eq!(
                run(packed, 2, partner),
                packed != CapMux::Plain,
                "the partner digest passes the packed compare only ({packed:?})"
            );
        }
    });
}

/// The cap mux's knob: unset and `packed` are the packed mux (the default),
/// `plain` the rollback, and anything else is refused.
#[test]
fn the_cap_mux_knob_reads_its_words_and_refuses_the_rest() {
    use super::merkle_cap::{CapMux, cap_mux, cap_mux_setting};
    assert_eq!(
        cap_mux(),
        CapMux::Packed,
        "no test sets the knob: the default is packed"
    );
    for (word, want) in [
        (None, CapMux::Packed),
        (Some("plain"), CapMux::Plain),
        (Some(" packed "), CapMux::Packed),
        (Some("packed"), CapMux::Packed),
        (Some("indexed"), CapMux::Indexed),
    ] {
        assert_eq!(cap_mux_setting(word), Ok(want), "{word:?}");
    }
    for word in ["", "on", "1", "Packed", "pack", "index", "Indexed"] {
        assert!(cap_mux_setting(Some(word)).is_err(), "{word:?}");
    }
}

/// ★ THE INDEXED CAP READ GATE. Every BLAKE3 chain and group gate again with
/// the cap checked by an indexed read of a lookup table of its packed nodes
/// (`LAMBDA_VM_LFM_CAP_MUX=indexed`): under the lean lowering the reads replace
/// the mux, under `LFM_BLAKE3` the packed mux stands in. Honest proofs execute,
/// every tamper has no execution, the compressions are the closed form's.
#[test]
fn every_blake3_gate_holds_under_the_indexed_cap_read() {
    with_cap_mux(super::merkle_cap::CapMux::Indexed, || {
        chain::a_blake3_chain_executes_on_a_proof_the_host_accepts();
        chain::a_blake3_chain_emits_its_compression_closed_form();
        chain::a_tampered_blake3_chain_cannot_execute();
        chain::a_lean_blake3_chain_checks_its_tail_once_a_leaf();
        chain::a_blake3_chain_with_a_final_polynomial_executes_and_refuses_its_tampers();
        stacked::a_blake3_group_opening_executes_on_proofs_the_host_accepts();
        stacked::a_tampered_joint_blake3_leg_cannot_execute();
    });
}

/// The indexed read is the mode the gate above ran: in a lean program a
/// capped chain emits one cap table a capped tree and one read an opening into
/// it, in place of the packed mux's `Select` tree (fewer select-type rows, the
/// same compressions); outside the lean lowering it emits no table and is the
/// packed mux.
#[test]
fn the_indexed_cap_read_replaces_the_mux_in_a_lean_program() {
    use super::merkle_cap::CapMux;
    use super::whir_chain::ChainShape;
    use multilinear::whir_chain::{CapPolicy, LeafOrder};
    let cfg = chain::config(4, 0, CapPolicy::Fixed(4), LeafOrder::Coset, Some(6));
    let shape = ChainShape::for_hash(&cfg, 12, WrapHash::Blake3);
    let count =
        |p: &LfmProgram, f: &dyn Fn(&Instr) -> bool| p.instrs.iter().filter(|i| f(i)).count();
    let selects = |i: &Instr| matches!(i, Instr::Select { .. } | Instr::Mux { .. });
    let tables = |i: &Instr| matches!(i, Instr::CapTable(_));
    let reads = |i: &Instr| matches!(i, Instr::CapRead { .. });
    under_both(|lowering| {
        let (mut packed, mut indexed) = (None, None);
        with_cap_mux(CapMux::Packed, || packed = Some(chain::program(&shape)));
        with_cap_mux(CapMux::Indexed, || indexed = Some(chain::program(&shape)));
        let (packed, indexed) = (packed.expect("packed"), indexed.expect("indexed"));
        assert_eq!(
            hash_rows(&packed),
            hash_rows(&indexed),
            "{lowering}: the read hashes nothing"
        );
        let capped_trees = shape.caps.iter().filter(|&&c| c > 0).count();
        println!(
            "INDEXED CAP READ {lowering}: caps {:?} · select-type rows {} -> {} · cap tables {} · reads {}",
            shape.caps,
            count(&packed, &selects),
            count(&indexed, &selects),
            count(&indexed, &tables),
            count(&indexed, &reads)
        );
        if LEAN.with(std::cell::Cell::get) {
            assert_eq!(
                count(&indexed, &tables),
                capped_trees,
                "one table a capped tree"
            );
            assert!(
                count(&indexed, &reads) > 0,
                "the openings read their tables"
            );
            assert!(
                count(&indexed, &selects) < count(&packed, &selects),
                "the reads replace the mux"
            );
        } else {
            assert_eq!(
                count(&indexed, &tables) + count(&indexed, &reads),
                0,
                "no lean table outside the lean lowering"
            );
            assert_eq!(
                indexed.instrs.len(),
                packed.instrs.len(),
                "the packed mux stands in"
            );
        }
    });
}

// ---------------------------------------------------------------------------
// Today's chain programs, pinned across the round form's arrival.
// ---------------------------------------------------------------------------

/// The digest of a program's emission: its instructions, address count,
/// arena lengths and public words, in order.
fn emission_digest(program: &LfmProgram) -> [u8; 32] {
    use digest::Digest;
    let text = format!(
        "{:?}|{}|{:?}|{}",
        program.instrs, program.num_addrs, program.arena_schema.lens, program.public_len
    );
    <Blake3Std as Digest>::digest(text.as_bytes()).into()
}

/// ★ TODAY'S FORM EMITS WHAT IT EMITTED. Every chain program the suites above
/// gate under today's form (`RoundForm::Fri`) — the BLAKE3 chain cases under
/// both lowerings, the lean tail chain, the final-polynomial chains, the
/// grouped and joint group openings, and the RPX chain at the cost shapes —
/// hashed instruction for instruction into one digest, pinned at the value
/// the emitter gave before the round form's in-guest verifier existed. Shape
/// only: no proof is read but the group openings' layouts.
#[test]
fn todays_chain_programs_are_byte_identical() {
    use multilinear::whir_chain::{
        CapPolicy, ChainConfig, ChainFormat, FinalPoly, FirstFold, GrindBits, LeafOrder, WhirFolds,
    };
    let mut all = Vec::new();
    let mut count = 0usize;
    let mut push = |program: &LfmProgram| {
        all.extend_from_slice(&emission_digest(program));
        count += 1;
    };
    under_both(|_| {
        for (num_vars, cfg) in chain::cases() {
            push(&chain::program(&super::whir_chain::ChainShape::for_hash(
                &cfg,
                num_vars,
                WrapHash::Blake3,
            )));
        }
        let mut tail = chain::config(400, 0, CapPolicy::Fixed(4), LeafOrder::Coset, Some(6));
        tail.log_folding = 1;
        push(&chain::program(&super::whir_chain::ChainShape {
            query: super::whir_chain::LeafQuery::Lean,
            ..super::whir_chain::ChainShape::for_hash(&tail, 9, WrapHash::Blake3)
        }));
        for (num_vars, v) in [(10usize, 2u8), (12, 4), (12, 2)] {
            let mut cfg = chain::config(5, 4, CapPolicy::Fixed(2), LeafOrder::Coset, Some(6));
            cfg.format.final_poly = FinalPoly::vars(v).expect("a final width");
            push(&chain::program(&super::whir_chain::ChainShape::for_hash(
                &cfg,
                num_vars,
                WrapHash::Blake3,
            )));
        }
        for (_, case) in stacked::cases() {
            push(&stacked::program(&stacked::build(&case)));
        }
    });
    let rpx = |q: usize, g: u8, k0: Option<usize>| ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: q,
        grind: GrindBits::uniform(g),
        format: ChainFormat {
            folds: k0.map_or(WhirFolds::Uniform, |k| {
                WhirFolds::First(FirstFold::new(k).expect("a tested first fold"))
            }),
            ..ChainFormat::DEFAULT
        },
    };
    for (n, q, g, k0) in [
        (6usize, 3usize, 0u8, None),
        (6, 5, 0, None),
        (5, 3, 0, None),
        (6, 3, 8, None),
        (9, 3, 8, None),
        (9, 3, 8, Some(6)),
        (11, 3, 8, Some(5)),
        (7, 3, 8, Some(6)),
    ] {
        push(&super::whir_chain_tests::chain_program(
            &super::whir_chain::ChainShape::new(&rpx(q, g, k0), n),
        ));
    }
    let digest: [u8; 32] = {
        use digest::Digest;
        <Blake3Std as Digest>::digest(&all).into()
    };
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    println!("TODAY'S CHAIN PROGRAMS: {count} programs · digest {hex}");
    assert_eq!(count, TODAYS_CHAIN_PROGRAMS);
    assert_eq!(hex, TODAYS_CHAIN_DIGEST, "today's chain emission moved");
}

/// [`todays_chain_programs_are_byte_identical`]'s programs.
const TODAYS_CHAIN_PROGRAMS: usize = 120;

/// [`todays_chain_programs_are_byte_identical`]'s digest, as the emitter gave
/// it before the round form's in-guest verifier (the same test run against
/// that emitter printed this value).
const TODAYS_CHAIN_DIGEST: &str =
    "5a3d66026ae7e08355567afb26735fa516a5f8dbbf68b6cd99e9965484901c0c";

// ---------------------------------------------------------------------------
// The WHIR round form's in-guest verifier, against the host.
// ---------------------------------------------------------------------------

mod round_form;
