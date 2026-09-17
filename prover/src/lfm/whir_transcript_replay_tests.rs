//! The replay against the vectors — the machine on one side, the host on the
//! other.
//!
//! [`super::whir_transcript_kat_tests`] pins what the HOST produces for four
//! scripts (K1 order, K2 width and padding, K3 chain advance, K4 sampler
//! offset). These run the same scripts through the emitter and compare against
//! the host directly, so the two sides are a Rust byte sponge and a
//! straight-line field machine rather than two halves of one implementation.
//!
//! W1's position KAT — where the derived DECODE preprocessed root is absorbed —
//! is CITED, not duplicated: that is a property of the epoch program's order,
//! not of this object.

use crypto::hash::rpx::{digest_to_commitment, sponge_leaf_bytes};

use crate::tables::types::{FE, FEE};

use super::builder::{Bit, Ext, Felt, LfmBuilder};
use super::compiler::{LfmProgram, compile};
use super::executor::execute;
use super::validator::validate;
use super::whir_transcript::{
    WhirTranscript, absorb_unpack_rows, sample_ext_rows, sample_u64_rows, squeeze_rows, state_rows,
};
use super::whir_transcript_kats::{Observed, Script, Step, bytes_from, ext_from, run};
use super::word::{LfmWord, ext_word, word_as_ext};

/// `LFM_CONST` rows any program containing one leaf hash interns: the zero word
/// — which `felt_const(0)`, the empty digest and `pack_ext`'s lane 3 all share,
/// because interning is by canonical word — and the capacity word, which
/// carries the padding flag and so is one per distinct buffer length.
fn leaf_hash_consts(distinct_lengths: usize) -> usize {
    1 + distinct_lengths
}

fn builder() -> LfmBuilder {
    LfmBuilder::new().with_wrap_hash(super::edsl::WrapHash::production())
}

fn run_program(program: &LfmProgram, arenas: &[Vec<LfmWord>]) -> Vec<LfmWord> {
    let exec =
        execute(program, arenas, &crate::hash_pin::BLOCK_HASHER).expect("the replay must execute");
    exec.public_words.iter().map(|(_, word)| *word).collect()
}

// ---------------------------------------------------------------------------
// The gate: the scripts the host vectors pin, replayed by the machine.
// ---------------------------------------------------------------------------

/// Every `AbsorbExt` value of a script, HINTED rather than folded in as a
/// constant, so the runtime absorb path and its felt-alignment rule are what
/// run.
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
    rows: usize,
}

fn emit(script: &Script) -> Emitted {
    let inputs = runtime_inputs(script);
    let mut b = builder();
    let arena = b.declare_arena(inputs.len().max(1) as u32);
    let hinted: Vec<Ext> = (0..inputs.len())
        .map(|index| b.hint_word(arena, index as u32).as_ext())
        .collect();

    let mut transcript = WhirTranscript::new();
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
                assert_eq!(
                    bound,
                    1u64 << nbits,
                    "{}: the replay's bounded sampler is the power-of-two one",
                    script.name
                );
                u64_bits.push(nbits);
                drawn.push(transcript.sample_u64_pow2(&mut b, nbits));
            }
        }
    }
    let state = transcript.state(&mut b);

    for value in &sampled {
        b.public(value.as_cell());
    }
    for bits in &drawn {
        for bit in bits {
            b.public(bit.as_cell());
        }
    }
    b.public(state);

    let publics = sampled.len() + u64_bits.iter().sum::<usize>() + 1;
    let program = compile(b.finish());
    validate(&program).expect("the replay must be admissible");
    let rows = program.instrs.len() - inputs.len() - publics;
    Emitted {
        program,
        ext_count: sampled.len(),
        u64_bits,
        rows,
    }
}

fn arena_for(script: &Script) -> Vec<LfmWord> {
    let inputs = runtime_inputs(script);
    if inputs.is_empty() {
        return vec![[FE::zero(); 4]];
    }
    inputs.iter().map(ext_word).collect()
}

fn observe(script: &Script, emitted: &Emitted) -> Observed {
    let words = run_program(&emitted.program, &[arena_for(script)]);
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
                let lane = words[at][0];
                assert!(
                    lane == FE::zero() || lane == FE::one(),
                    "a published bit must be zero or one"
                );
                if lane == FE::one() {
                    value |= 1 << bit;
                }
                at += 1;
            }
            value
        })
        .collect();
    let digest = words[at];
    let state = digest_to_commitment(&[digest[0], digest[1], digest[2], digest[3]]);
    Observed { ext, u64s, state }
}

/// ★ THE GATE. Every script, three streams: the sampled extension elements, the
/// sampled `u64`s, and the transcript state left at the end.
///
/// The state is the strong one — a function of the whole buffer history — so a
/// wrong clear, a missing chain advance or a mis-packed felt moves it even when
/// every challenge happens to look plausible.
#[test]
fn the_replay_reproduces_what_the_host_transcript_produces() {
    for script in super::whir_transcript_kat_tests::scripts() {
        let emitted = emit(script);
        let machine = observe(script, &emitted);
        let host = run(script);
        assert_eq!(
            machine.ext, host.ext,
            "{}: the extension challenges must match the host's",
            script.name
        );
        assert_eq!(
            machine.u64s, host.u64s,
            "{}: the bounded draws must match the host's",
            script.name
        );
        assert_eq!(
            machine.state, host.state,
            "{}: the transcript state must match the host's",
            script.name
        );
        println!(
            "{:>6}: {:>2} challenges, {} bounded draws, {:>3} rows emitted",
            script.name,
            machine.ext.len(),
            machine.u64s.len(),
            emitted.rows
        );
    }
}

/// ★ K5, in the world where the reversal is gone: a squeeze's four lanes ARE
/// the leaf digest of the buffer, unreversed.
///
/// `RpxTranscriptHash::REVERSES_SQUEEZE` is `false`, so `sample()` hands back
/// `digest_to_commitment(sponge_leaf_bytes(buf))` as it stands. Pinned against
/// the host's own two functions at every length class mod 8 and mod 64.
#[test]
fn a_squeeze_is_the_unreversed_leaf_digest_of_the_buffer() {
    for len in [0usize, 1, 7, 8, 9, 63, 64, 65, 72] {
        let bytes = bytes_from(0x5A, len);
        let mut b = builder();
        let mut transcript = WhirTranscript::new();
        transcript.absorb_const_bytes(&bytes);
        for lane in transcript.squeeze(&mut b) {
            b.public(lane.as_cell());
        }
        let program = compile(b.finish());
        validate(&program).expect("admissible");
        let words = run_program(&program, &[]);

        let want = sponge_leaf_bytes(&bytes);
        let machine: [FE; 4] = core::array::from_fn(|i| words[i][0]);
        assert_eq!(
            machine, want,
            "{len} bytes: the squeeze's lanes must be the leaf digest's felts, unreversed"
        );
        assert_eq!(
            digest_to_commitment(&machine),
            digest_to_commitment(&want),
            "{len} bytes: and so must the commitment bytes a byte consumer would see"
        );
    }
}

/// ★ The alignment rule is ENFORCED, not remembered.
///
/// This is what "hold the first window" looks like in code: the emitter cannot
/// silently produce a byte shift, so a statement whose computed padding has not
/// landed fails loudly at emit time rather than producing a program that is
/// right on a fixture and wrong on a block.
#[test]
#[should_panic(expected = "must start on a felt boundary")]
fn a_runtime_absorb_at_an_unaligned_offset_refuses() {
    let mut b = builder();
    let arena = b.declare_arena(1);
    let value = b.hint_word(arena, 0).as_ext();
    let mut transcript = WhirTranscript::new();
    transcript.absorb_const_bytes(&[0u8; 5]);
    transcript.absorb_ext(&mut b, value);
}

// ---------------------------------------------------------------------------
// The cost pins: four purpose-built programs whose every row is named.
// ---------------------------------------------------------------------------

/// `n` hinted felts absorbed, then one squeeze, then the four lanes published.
fn squeeze_only(n: usize) -> LfmProgram {
    let mut b = builder();
    let arena = b.declare_arena(n.max(1) as u32);
    let felts: Vec<Felt> = (0..n)
        .map(|index| b.hint_felt(arena, index as u32))
        .collect();
    let mut transcript = WhirTranscript::new();
    transcript.absorb_felts(&mut b, &felts);
    for lane in transcript.squeeze(&mut b) {
        b.public(lane.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("admissible");
    program
}

/// ★ F1 for the squeeze, across the width classes: every row named.
///
/// `n` hints, the leaf hash's two interned constants, the squeeze itself, and
/// four publics. The three terms of `squeeze_rows` grow differently in `n` — one
/// `Pack` per four felts, one permutation per eight, one `Unpack` always — so a
/// sweep separates them rather than fixing only their sum.
#[test]
fn the_squeeze_costs_its_closed_form() {
    for n in [0usize, 1, 3, 4, 5, 8, 9, 16, 17] {
        let program = squeeze_only(n);
        let measured = program.instrs.len();
        let predicted = n + leaf_hash_consts(1) + squeeze_rows(n) + 4;
        println!("squeeze over {n:>2} felts: {measured:>3} rows emitted, {predicted:>3} predicted");
        assert_eq!(
            measured, predicted,
            "a squeeze over {n} felts must emit its closed form"
        );
    }
}

/// ★ F1 for the three one-row operations, each isolated in its own program.
///
/// An absorbed extension element is one `Unpack` and contributes three felts to
/// the buffer; a challenge is one `Pack` on top of the squeeze its candidates
/// force; a bounded draw is one `BitDec`, whose bits are the answer and need no
/// recomposition.
#[test]
fn the_one_row_operations_cost_one_row() {
    // An extension element absorbed, then squeezed: one Unpack, and a buffer of
    // three felts.
    let mut b = builder();
    let arena = b.declare_arena(1);
    let value = b.hint_word(arena, 0).as_ext();
    let mut transcript = WhirTranscript::new();
    transcript.absorb_ext(&mut b, value);
    for lane in transcript.squeeze(&mut b) {
        b.public(lane.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("admissible");
    assert_eq!(
        program.instrs.len(),
        1 + absorb_unpack_rows() + leaf_hash_consts(1) + squeeze_rows(3) + 4,
        "an absorbed extension element is one Unpack and three buffered felts"
    );

    // A challenge: the squeeze its three candidates force, plus one Pack.
    let mut b = builder();
    let arena = b.declare_arena(4);
    let felts: Vec<Felt> = (0..4).map(|index| b.hint_felt(arena, index)).collect();
    let mut transcript = WhirTranscript::new();
    transcript.absorb_felts(&mut b, &felts);
    let challenge = transcript.sample_ext(&mut b);
    b.public(challenge.as_cell());
    let program = compile(b.finish());
    validate(&program).expect("admissible");
    assert_eq!(
        program.instrs.len(),
        4 + leaf_hash_consts(1) + squeeze_rows(4) + sample_ext_rows() + 1,
        "a challenge is one Pack on top of its squeeze"
    );

    // A bounded draw: the same squeeze, plus one BitDec.
    let nbits = 20usize;
    let mut b = builder();
    let arena = b.declare_arena(4);
    let felts: Vec<Felt> = (0..4).map(|index| b.hint_felt(arena, index)).collect();
    let mut transcript = WhirTranscript::new();
    transcript.absorb_felts(&mut b, &felts);
    for bit in transcript.sample_u64_pow2(&mut b, nbits) {
        b.public(bit.as_cell());
    }
    let program = compile(b.finish());
    validate(&program).expect("admissible");
    assert_eq!(
        program.instrs.len(),
        4 + leaf_hash_consts(1) + squeeze_rows(4) + sample_u64_rows() + nbits,
        "a bounded draw is one BitDec, and its bits are the answer"
    );
}

/// ★ `state()` observes without advancing: the same hash, no digest `Unpack`,
/// and the buffer it leaves is the one a later squeeze hashes again.
#[test]
fn state_costs_the_hash_without_the_unpack() {
    for n in [1usize, 4, 9] {
        let mut b = builder();
        let arena = b.declare_arena(n as u32);
        let felts: Vec<Felt> = (0..n)
            .map(|index| b.hint_felt(arena, index as u32))
            .collect();
        let mut transcript = WhirTranscript::new();
        transcript.absorb_felts(&mut b, &felts);
        let observed = transcript.state(&mut b);
        b.public(observed);
        let program = compile(b.finish());
        validate(&program).expect("admissible");
        assert_eq!(
            program.instrs.len(),
            n + leaf_hash_consts(1) + state_rows(n) + 1,
            "state over {n} felts is the hash without the Unpack"
        );
        assert_eq!(
            transcript.out_pos(),
            super::whir_transcript::CANDIDATES_PER_SQUEEZE,
            "state must not fill the output buffer"
        );
        assert_eq!(transcript.squeezes(), 0, "state must not advance the chain");
    }
}
