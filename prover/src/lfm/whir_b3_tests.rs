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

fn builder() -> LfmBuilder {
    LfmBuilder::new().with_wrap_hash(WrapHash::Blake3)
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

fn publish_digest(b: &mut LfmBuilder, d: WrapDigest) {
    for cell in d.iter() {
        b.public(*cell);
    }
}

/// Instructions of each hash kind a program emits: (BLAKE3 rows, other hash
/// rows). The BLAKE3 arm must emit no felt-sponge row at all.
fn hash_rows(program: &LfmProgram) -> (usize, usize) {
    let blake3 = program
        .instrs
        .iter()
        .filter(|i| matches!(i, Instr::Blake3(_)))
        .count();
    let other = program
        .instrs
        .iter()
        .filter(|i| matches!(i, Instr::Hash { .. } | Instr::Hash16(_) | Instr::KeccakF(_)))
        .count();
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
}

/// A root whose two hinted words are swapped is a different 32 bytes: the
/// challenge after it moves. (The words are the digest's byte order, not a set.)
#[test]
fn a_root_absorbed_with_its_words_swapped_moves_the_challenge() {
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

/// ★ THE GRIND GATE AND ITS TAMPER ARM. At several factors the machine accepts
/// the host's own smallest valid nonce, leaves the host's transcript after
/// absorbing it, and refuses to execute on every nonce
/// `is_valid_nonce::<Blake3Std>` rejects.
#[test]
fn the_blake3_grind_accepts_what_the_host_accepts_and_refuses_the_rest() {
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
}

/// `bits == 0` emits nothing at all, as on the host (no state read, no
/// nonce absorb): the program equals one with no grind call.
#[test]
fn a_zero_bit_blake3_grind_touches_nothing() {
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
}

/// The grind's state seed is the transcript's, so an absorb the host did not
/// make before the grind refuses the host's nonce (the seed moved).
#[test]
fn a_grind_over_another_state_refuses_the_hosts_nonce() {
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

    use super::super::builder::Ext;
    use super::super::compiler::{LfmProgram, compile};
    use super::super::edsl::{WrapDigest, WrapHash};
    use super::super::executor::execute;
    use super::super::validator::validate;
    use super::super::whir_chain::{
        ChainShape, RoundStorage, byte_digest_words, emit_verify_weighted, push_round_words,
        round_words,
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
        let mut b = builder();
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
    fn a_blake3_chain_executes_on_a_proof_the_host_accepts() {
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
    fn a_tampered_blake3_chain_cannot_execute() {
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
    }

    /// A round-0 value hinted with a nonzero upper lane is refused before any
    /// hash: the leaf's `BitDec` reads it as a base word, so the lift into the
    /// fold is pinned (whir_open's obligation, under BLAKE3).
    #[test]
    fn a_round0_value_with_an_upper_lane_is_refused() {
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
    }

    /// The coset program and the natural one refuse each other's proofs (and
    /// so does the host): round 0's walks are wired to the order.
    #[test]
    fn a_coset_blake3_program_refuses_a_natural_proof() {
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
    }
}
