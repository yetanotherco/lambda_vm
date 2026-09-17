//! Sequence KATs for the WHIR transcript the LFM replay has to reproduce.
//!
//! [`crypto::fiat_shamir::transcript_hash`]'s own tests pin what an RPX squeeze
//! IS — unreversed, four canonical felts, consumed in whole groups, one
//! candidate per coordinate. They say nothing about ORDER, because order is not
//! a property of the sponge; it is a property of the verifier that drives it.
//! The replay's failure mode is a sequence that differs from the host's by one
//! absorb or one draw, which every one of those tests passes.
//!
//! So these are scripts. A [`Script`] is a list of [`Step`]s, running one
//! produces the values it sampled and the state it left, and a vector is a
//! script beside what it produced. When the emitter exists it replays the same
//! scripts and is checked against the same vectors — the host is Rust driving a
//! byte sponge and the emitter is a straight-line field machine, so agreement
//! between them is a comparison of two constructions rather than a round trip
//! through shared code.
//!
//! # ⚠ What these vectors are, and what they are not
//!
//! They are **RECORDINGS**, not a specification. The LFM's other transcript
//! KATs ([`super::transcript_kats`]) come from an oracle written before any
//! Rust existed, which is why they can say the implementation is right. These
//! cannot: there is no second implementation of this sponge to take them from.
//!
//! What they catch is a transcript order that CHANGES — any edit to the
//! verifier's sequence moves a vector, and an emitter that drifts from the host
//! fails against one. What they cannot catch is the host and the emitter being
//! wrong together, i.e. an order that was never right. That is guarded by prose
//! instead: the sequence is written out in `V1-sizing-note.md` part (4) and the
//! preprocessed root's position in W1-B's design note §8, and the vectors here
//! are what makes an unintended divergence from those loud.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::transcript_hash::RpxTranscriptHash;

use crate::tables::types::{FE, FEE, GoldilocksExtension};

type Transcript = DefaultTranscript<GoldilocksExtension, RpxTranscriptHash>;

/// One step of a transcript script.
///
/// Field elements are named by a seed rather than written out, so a script
/// stays readable as data and a vector's inputs cannot silently disagree with
/// its outputs.
#[derive(Clone, Copy, Debug)]
pub enum Step {
    /// `append_bytes` of `n` bytes, filled deterministically from `seed`.
    AbsorbBytes { seed: u8, len: usize },
    /// `append_field_element` of the element `seed` names.
    AbsorbExt { seed: u64 },
    /// `sample_field_element`, recorded.
    SampleExt,
    /// `sample_u64(bound)`, recorded.
    SampleU64 { bound: u64 },
}

/// A named sequence of steps.
pub struct Script {
    pub name: &'static str,
    pub steps: &'static [Step],
}

/// What running a script produced.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observed {
    /// Every sampled extension element, in order.
    pub ext: Vec<FEE>,
    /// Every sampled `u64`, in order.
    pub u64s: Vec<u64>,
    /// The transcript state the script left.
    pub state: [u8; 32],
}

fn bytes_from(seed: u8, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| seed.wrapping_mul(31).wrapping_add(i as u8))
        .collect()
}

fn ext_from(seed: u64) -> FEE {
    FEE::new([
        FE::from(seed),
        FE::from(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        FE::from(seed.wrapping_add(0x1234_5678_9ABC_DEF0)),
    ])
}

/// Runs a script against the production transcript type.
pub fn run(script: &Script) -> Observed {
    let mut t = Transcript::new(&[]);
    let mut ext = Vec::new();
    let mut u64s = Vec::new();
    for step in script.steps {
        match *step {
            Step::AbsorbBytes { seed, len } => t.append_bytes(&bytes_from(seed, len)),
            Step::AbsorbExt { seed } => t.append_field_element(&ext_from(seed)),
            Step::SampleExt => ext.push(t.sample_field_element()),
            Step::SampleU64 { bound } => u64s.push(t.sample_u64(bound)),
        }
    }
    Observed {
        ext,
        u64s,
        state: t.state(),
    }
}
