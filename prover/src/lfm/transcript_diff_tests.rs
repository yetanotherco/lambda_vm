//! ★★★ **THE TRANSCRIPT ABSORB DIFFERENTIAL** — the machine's
//! [`TranscriptReplay`] against the host's [`AlgebraicTranscript`], step by
//! step.
//!
//! The machine re-derives the host's Fiat–Shamir chain. If the two absorb
//! anything differently the states part, every later challenge is a different
//! value, and the first thing that NOTICES is whatever assert consumes a
//! challenge — in a wrap leg that is the grinding check, failing as a
//! `DivByZero` that names neither the absorb nor the step.
//!
//! ⚠ **Why this compares STATE AFTER EACH STEP.** Comparing only the final
//! challenge says THAT the chains disagree; it cannot say which absorb did it,
//! and the absorbs have different fixes — a length prefix, a felt's byte order,
//! a root's cell count, a payload grouping. So both sides expose their state
//! after every step and the assertion names the first step at which they part.
//!
//! ⚠ **The host side is driven through the HOST's own API**, never a
//! reimplementation of it: a root goes in as `append_bytes(&commitment)`, a
//! felt as the bytes `FieldElement::stream_bytes` produces (production's own
//! serialisation, the one `algebraic_commit::element_felts` also goes through),
//! and an extension element as `append_field_element`. What this file chooses
//! is the script, not the encodings.
//!
//! Absorbs only, deliberately. The spine's first squeezed challenge is the
//! shared LogUp `z`, so everything that can move it is an absorb; squeezes have
//! their own asymmetries (`sample` is defined on the algebraic arm rather than
//! mirrored, per its doc) and gating them means deciding what they should mean,
//! which is not this gate's job.

use math::traits::AsBytes;

use crypto::fiat_shamir::is_transcript::IsTranscript;

use super::algebraic_commit::{
    AlgebraicHasher, PoseidonCommit, RpoCommit, RpxCommit, commitment_to_digest,
};
use super::algebraic_transcript::AlgebraicTranscript;
use super::builder::LfmBuilder;
use super::compiler::compile;
use super::edsl::WrapHash;
use super::executor::execute;
use super::transcript_replay::TranscriptReplay;
use super::word::{LfmWord, base_word, ext_word};
use crate::tables::types::{FE, FEE};

macro_rules! for_each_tenant {
    ($body:ident) => {
        $body::<RpoCommit>("Rpo");
        $body::<RpxCommit>("Rpx");
        $body::<PoseidonCommit>("Poseidon");
    };
}

/// A commitment whose bytes are distinctive in every position, so a regrouping
/// or a reversal is visible rather than accidentally symmetric.
fn probe_root() -> [u8; 32] {
    core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(3))
}

/// Production's own serialisation of a base element — the bytes the host
/// transcript absorbs for a felt. Through `AsBytes::stream_bytes` rather than
/// spelled out, for the reason `algebraic_commit::element_felts` gives.
fn felt_bytes(v: FE) -> Vec<u8> {
    let mut out = Vec::new();
    v.stream_bytes(&mut |bytes| out.extend_from_slice(bytes));
    out
}

/// ★★★ The gate: every absorb the batched spine uses, machine against host,
/// compared after each step.
#[test]
fn the_machine_transcript_tracks_the_host_absorb_for_absorb() {
    fn check<H: AlgebraicHasher>(tenant: &str) {
        let root = probe_root();
        let root_word = commitment_to_digest(&root);
        let felt_v = FE::from(0x0123_4567_89ab_cdefu64);
        let ext_v = FEE::new([FE::from(11u64), FE::from(22), FE::from(33)]);

        // ---- the host chain, driven through the host's own API ----
        let mut host = AlgebraicTranscript::new(H::KIND);
        let mut want: Vec<LfmWord> = Vec::new();
        let mut steps: Vec<String> = Vec::new();
        fn record(
            host: &AlgebraicTranscript,
            want: &mut Vec<LfmWord>,
            steps: &mut Vec<String>,
            label: &str,
        ) {
            want.push(host.state_word());
            steps.push(label.to_string());
        }

        // `TranscriptReplay::new(seed)` absorbs the seed as its first append,
        // and `AlgebraicTranscript::with_seed` is `new` plus that same call.
        host.append_bytes(&[]);
        record(&host, &mut want, &mut steps, "seed (empty)");

        // Const byte strings across the cell-grouping boundary in both
        // directions — 32 bytes is one payload cell, so 31/32/33 straddle it,
        // and the empty string is the degenerate case a length prefix has to
        // separate from every other.
        let consts: Vec<Vec<u8>> = vec![
            vec![],
            vec![0xa5],
            (0..7u8).collect(),
            (0..8u8).collect(),
            (0..9u8).collect(),
            (0..31u8).collect(),
            (0..32u8).collect(),
            (0..33u8).collect(),
            (0..40u8).collect(),
        ];
        for c in &consts {
            host.append_bytes(c);
            record(
                &host,
                &mut want,
                &mut steps,
                &format!("const bytes len {}", c.len()),
            );
        }

        host.append_bytes(&root);
        record(&host, &mut want, &mut steps, "root (32 bytes)");

        host.append_bytes(&felt_bytes(felt_v));
        record(&host, &mut want, &mut steps, "felt");

        host.append_field_element(&ext_v);
        record(&host, &mut want, &mut steps, "ext");

        // ★ The PHASE A path: a root absorbed through the halves family rather
        // than through `append_root_cells`. `replay_phase_a` takes its roots as
        // `&[Felt]` and calls `append_halves_misaligned`, whose declared byte
        // length is `4 · halves.len()` — so what it is handed must be the eight
        // halves of the root's THIRTY-TWO bytes on both arms. Handing it
        // `RootCells::lanes_flat` gives four FULL FELTS on the algebraic arm,
        // declaring sixteen bytes where the host declared thirty-two.
        host.append_bytes(&root);
        record(
            &host,
            &mut want,
            &mut steps,
            "root via the Phase A halves path",
        );

        // ★ SQUEEZES. `sample_ext` is where every shared challenge comes from —
        // the LogUp pair, every beta, every z, every gamma — so the first
        // challenge a spine draws is one of these. Its host counterpart is
        // unambiguous (`sample_field_element`, one squeezed cell read as lanes
        // 0-2), unlike `sample()`, which the algebraic arm DEFINES rather than
        // mirrors and which is therefore still out of scope here.
        //
        // Both the drawn VALUE and the state after it, because they fail
        // differently: a wrong value with a right state is a read of the wrong
        // lanes, a right value with a wrong state is a wrong advance, and only
        // the second corrupts everything downstream.
        for k in 0..3 {
            let e = host.sample_field_element();
            want.push(ext_word(&e));
            steps.push(format!("sample_ext {k} — the drawn value"));
            record(
                &host,
                &mut want,
                &mut steps,
                &format!("sample_ext {k} — the state after"),
            );
        }

        // ---- the machine chain, same script ----
        let mut b = LfmBuilder::new().with_wrap_hash(WrapHash::Algebraic);
        let arena = b.declare_arena(3);
        let c_root = b.hint_word(arena, 0);
        let c_felt = b.hint_felt(arena, 1);
        let c_ext = b.hint_word(arena, 2);
        let ext_lanes = b.unpack(c_ext);

        let mut t = TranscriptReplay::new(&[]);
        let publish = |b: &mut LfmBuilder, t: &mut TranscriptReplay| {
            let s = t.state(b);
            assert_eq!(s.len(), 1, "{tenant}: an algebraic state is ONE cell");
            b.public(s[0]);
        };
        publish(&mut b, &mut t);
        for c in &consts {
            t.append_const_bytes(c);
            publish(&mut b, &mut t);
        }
        t.append_root_cells(&mut b, &[c_root]);
        publish(&mut b, &mut t);
        t.append_felt(&mut b, c_felt);
        publish(&mut b, &mut t);
        t.append_ext(&mut b, [ext_lanes[0], ext_lanes[1], ext_lanes[2]]);
        publish(&mut b, &mut t);

        let root_arena = b.declare_arena(super::epoch::RootCells::words_per_root(&b));
        let root_cells = super::epoch::RootCells::hint(&mut b, root_arena, 0);
        let phase_a_halves = root_cells.halves(&mut b);
        t.append_halves_misaligned(&phase_a_halves);
        publish(&mut b, &mut t);

        for _ in 0..3 {
            let e = t.sample_ext(&mut b);
            b.public(e.as_cell());
            publish(&mut b, &mut t);
        }

        let program = compile(b.finish());
        let mut arenas = vec![vec![root_word, base_word(felt_v), ext_word(&ext_v)]];
        // ⚠ The ALGEBRAIC form directly, NOT `proof_arena::commitment_words`.
        // That helper reads `WrapHash::production()` — the workspace PIN — while
        // this program's reader is `RootCells::words_per_root`, which reads THIS
        // BUILDER's arm, and the builder is unconditionally `Algebraic` here.
        // The two coincide only on an algebraic pin, so mixing them makes the
        // gate pass on one branch and fail on another for a reason that has
        // nothing to do with what it tests.
        arenas.push(vec![commitment_to_digest(&root)]);
        let exec = execute(&program, &arenas, &H::KIND).expect("the transcript program executes");

        assert_eq!(
            exec.public_words.len(),
            want.len(),
            "{tenant}: the two chains must have the same number of steps"
        );
        for (i, (got, expect)) in exec.public_words.iter().zip(&want).enumerate() {
            assert_eq!(
                &got.1, expect,
                "{tenant}: transcript states part at step {i} ({}) — \
                 every later challenge is a different value",
                steps[i]
            );
        }
    }
    for_each_tenant!(check);
}

/// ⚠ The differential's own control: `state()` must actually TRACK the chain,
/// or a gate comparing a constant against a constant would pass. Absorbing
/// anything must move the state.
#[test]
fn the_state_moves_on_every_absorb() {
    fn check<H: AlgebraicHasher>(tenant: &str) {
        let mut host = AlgebraicTranscript::new(H::KIND);
        let mut seen = vec![host.state_word()];
        for c in [vec![], vec![1u8], (0..32u8).collect::<Vec<_>>()] {
            host.append_bytes(&c);
            let s = host.state_word();
            assert!(
                !seen.contains(&s),
                "{tenant}: absorbing must move the state to a fresh value"
            );
            seen.push(s);
        }
    }
    for_each_tenant!(check);
}
