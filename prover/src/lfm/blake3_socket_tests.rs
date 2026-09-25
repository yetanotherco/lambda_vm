//! The BLAKE3 arm of `LFM_HASH`: its framing, its layout, its degree bound,
//! what it accepts, what it rejects, and the prove+verify that turns a
//! predicted cell count into a measured one.
//!
//! ## What pins what
//!
//! Three layers, and they are deliberately not the same evidence:
//!
//! 1. **The primitive** is pinned elsewhere, to the `blake3` crate:
//!    `blake3::tests::seven_rounds_is_the_blake3_crate`. Nothing here re-checks
//!    the G function or the message schedule.
//! 2. **The framing** — the six choices between "a correct `f`" and "a correct
//!    2-to-1 compress" — is pinned here by [`SOCKET_VECTORS`], which came from
//!    two independent generators, plus one negative control per choice. A right
//!    constant inside a wrong framing is the normal way this goes wrong, and
//!    every primitive test stays green while it happens.
//! 3. **The chip** is pinned here too: that `NUM_CONSTRAINTS` constraints over
//!    `MAIN_COLUMNS` value columns say exactly what that framing says, and that
//!    they say it inside a real proof produced by the production prover.
//!
//! ## What this suite cannot see
//!
//! It says nothing about the machine's DEFAULT hash, which is still
//! `TestPermutation`; every test constructs the BLAKE3 configuration
//! explicitly. It covers the two two-to-one modes — `compress` and the
//! `transcript` step, which are the same socket under different domain tags —
//! and no `permute`, because option B1 settled that no permute socket is ever
//! built. See `blake3_socket`'s module docs.
//!
//! The transcript's own vectors and its end-to-end behaviour live in
//! `transcript_tests`; what is here is the CHIP side of it — the mode-selected
//! tag, and the M1–M7 controls the transcript spec pre-committed.

use math::field::element::FieldElement;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintSet, ProverEvalFolder, RootKind, check_dense_index_set,
    num_base_from_meta,
};
use stark::frame::Frame;
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::table::TableView;
use stark::trace::TraceTable;
use stark::traits::TransitionEvaluationContext;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField, VmTable};

use super::airs::lfm_chip_census_with_hasher;
use super::blake3::{BLAKE3_IV, BLAKE3_MSG_PERMUTATION};
use super::blake3_socket::{
    self, BLOCK_LEN_LFMC, Blake3Permutation, COUNTER_LFMC, FLAGS_LFMC, MAIN_COLUMNS,
    NUM_CONSTRAINTS, NUM_G, SOCKET_ROUNDS, TAG_LFMC, cols, lanes_of, socket_digest,
    socket_digest_rounds, word_of,
};
use super::blake3_socket_kats::SOCKET_VECTORS;
use super::builder::{Cell, LfmBuilder, LfmProgramSource};
use super::chips::hash::{self, HashConstraints};
use super::compiler::{LfmProgram, compile};
use super::executor::{LfmExecError, execute};
use super::hash::{HASH_STATE_FELTS, HasherKind, LfmHasher};
use super::instr::HashMode;
use super::programs::{permute_coverage_program, trivial_program};
use super::proof::{lfm_prove_with_hasher, prove_traces_with_hasher, verify_against};
use super::registry::{build_artifacts, build_artifacts_with_hasher};
use super::trace::build_traces_with_hasher;
use super::word::LfmWord;

type F = GoldilocksField;
type E = GoldilocksExtension;

const KIND: HasherKind = HasherKind::Blake3;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("blowup=2 is valid")
}

// =========================================================================
// The closed-form budget, written out for BOTH round counts
// =========================================================================

/// Value columns per compression, as a function of the round count.
///
/// Written as a formula over named blocks rather than taken from the layout,
/// because a closed form taken from the code under test would agree with any
/// layout, including a wrong one. 28 shared prefix + `4·12` lane bytes +
/// `8·rounds` G-blocks of 60 + 16 digest bytes + 8 leaf-canonicity witnesses
/// (`Z`/`GINV` per felt — present on EVERY row, since a chip has one width).
///
/// ★ The lane count is spelled `4 + 2·4` — the accumulator cell plus one felt
/// cell's halves — rather than read from `cols::NUM_LANES`, for exactly the
/// reason the rest is spelled out: taking it from the layout would make this
/// agree with ANY widening instead of with the one the RATE specifies. Note
/// what does NOT move with it: the canonicity witnesses stay at 8, because the
/// accumulator is a previous digest and needs byte decomposition but no
/// canonicity gate. That is the claim that made the RATE cost +16 columns
/// rather than +24.
const fn predicted_main(rounds: usize) -> usize {
    28 + 4 * (4 + 2 * 4) + 60 * (8 * rounds) + 16 + 8
}

/// Bus interactions per compression: the frozen six `LfmMem` tuples, four
/// `ByteAlu[XOR]` per XOR word (`4·8·rounds` mixing words + 4 feed-forward),
/// four `AreBytes` per rotation (`2·8·rounds` of them), and two lane `AreBytes`
/// per lane.
const fn predicted_interactions(rounds: usize) -> usize {
    6 + 4 * (4 * (8 * rounds) + 4) + 4 * (2 * (8 * rounds)) + 2 * (4 + 2 * 4)
}

/// `main + 3·aux` with `aux = ceil(interactions / 2)` — `airs.rs`'s census
/// formula, the same instrument that produced the keccak, Poseidon and
/// standalone-blake columns, so all four are comparable by construction.
///
/// `pub(super)` so [`super::blake3_probe`] can state the socket-vs-standalone
/// comparison against THIS number instead of a transcription of it. The copy it
/// carried had drifted by 8 — it predated the leaf mode's canonicity block — and
/// a hand-copied cost figure is exactly the kind that rots in silence, because
/// nothing recomputes it.
pub(super) const fn predicted_cells(rounds: usize) -> usize {
    predicted_main(rounds) + 3 * predicted_interactions(rounds).div_ceil(2)
}

const fn predicted_constraints(rounds: usize) -> usize {
    50 + 16 * (8 * rounds)
}

/// The whole budget, at both round counts, as literals.
///
/// These are the numbers the report carries and the A6R decision is priced
/// against, so they are written out rather than left as an expression: the
/// arithmetic and the layout are two statements, and a test is only worth
/// having if they can disagree.
#[test]
fn the_socket_budget_is_the_predicted_one_at_both_round_counts() {
    // 6 rounds — the A6R variant.
    assert_eq!(predicted_main(6), 2_980);
    assert_eq!(predicted_interactions(6), 1_198);
    assert_eq!(predicted_interactions(6).div_ceil(2), 599);
    assert_eq!(predicted_cells(6), 4_777);
    assert_eq!(predicted_constraints(6), 818);

    // 7 rounds — standard BLAKE3, the default.
    assert_eq!(predicted_main(7), 3_460);
    assert_eq!(predicted_interactions(7), 1_390);
    assert_eq!(predicted_interactions(7).div_ceil(2), 695);
    assert_eq!(predicted_cells(7), 5_545);
    assert_eq!(predicted_constraints(7), 946);

    // ★★ WHAT THE LEAF RATE COST, priced here rather than asserted anywhere
    // else: +16 main columns (four lanes × four bytes), +8 bus interactions
    // (two `AreBytes` per new lane) and +28 census cells at 7 rounds, against a
    // 2.0× cut in leaf absorption — which is ~70% of a recursion tower node's
    // bill (COMMIT.md §1.4.1). The pre-RATE figures were 3,444 / 1,382 / 5,517.
    // The CONSTRAINT count did not move at all, and that is not luck: the lane
    // identities gained four while the unread-`IN` pins lost four, which is
    // precisely why a hand-numbered framing block could have overwritten the
    // pins in silence (§1.4.4 H1).
    assert_eq!(predicted_main(7) - (28 + 32 + 60 * 56 + 16 + 8), 16);
    assert_eq!(predicted_cells(7) - 5_517, 28);
    assert_eq!(predicted_constraints(7), 946, "unchanged by the widening");

    // ★ The A6R price, on this socket: going 6 → 7 rounds costs +16.07% per
    // compression. The plan's paper estimate for the syscall-shaped chip was
    // +15.5%; the socket pays slightly more because its constant framing makes
    // the round-INDEPENDENT part smaller, so the rounds are a larger share.
    assert_eq!(
        (predicted_cells(7) - predicted_cells(6)) * 10_000 / predicted_cells(6),
        1_607,
        "hundredths of a percent"
    );

    // Both are BELOW the standalone chip's measured 4,946, which is the point of
    // hosting: constant `h`/`t`/`block_len`/`flags`, a constant message tail, and a
    // truncation window that never builds twelve of the sixteen output words.
    assert!(predicted_cells(6) < 4_946);
}

/// The compiled arm IS the prediction at the round count it was compiled for.
#[test]
fn the_built_layout_matches_the_prediction() {
    // The single-knob invariant, asserted where it can actually fail: the
    // socket and the standalone probe must be priced at the SAME round count.
    // `NUM_G == 8 * SOCKET_ROUNDS` below is internally consistent either way,
    // so it cannot see the two chips drifting apart; this can.
    assert_eq!(SOCKET_ROUNDS, super::blake3::BLAKE3_ROUNDS);
    assert_eq!(
        NUM_G,
        super::blake3_chip::NUM_G,
        "the socket arm and the standalone LFM_BLAKE3 probe must be compiled \
         for the same round count, or the probe prices a hash the machine does \
         not use"
    );
    assert_eq!(NUM_G, 8 * SOCKET_ROUNDS);
    assert_eq!(MAIN_COLUMNS, predicted_main(SOCKET_ROUNDS));
    assert_eq!(
        hash::num_columns(KIND) - cols::PREP_WIDTH,
        predicted_main(SOCKET_ROUNDS)
    );
    assert_eq!(
        hash::bus_interactions(KIND).len(),
        predicted_interactions(SOCKET_ROUNDS)
    );
    assert_eq!(NUM_CONSTRAINTS, predicted_constraints(SOCKET_ROUNDS));
    // 12 since option B1 added `MODE_T` (was 11). The prefix is the hasher-
    // independent instruction group, so this number is the same under every
    // candidate — `poseidon_chip_tests` pins the identical value, and the two
    // together are what would catch one arm's layout drifting from the other's.
    assert_eq!(
        cols::PREP_WIDTH,
        13,
        "the preprocessed prefix does not move"
    );
    assert_eq!(cols::LANES, 41, "the shared value prefix is not reflowed");
}

/// The layout is injective and gapless — no column written twice, none unread.
///
/// The width alone cannot see an off-by-one inside `lane_byte`/`g_base`/
/// `out_byte`: two blocks could overlap and the total still come out right.
#[test]
fn the_layout_assigns_every_column_exactly_once() {
    let mut seen = vec![0usize; cols::NUM_COLUMNS];
    let mut claim = |c: usize| seen[c] += 1;
    for i in 0..HASH_STATE_FELTS {
        claim(cols::IN0 + i);
        claim(cols::OUT0 + i);
    }
    for k in 0..4 {
        claim(cols::S8 + k);
    }
    for lane in 0..cols::NUM_LANES {
        for b in 0..4 {
            claim(cols::lane_byte(lane, b));
        }
    }
    for g in 0..NUM_G {
        for off in 0..cols::G_SIZE {
            claim(cols::g_base(g) + off);
        }
    }
    for i in 0..4 {
        for b in 0..4 {
            claim(cols::out_byte(i, b));
        }
    }
    for i in 0..blake3_socket::FELTS_PER_LEAF {
        claim(cols::canon_z(i));
        claim(cols::canon_ginv(i));
    }
    for (c, &n) in seen.iter().enumerate().skip(cols::PREP_WIDTH) {
        assert_eq!(
            n, 1,
            "value column {c} is claimed {n} times, want exactly 1"
        );
    }
    for (c, &n) in seen.iter().enumerate().take(cols::PREP_WIDTH) {
        assert_eq!(n, 0, "preprocessed column {c} must not be claimed");
    }
}

/// The census reports the arm at its real width and its real interaction count,
/// so the hash-matrix instrument prices BLAKE3 rather than a stale Test column.
#[test]
fn the_census_prices_the_blake3_arm() {
    let opts = options();
    let program = compress_program();
    let census = lfm_chip_census_with_hasher(&program, KIND);
    let hash_chip = census
        .iter()
        .find(|c| c.name == "LFM_HASH")
        .expect("LFM_HASH is in the census");
    assert_eq!(hash_chip.main_cols, predicted_main(SOCKET_ROUNDS));
    assert_eq!(
        hash_chip.aux_cols,
        predicted_interactions(SOCKET_ROUNDS).div_ceil(2)
    );
    assert_eq!(
        hash_chip.main_cols + 3 * hash_chip.aux_cols,
        predicted_cells(SOCKET_ROUNDS),
        "base-field-equivalent cells per compression row"
    );
    let _ = opts;
}

// =========================================================================
// The framing — SOCKET.md §2, and the controls that make it discriminating
// =========================================================================

/// Every framing degree of freedom, in one object, so a negative control can
/// break exactly one at a time and nothing else.
#[derive(Clone, Copy)]
struct Framing {
    rounds: usize,
    cv: [u32; 8],
    tag_word: u32,
    counter: u64,
    block_len: u32,
    flags: u32,
    a_slot: usize,
    b_slot: usize,
    tag_slot: usize,
    out_window: usize,
    lane_le: bool,
    msg_permutation: [usize; 16],
}

const HONEST: Framing = Framing {
    rounds: SOCKET_ROUNDS,
    cv: BLAKE3_IV,
    tag_word: TAG_LFMC,
    counter: COUNTER_LFMC,
    block_len: BLOCK_LEN_LFMC,
    flags: FLAGS_LFMC,
    a_slot: 0,
    b_slot: 4,
    // Straight after the twelve lanes, not at a fixed 8: the tag is the last
    // word of the message under every lane count, which is what keeps the byte
    // string `LE32(lanes) ‖ tag` (COMMIT.md §1.2).
    tag_slot: 12,
    out_window: 0,
    lane_le: true,
    msg_permutation: BLAKE3_MSG_PERMUTATION,
};

/// The message words under a framing. A big-endian lane serialisation changes
/// the message WORDS, because a word is read little-endian from the bytes.
fn framed_message(a: &[u32; 4], b: &[u32; 4], fr: Framing) -> [u32; 16] {
    let lane = |v: u32| if fr.lane_le { v } else { v.swap_bytes() };
    let mut m = [0u32; 16];
    for i in 0..4 {
        m[fr.a_slot + i] = lane(a[i]);
        m[fr.b_slot + i] = lane(b[i]);
    }
    m[fr.tag_slot] = fr.tag_word;
    m
}

/// A deliberately *parameterised* socket compress, used only to build negative
/// controls: the same dataflow with [`Framing`] as an input.
///
/// It is NOT what `socket_digest_rounds` calls. Keeping the two apart costs a
/// duplicated loop and buys the thing the controls are for — they compare
/// against [`SOCKET_VECTORS`], constants that came from outside this file, so
/// they stay meaningful no matter how the real function is later refactored.
fn framed_digest(a: &[u32; 4], b: &[u32; 4], fr: Framing) -> [u32; 4] {
    let g = |s: &mut [u32; 16], ia: usize, ib: usize, ic: usize, id: usize, mx: u32, my: u32| {
        s[ia] = s[ia].wrapping_add(s[ib]).wrapping_add(mx);
        s[id] = (s[id] ^ s[ia]).rotate_right(16);
        s[ic] = s[ic].wrapping_add(s[id]);
        s[ib] = (s[ib] ^ s[ic]).rotate_right(12);
        s[ia] = s[ia].wrapping_add(s[ib]).wrapping_add(my);
        s[id] = (s[id] ^ s[ia]).rotate_right(8);
        s[ic] = s[ic].wrapping_add(s[id]);
        s[ib] = (s[ib] ^ s[ic]).rotate_right(7);
    };
    let mut v: [u32; 16] = [
        fr.cv[0],
        fr.cv[1],
        fr.cv[2],
        fr.cv[3],
        fr.cv[4],
        fr.cv[5],
        fr.cv[6],
        fr.cv[7],
        BLAKE3_IV[0],
        BLAKE3_IV[1],
        BLAKE3_IV[2],
        BLAKE3_IV[3],
        fr.counter as u32,
        (fr.counter >> 32) as u32,
        fr.block_len,
        fr.flags,
    ];
    let mut m = framed_message(a, b, fr);
    for r in 0..fr.rounds {
        g(&mut v, 0, 4, 8, 12, m[0], m[1]);
        g(&mut v, 1, 5, 9, 13, m[2], m[3]);
        g(&mut v, 2, 6, 10, 14, m[4], m[5]);
        g(&mut v, 3, 7, 11, 15, m[6], m[7]);
        g(&mut v, 0, 5, 10, 15, m[8], m[9]);
        g(&mut v, 1, 6, 11, 12, m[10], m[11]);
        g(&mut v, 2, 7, 8, 13, m[12], m[13]);
        g(&mut v, 3, 4, 9, 14, m[14], m[15]);
        if r < fr.rounds - 1 {
            let prev = m;
            for (i, &p) in fr.msg_permutation.iter().enumerate() {
                m[i] = prev[p];
            }
        }
    }
    let w = fr.out_window;
    core::array::from_fn(|i| v[w + i] ^ v[w + i + 8])
}

/// Everything `f` actually sees under a framing: the initial state, the message
/// schedule at *every* round, and the output window.
///
/// Two framings with equal traces compute equal digests *necessarily*, so a
/// control whose trace equals the honest one on some input is genuinely
/// INAPPLICABLE there rather than undetected — which is what lets the control
/// suite assert "changes the digest" unconditionally everywhere else. Deriving
/// applicability this way rather than hand-listing it is deliberate: a
/// hand-list goes stale as controls are added, and a stale entry is a control
/// that looks covered and is not.
///
/// The schedules, not the permutation, are what belong here. `a_one` is the
/// case that proves it: its message has `m[2] = m[6] = 0`, so transposing the
/// first two entries of the permutation produces the identical schedule and the
/// control cannot possibly fire.
fn effective_trace(
    a: &[u32; 4],
    b: &[u32; 4],
    fr: Framing,
) -> (usize, [u32; 8], u64, u32, u32, usize, Vec<[u32; 16]>) {
    let mut sched = framed_message(a, b, fr);
    let mut scheds = Vec::with_capacity(fr.rounds);
    for r in 0..fr.rounds {
        scheds.push(sched);
        if r < fr.rounds - 1 {
            let prev = sched;
            for (i, &p) in fr.msg_permutation.iter().enumerate() {
                sched[i] = prev[p];
            }
        }
    }
    (
        fr.rounds,
        fr.cv,
        fr.counter,
        fr.block_len,
        fr.flags,
        fr.out_window,
        scheds,
    )
}

/// ★ **The socket KATs.** The real function reproduces every vector at both
/// round counts.
#[test]
fn the_socket_matches_the_vectors_at_both_round_counts() {
    for v in SOCKET_VECTORS.iter() {
        assert_eq!(
            socket_digest_rounds(&v.a, &v.b, 6),
            v.digest_6,
            "6-round socket vector {}",
            v.name
        );
        assert_eq!(
            socket_digest_rounds(&v.a, &v.b, 7),
            v.digest_7,
            "7-round socket vector {}",
            v.name
        );
    }
    // And the compiled-in round count is one of the two, reaching the vectors
    // through the entry point the chip and the host actually call.
    for v in SOCKET_VECTORS.iter() {
        let expected = if SOCKET_ROUNDS == 7 {
            v.digest_7
        } else {
            v.digest_6
        };
        assert_eq!(socket_digest(&v.a, &v.b), expected, "vector {}", v.name);
    }
}

/// ★ **The external anchor, direct.** At 7 rounds the socket is literally
/// `blake3::hash(a ‖ b ‖ "LFMC")` truncated to 16 bytes — a library call, no
/// oracle, no JSON.
///
/// This is what SOCKET.md §6 lists as ✗ DEFERRED ("the same equality against
/// the Rust `blake3` crate — needs cargo"). It also re-derives the 52-byte
/// message from the byte-level specification rather than from
/// `socket_message`, so the word-level and byte-level forms are two statements
/// that can disagree.
///
/// ★ **The anchor is what the leaf RATE had to keep.** The message grew by the
/// third input cell's four zero lanes — 36 bytes to 52 — and 52 is still under
/// 64, so a row is still ONE block and still a plain library call. Carrying the
/// leaf's accumulator in the chaining value `h` instead would have made the row
/// a chunk CONTINUATION and thrown this test away for the same rate.
#[test]
fn seven_rounds_is_blake3_of_the_domain_separated_message() {
    for v in SOCKET_VECTORS.iter() {
        let mut msg = Vec::with_capacity(52);
        for lane in v.a.iter().chain(v.b.iter()) {
            msg.extend_from_slice(&lane.to_le_bytes());
        }
        // The third input cell: unread by a digest mode, and pinned to zero by
        // the unread-`IN` pins, so its four lanes are zero on every honest row.
        msg.extend_from_slice(&[0u8; 16]);
        msg.extend_from_slice(b"LFMC");
        assert_eq!(msg.len(), 52, "the socket message is one 52-byte block");
        assert!(msg.len() < 64, "and one block is what the anchor needs");

        let full = blake3::hash(&msg);
        let want: [u32; 4] = core::array::from_fn(|i| {
            u32::from_le_bytes(full.as_bytes()[4 * i..4 * i + 4].try_into().unwrap())
        });
        assert_eq!(
            socket_digest_rounds(&v.a, &v.b, 7),
            want,
            "7-round socket vector {} must be blake3::hash of its message",
            v.name
        );
        assert_eq!(want, v.digest_7, "the table itself agrees with the crate");
    }
}

/// The parameterised control, at canonical parameters, IS the real function —
/// so every control below differs in exactly the one choice it names.
#[test]
fn the_framing_variant_at_canonical_parameters_is_the_socket() {
    for v in SOCKET_VECTORS.iter() {
        assert_eq!(
            framed_digest(&v.a, &v.b, HONEST),
            socket_digest(&v.a, &v.b),
            "control harness must reproduce the socket at canonical parameters"
        );
    }
}

/// NEGATIVE CONTROL, one per framing degree of freedom.
///
/// Without this, "the vectors pass" would be evidence only that the vectors are
/// *reachable*, not that they discriminate — and framing is precisely where a
/// correct `f` still gives a wrong hash. Each control must change the digest on
/// every vector where its effective trace differs from the honest one, and must
/// discriminate on at least one vector overall.
#[test]
fn breaking_one_framing_choice_at_a_time_breaks_the_digest() {
    let mut transposed = [0usize; 16];
    for (i, &p) in BLAKE3_MSG_PERMUTATION.iter().enumerate() {
        transposed[p] = i;
    }
    let controls: [(&str, Framing); 14] = [
        (
            "swap_a_b",
            Framing {
                a_slot: 4,
                b_slot: 0,
                ..HONEST
            },
        ),
        (
            "tag_changed",
            Framing {
                tag_word: u32::from_le_bytes(*b"LFMP"),
                ..HONEST
            },
        ),
        (
            "tag_omitted",
            Framing {
                tag_word: 0,
                ..HONEST
            },
        ),
        (
            // 8 — where the tag sat before the socket widened to twelve lanes,
            // so this control is also the discrimination between the two
            // framings: a chip that widened the lanes and left the tag behind
            // would compute this, and it is a different hash.
            "tag_slot_moved",
            Framing {
                tag_slot: 8,
                ..HONEST
            },
        ),
        (
            "truncate_high_half",
            Framing {
                out_window: 4,
                ..HONEST
            },
        ),
        ("flags_parent", Framing { flags: 4, ..HONEST }),
        ("flags_no_root", Framing { flags: 3, ..HONEST }),
        (
            "block_len_64",
            Framing {
                block_len: 64,
                ..HONEST
            },
        ),
        (
            "block_len_32",
            Framing {
                block_len: 32,
                ..HONEST
            },
        ),
        (
            "counter_one",
            Framing {
                counter: 1,
                ..HONEST
            },
        ),
        (
            "cv_zero",
            Framing {
                cv: [0; 8],
                ..HONEST
            },
        ),
        (
            "lanes_big_endian",
            Framing {
                lane_le: false,
                ..HONEST
            },
        ),
        (
            "msg_perm_swapped",
            Framing {
                msg_permutation: {
                    let mut p = BLAKE3_MSG_PERMUTATION;
                    p.swap(0, 1);
                    p
                },
                ..HONEST
            },
        ),
        (
            "other_round_count",
            Framing {
                rounds: if SOCKET_ROUNDS == 7 { 6 } else { 7 },
                ..HONEST
            },
        ),
    ];

    for (what, fr) in controls {
        let mut discriminated = 0;
        for v in SOCKET_VECTORS.iter() {
            let honest = socket_digest(&v.a, &v.b);
            if effective_trace(&v.a, &v.b, fr) == effective_trace(&v.a, &v.b, HONEST) {
                // Provably inapplicable on this input — `swap_a_b` when a == b,
                // `lanes_big_endian` when every lane is a byte-palindrome,
                // `msg_perm_swapped` when the two transposed slots hold equal
                // words. Asserted as an equality, not skipped: an inapplicable
                // control must produce the SAME digest, which is a check in its
                // own right on the applicability derivation.
                assert_eq!(framed_digest(&v.a, &v.b, fr), honest);
                continue;
            }
            assert_ne!(
                framed_digest(&v.a, &v.b, fr),
                honest,
                "{what} still reproduces the digest on vector {} — the vectors do not pin it",
                v.name
            );
            discriminated += 1;
        }
        assert!(
            discriminated > 0,
            "{what} is discriminated by no vector at all"
        );
    }
}

/// The transposed message permutation is a real permutation and a different
/// one — otherwise `msg_perm_swapped` above would be testing nothing.
#[test]
fn the_message_permutation_control_is_a_different_permutation() {
    let mut p = BLAKE3_MSG_PERMUTATION;
    p.swap(0, 1);
    assert_ne!(p, BLAKE3_MSG_PERMUTATION);
    let mut sorted = p;
    sorted.sort_unstable();
    assert_eq!(sorted, core::array::from_fn::<usize, 16, _>(|i| i));
}

// =========================================================================
// The lane boundary — obligation O1, host side
// =========================================================================

/// ★ **O1, host side.** An out-of-range lane is REJECTED, never reduced.
///
/// `edsl::merkle_walk` feeds `compress` arena-hinted — prover-chosen — sibling
/// cells, and a lane is a Goldilocks felt over `[0, p)`. The chip can only
/// commit a byte decomposition for a lane below `2^32`, so a host that reduced
/// instead of rejecting would claim a digest no proof can produce.
#[test]
fn an_out_of_range_lane_is_rejected_rather_than_reduced() {
    let ok: LfmWord = word_of(&[1, 2, 3, 4]);
    assert_eq!(lanes_of(&ok), Some([1, 2, 3, 4]));

    // The alias that would exist under silent reduction.
    let aliased: LfmWord = [
        FE::from(1u64 + (1u64 << 32)),
        FE::from(2u64),
        FE::from(3u64),
        FE::from(4u64),
    ];
    assert_eq!(lanes_of(&aliased), None, "2^32 + 1 is not a u32 lane");

    let mut state = [FE::zero(); HASH_STATE_FELTS];
    state[0..4].copy_from_slice(&aliased);
    state[4..8].copy_from_slice(&ok);
    assert!(
        Blake3Permutation
            .admits(HashMode::Compress, &state)
            .is_err(),
        "a non-u32 lane must be refused by admits"
    );

    // HONEST CONTROL: the in-range pair is still accepted. Without it, this
    // test would pass equally if `admits` rejected everything.
    let mut good = [FE::zero(); HASH_STATE_FELTS];
    good[0..4].copy_from_slice(&ok);
    good[4..8].copy_from_slice(&ok);
    assert!(Blake3Permutation.admits(HashMode::Compress, &good).is_ok());
}

/// The whole-machine version of the same thing: an arena word with a non-`u32`
/// lane makes the program fail to execute, with a reason.
#[test]
fn a_non_u32_arena_word_fails_execution_under_blake3() {
    let program = compress_program();
    let mut bad = arenas();
    bad[0][0][0] = FE::from(1u64 << 32);
    assert!(
        matches!(
            execute(&program, &bad, &KIND),
            Err(LfmExecError::HasherRejected(_))
        ),
        "a non-u32 hinted lane must be rejected at execution"
    );
    // HONEST CONTROL.
    assert!(execute(&program, &arenas(), &KIND).is_ok());
}

/// Obligation O2: the socket is closed on its own output, so a digest fed back
/// in as a sibling always satisfies O1. That is why only leaf digests and
/// prover-hinted siblings need the input check.
#[test]
fn the_socket_output_is_always_a_valid_input() {
    for v in SOCKET_VECTORS.iter() {
        let d = socket_digest(&v.a, &v.b);
        assert_eq!(
            lanes_of(&word_of(&d)),
            Some(d),
            "a socket digest must round-trip as a u32-lane cell"
        );
    }
}

/// Obligation O3: the IV enters through `h`, not through the capacity lanes, so
/// this arm overrides `compress` rather than inheriting permute-and-truncate.
///
/// Asserted through `HasherKind`'s dispatch, because that is the path the
/// executor takes and a candidate whose override was not honoured there would
/// prove one thing and record another.
#[test]
fn compress_is_overridden_and_the_upper_out_lanes_are_empty() {
    let a = word_of(&[0x0102_0304, 0, 0, 0]);
    let b = word_of(&[0, 0, 0, 0x0506_0708]);
    let via_kind = LfmHasher::compress(&KIND, &a, &b);
    assert_eq!(
        via_kind,
        word_of(&socket_digest(
            &[0x0102_0304, 0, 0, 0],
            &[0, 0, 0, 0x0506_0708]
        ))
    );

    let out = LfmHasher::compress_out(&KIND, &a, &b);
    assert_eq!(&out[0..4], &via_kind[..]);
    for (j, felt) in out.iter().enumerate().skip(4) {
        assert_eq!(
            *felt,
            FE::zero(),
            "OUT lane {j} carries nothing on a compress row"
        );
    }

    // `compress_iv` is meaningful if read, and is NOT what the framing uses.
    assert_eq!(
        LfmHasher::compress_iv(&KIND),
        word_of(&[BLAKE3_IV[0], BLAKE3_IV[1], BLAKE3_IV[2], BLAKE3_IV[3]])
    );
}

/// ✗ There is no permute socket, and the refusal is explicit rather than a
/// wrong answer. `permute_coverage_program` contains one, so it is unprovable
/// under BLAKE3 — which is the settled state of option B1, not a defect.
///
/// The program under test used to be `trivial_program`, which no longer has a
/// permute in it: B1 gave the registry's entries up to the real hash, and this
/// test moved to the unregistered fixture that took over permute coverage.
#[test]
fn a_permute_row_is_refused_under_blake3() {
    assert!(
        Blake3Permutation
            .admits(HashMode::Permute, &[FE::zero(); HASH_STATE_FELTS])
            .is_err()
    );
    assert!(
        matches!(
            execute(&permute_coverage_program(), &permute_arenas(), &KIND),
            Err(LfmExecError::HasherRejected(_))
        ),
        "a program containing a permute must be refused under BLAKE3"
    );
    // HONEST CONTROL: the same program executes fine under the hashers that do
    // have a permute socket, so the refusal is BLAKE3's domain and not a break.
    assert!(
        execute(
            &permute_coverage_program(),
            &permute_arenas(),
            &HasherKind::Test
        )
        .is_ok()
    );
}

/// ★ The F3.4-retirement milestone at the registry level: `TrivialV0` — a
/// REGISTERED program — now executes under BLAKE3, which it could not while it
/// held a raw permute.
///
/// Its arena has to be `u32`-laned (obligation O1); that is the socket's domain,
/// not a property of this program.
#[test]
fn the_trivial_program_runs_under_blake3_now_that_it_has_no_permute() {
    assert!(
        !trivial_program().instrs.iter().any(
            |i| matches!(i, super::instr::Instr::Hash { mode, .. } if *mode == HashMode::Permute)
        ),
        "a registered program must not contain a permute"
    );
    let arenas = vec![
        (0..4u32)
            .map(|i| word_of(&[0x1000_0000 * (i + 1), 0x0BAD_F00D ^ i, i, 0xFFFF_FFFF - i]))
            .collect(),
    ];
    execute(&trivial_program(), &arenas, &KIND).expect("TrivialV0 executes under BLAKE3");
    // HONEST CONTROL: still fine under the default hasher, on its own arenas.
    execute(&trivial_program(), &trivial_arenas(), &HasherKind::Test).expect("and under Test");
}

// =========================================================================
// The constraints
// =========================================================================

/// Every constraint index is emitted exactly once, the count is the one the
/// module documents, and the degree really reaches — and does not exceed — 3.
#[test]
fn the_arm_emits_its_constraints_at_degree_3() {
    let set = HashConstraints::BLAKE3;
    assert_eq!(HashConstraints::num_constraints(KIND), NUM_CONSTRAINTS);

    let meta = ConstraintSet::<F, E>::meta(&set);
    assert_eq!(meta.len(), NUM_CONSTRAINTS, "constraints emitted");
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i, "meta must be dense and idx-ordered");
        assert_eq!(m.kind, RootKind::Base, "every hash constraint is base");
    }

    let mut cb = CaptureBuilder::<F, E>::new();
    set.eval(&mut cb);
    let (_prog, degrees) = cb.finish(num_base_from_meta(&meta));
    assert_eq!(degrees.len(), NUM_CONSTRAINTS, "one emit per constraint");

    let declared = ConstraintSet::<F, E>::max_degree(&set);
    assert_eq!(declared, 3, "the wrap's blowup 2 depends on this staying 3");
    for &(idx, measured) in &degrees {
        assert!(
            measured <= declared,
            "constraint {idx}: measured degree {measured} EXCEEDS declared {declared}"
        );
    }
    // Not merely `<=`: the mu-gated carry booleanities really are cubic, so a
    // set that quietly topped out at 2 would mean the carries had stopped being
    // constrained.
    assert_eq!(degrees.iter().map(|&(_, d)| d).max(), Some(3));
}

/// The mode column a row in `mode` sets.
pub(super) fn mode_col(mode: HashMode) -> usize {
    match mode {
        HashMode::Compress => cols::MODE_C,
        HashMode::Transcript => cols::MODE_T,
        HashMode::Leaf => cols::MODE_L,
        HashMode::Permute => cols::MODE_P,
    }
}

/// A hash row in `mode`, exactly as `trace::build_traces_with_hasher` fills
/// one.
fn hash_row_mode(mode: HashMode, a: [u32; 4], b: [u32; 4]) -> Vec<FE> {
    let tag = blake3_socket::tag_for_mode(mode).expect("BLAKE3 has a socket for this mode");
    let mut row = vec![FE::zero(); cols::NUM_COLUMNS];
    row[mode_col(mode)] = FE::one();
    row[cols::IN0..cols::IN0 + 4].copy_from_slice(&word_of(&a));
    row[cols::IN0 + 4..cols::IN0 + 8].copy_from_slice(&word_of(&b));
    for (k, iv) in BLAKE3_IV.iter().take(4).enumerate() {
        row[cols::S8 + k] = FE::from(u64::from(*iv));
    }
    let digest = blake3_socket::socket_digest_rounds_tagged(&a, &b, SOCKET_ROUNDS, tag);
    row[cols::OUT0..cols::OUT0 + 4].copy_from_slice(&word_of(&digest));
    blake3_socket::fill_socket_witness_tagged(&mut row, tag);
    row
}

/// A `Compress` row — the shape most of these tests are about.
fn hash_row(a: [u32; 4], b: [u32; 4]) -> Vec<FE> {
    hash_row_mode(HashMode::Compress, a, b)
}

fn evaluate(row: &[FE]) -> Vec<FE> {
    let set = HashConstraints::BLAKE3;
    let n = ConstraintSet::<F, E>::meta(&set).len();
    let no_ch: Vec<FieldElement<E>> = vec![];
    let offset = FieldElement::<E>::zero();
    let frame = Frame::<F, E>::new(vec![TableView::new(vec![row.to_vec()], vec![vec![]])]);
    let ctx =
        TransitionEvaluationContext::new_prover(frame.as_row_frame(), &no_ch, &no_ch, &offset);
    let mut base_out = vec![FE::zero(); n];
    let mut ext_out = vec![FieldElement::<E>::zero(); n];
    let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
    set.eval(&mut folder);
    folder.assert_all_emitted();
    base_out
}

pub(super) fn violations(row: &[FE]) -> Vec<usize> {
    evaluate(row)
        .iter()
        .enumerate()
        .filter(|(_, v)| **v != FE::zero())
        .map(|(i, _)| i)
        .collect()
}

/// An honest row satisfies every constraint, and the digest it carries is the
/// KAT's — so the constraint set and the vectors agree about the same row.
#[test]
fn an_honest_row_satisfies_every_constraint() {
    for v in SOCKET_VECTORS.iter() {
        let row = hash_row(v.a, v.b);
        assert_eq!(violations(&row), Vec::<usize>::new(), "vector {}", v.name);
        let want = if SOCKET_ROUNDS == 7 {
            v.digest_7
        } else {
            v.digest_6
        };
        for (i, lane) in want.iter().enumerate() {
            assert_eq!(row[cols::OUT0 + i], FE::from(u64::from(*lane)));
            for byte in 0..4 {
                assert_eq!(
                    row[cols::out_byte(i, byte)],
                    FE::from(u64::from((lane >> (8 * byte)) as u8)),
                    "digest byte ({i}, {byte}) of vector {}",
                    v.name
                );
            }
        }
    }
}

/// An all-zero padding row satisfies the set, and a row claiming to be real
/// with no witness does not.
///
/// The second half is what stops the first from being vacuous: a constraint set
/// that accepted anything would pass the padding check just as well.
#[test]
fn padding_is_satisfied_and_a_real_marked_empty_row_is_not() {
    assert_eq!(
        violations(&vec![FE::zero(); cols::NUM_COLUMNS]),
        Vec::<usize>::new(),
        "an all-zero padding row must satisfy every constraint"
    );

    let mut row = vec![FE::zero(); cols::NUM_COLUMNS];
    row[cols::MODE_C] = FE::one();
    assert!(
        !violations(&row).is_empty(),
        "a real-marked row with an all-zero witness must be rejected"
    );
}

/// The `MODE_P = 0` pin: a permute-marked row is rejected by the AIR itself,
/// independently of the executor's refusal.
#[test]
fn a_permute_marked_row_violates_the_air() {
    let mut row = vec![FE::zero(); cols::NUM_COLUMNS];
    row[cols::MODE_P] = FE::one();
    assert!(
        !violations(&row).is_empty(),
        "MODE_P = 1 must be unsatisfiable under the BLAKE3 arm"
    );
}

/// ★ **The lane-decomposition constraint bites.** Retagging a lane's bytes to a
/// different value, or the lane felt to `v + 2^32`, must violate the AIR.
///
/// The second case is the identity's own job: a lane moved without its bytes.
/// See `the_lane_range_check_is_load_bearing_on_its_own` for the other half —
/// the witness this identity cannot see, which is what the `AreBytes` sends
/// are for.
#[test]
fn the_lane_decomposition_binds_the_felt_to_its_bytes() {
    let base = hash_row([0x1234_5678, 1, 2, 3], [4, 5, 6, 7]);

    let mut tampered = base.clone();
    tampered[cols::lane_byte(0, 0)] += FE::one();
    assert!(
        !violations(&tampered).is_empty(),
        "moving a lane byte must violate the decomposition"
    );

    let mut aliased = base.clone();
    aliased[cols::IN0] += FE::from(1u64 << 32);
    assert!(
        !violations(&aliased).is_empty(),
        "lane + 2^32 must violate the decomposition — this IS obligation O1"
    );

    // HONEST CONTROL.
    assert_eq!(violations(&base), Vec::<usize>::new());
}

/// ★ **O1's OTHER half — the one the rest of this suite never exercises.**
///
/// The lane contract is an eval constraint AND two `AreBytes` sends, and the
/// module comment says "NEITHER ALONE SUFFICES". Every other control here
/// breaks the linear identity, which the identity alone catches — so until this
/// test existed, the `AreBytes` half was asserted in prose and exercised
/// nowhere.
///
/// The witness that separates them moves `2^8` from one byte column into the
/// next (`MB[0] += 256`, `MB[1] -= 1`). The weighted sum is unchanged
/// **exactly**, over the field, with no borrow — `256·(b1 − 1) + (b0 + 256) =
/// 256·b1 + b0` — so the lane identity passes, the message word the mixing core
/// reads is the same linear form and therefore also unchanged, and the honest
/// digest still comes out. Nothing in the eval set is wrong with the row. The
/// only defect is that `MB[0]` is no longer a byte, and only the range check
/// can see that.
///
/// Recorded because it is not what I expected and it sharpens the argument:
/// **a carry-absorbing witness cannot be silent, because the lane bytes ARE the
/// message bytes.** Trying to alias a lane to `v + 2^32` and letting `MB[3]`
/// absorb the carry does satisfy the lane identity — and then breaks the mixing
/// core instead, because the word the core hashes moved by `2^32` too. So the
/// alias is caught either way; what the range check uniquely buys is the case
/// where the *sum* is preserved.
///
/// And that case is not a curiosity — it is the whole attack surface. The
/// message words reach `add3` and nothing else (never an XOR), so these sends
/// are their ONLY bound. A sum-preserving witness is exactly the door to an
/// unbounded `m`, and `add3`'s exactness in round 0 — constant `a` and `b`, a
/// byte-bounded `s` — is what an unbounded `m` breaks: the prover solves for
/// any `s` it likes and owns the compression.
#[test]
fn the_lane_range_check_is_load_bearing_on_its_own() {
    let base = hash_row([0x1234_5678, 1, 2, 3], [4, 5, 6, 7]);
    /// Constraint index of input lane 0's decomposition (idx 6–13 are the eight
    /// lanes); `CORE_IDX` is 26, so anything below it is framing.
    const LANE0: usize = 6;

    // (a) THE ONE ONLY `AreBytes` CATCHES. Identity preserved, core preserved,
    // eval set entirely silent. If this half ever starts failing, the proof
    // below has stopped testing the range check and has become a duplicate of
    // `the_lane_decomposition_binds_the_felt_to_its_bytes`.
    let mut shifted = base.clone();
    shifted[cols::lane_byte(0, 0)] += FE::from(256u64);
    shifted[cols::lane_byte(0, 1)] = shifted[cols::lane_byte(0, 1)] - FE::one();
    assert_eq!(
        violations(&shifted),
        Vec::<usize>::new(),
        "the linear identity alone cannot see a byte column carrying 2^8 — \
         which is exactly why the AreBytes sends are not optional"
    );

    // (b) The naive alias: claim `v + 2^32` and leave the bytes alone. The
    // IDENTITY catches this one, at lane 0's own index.
    let mut naive = base.clone();
    naive[cols::IN0] += FE::from(1u64 << 32);
    assert!(
        violations(&naive).contains(&LANE0),
        "a lane moved without its bytes must violate its own decomposition"
    );

    // (c) The alias with the carry absorbed, which is the interesting one: the
    // lane identity is satisfied — `LANE0` is NOT among the violations — and the
    // MIXING CORE rejects instead, because `MB[3]` is a message byte and the
    // word being hashed moved by 2^32 as well.
    let mut absorbed = base.clone();
    absorbed[cols::IN0] += FE::from(1u64 << 32);
    absorbed[cols::lane_byte(0, 3)] += FE::from(256u64);
    let v = violations(&absorbed);
    assert!(
        !v.contains(&LANE0),
        "absorbing the carry must satisfy the lane identity — otherwise this \
         case is not demonstrating what it claims"
    );
    assert!(
        v.iter().all(|&i| i >= 26) && !v.is_empty(),
        "and the mixing core must reject it instead, got {v:?}"
    );

    // (d) In a real proof, the `AreBytes` send catches (a). Only the byte
    // shuffle is used: it leaves `IN0` untouched, so the `LfmMem` receive token
    // is unchanged and the rejection can only come from the range check, not
    // from a memory-bus mismatch.
    assert_not_accepted("a byte column carrying 2^8, identity preserved", |t| {
        let b0 = t.main_table.get_row(0)[cols::lane_byte(0, 0)];
        let b1 = t.main_table.get_row(0)[cols::lane_byte(0, 1)];
        t.main_table
            .set_fe(0, cols::lane_byte(0, 0), b0 + FE::from(256u64));
        t.main_table
            .set_fe(0, cols::lane_byte(0, 1), b1 - FE::one());
    });
}

/// The digest recomposition binds `OUT` to the mixing core's output bytes.
#[test]
fn the_digest_recomposition_binds_out_to_the_core() {
    let base = hash_row([9, 8, 7, 6], [5, 4, 3, 2]);
    let mut tampered = base.clone();
    tampered[cols::OUT0] += FE::one();
    assert!(!violations(&tampered).is_empty());

    let mut upper = base.clone();
    upper[cols::OUT0 + 4] = FE::one();
    assert!(
        !violations(&upper).is_empty(),
        "the unused upper OUT lanes are pinned to zero"
    );
    assert_eq!(violations(&base), Vec::<usize>::new());
}

// =========================================================================
// M1–M7 — the mode-selected tag, PRE-COMMITTED controls
//
// Named in the transcript spec §5.3 before this chip existed, so they are
// inherited obligations rather than tests written to fit what got built. Each
// one is paired with an honest-path assertion: "the bad row is rejected" passes
// just as well when every row is rejected.
// =========================================================================

/// **M1 — a transcript row that hashed under the MERKLE tag is rejected.**
///
/// Spec form: "`m[8]` pinned to `TAG_LFMC` while `MODE_T = 1` — SAT", i.e. in a
/// model where the tag is free, a transcript row can compute the Merkle
/// function. On the real chip the tag is NOT free, so the same statement is a
/// rejection, and that is what is asserted here.
#[test]
fn m1_a_transcript_row_computing_the_merkle_tag_is_rejected() {
    let (a, b) = ([9u32, 8, 7, 6], [5u32, 4, 3, 2]);
    let mut row = hash_row_mode(HashMode::Transcript, a, b);
    // Recompute the whole witness under the WRONG domain, leaving MODE_T set.
    let digest = socket_digest(&a, &b); // "LFMC"
    row[cols::OUT0..cols::OUT0 + 4].copy_from_slice(&word_of(&digest));
    blake3_socket::fill_socket_witness_tagged(&mut row, TAG_LFMC);
    assert!(
        !violations(&row).is_empty(),
        "a MODE_T row carrying the Merkle computation must be rejected"
    );

    // HONEST CONTROL: the same row under its own domain satisfies everything.
    assert_eq!(
        violations(&hash_row_mode(HashMode::Transcript, a, b)),
        Vec::<usize>::new()
    );
}

/// **M2 — the mirror: a compress row that hashed under the TRANSCRIPT tag is
/// rejected.** Both directions, because a one-directional separation is not one.
#[test]
fn m2_a_compress_row_computing_the_transcript_tag_is_rejected() {
    let (a, b) = ([1u32, 2, 3, 4], [5u32, 6, 7, 8]);
    let mut row = hash_row_mode(HashMode::Compress, a, b);
    let digest = blake3_socket::transcript_digest(&a, &b);
    row[cols::OUT0..cols::OUT0 + 4].copy_from_slice(&word_of(&digest));
    blake3_socket::fill_socket_witness_tagged(&mut row, blake3_socket::TAG_LFMT);
    assert!(
        !violations(&row).is_empty(),
        "a MODE_C row carrying the transcript computation must be rejected"
    );

    assert_eq!(
        violations(&hash_row_mode(HashMode::Compress, a, b)),
        Vec::<usize>::new()
    );
}

/// **M3 — both mode bits set on one row is unsatisfiable**, and it is the
/// mode-sum booleanity (idx 4) that says so.
///
/// This is the constraint that stops `m[8]` being `TAG_LFMC + TAG_LFMT`, a tag
/// in neither domain.
#[test]
fn m3_both_two_to_one_modes_on_one_row_is_unsatisfiable() {
    let (a, b) = ([9u32, 8, 7, 6], [5u32, 4, 3, 2]);
    let mut row = hash_row_mode(HashMode::Compress, a, b);
    row[cols::MODE_T] = FE::one();
    assert!(
        violations(&row).contains(&4),
        "idx 4 — the mode-sum booleanity — must be the constraint that fires"
    );

    // HONEST CONTROL: clearing it again restores an accepted row.
    row[cols::MODE_T] = FE::zero();
    assert_eq!(violations(&row), Vec::<usize>::new());
}

/// **M4 — the mu gate IS the sum of the two two-to-one selectors**, so it
/// cannot be 1 while both are 0.
///
/// Structural rather than algebraic: on this chip `MU` is not a column a row
/// could set independently, it is the expression `MODE_C + MODE_T`. The test
/// pins that, and pins the consequence — with both zero the row is padding, it
/// satisfies the set vacuously and its bus sends carry multiplicity zero.
#[test]
fn m4_the_mu_gate_is_exactly_the_two_to_one_selector_sum() {
    assert_eq!(cols::MU_COLUMNS, [cols::MODE_C, cols::MODE_T, cols::MODE_L]);

    // A row with garbage in every witness column but no mode set is padding.
    let (a, b) = ([9u32, 8, 7, 6], [5u32, 4, 3, 2]);
    let mut row = hash_row_mode(HashMode::Compress, a, b);
    row[cols::MODE_C] = FE::zero();
    row[cols::OUT0..cols::OUT0 + 4].copy_from_slice(&word_of(&[0, 0, 0, 0]));
    for k in 0..4 {
        row[cols::S8 + k] = FE::zero();
    }
    assert_eq!(
        violations(&row),
        Vec::<usize>::new(),
        "with no mode set the row is padding and every mu-gated constraint is vacuous"
    );

    // HONEST CONTROL: it is vacuous because it is UNGATED-satisfiable, not
    // because the set accepts anything — restoring the mode makes the same
    // garbage row fail.
    row[cols::MODE_C] = FE::one();
    assert!(!violations(&row).is_empty());
}

/// **M5/M6 — ⚠ the AIR alone does not pin the tag; the PREPROCESSED binding
/// does.** This is the control that turns §3.3 from an assertion into a checked
/// claim, and it fires.
///
/// Constraint idx 4 pins the mode SUM to a bit, not each selector to a bit. So
/// a row with `MODE_C = x`, `MODE_T = 1 − x` satisfies it for every field
/// element `x`, and `m[8]` becomes `x·"LFMC" + (1−x)·"LFMT"` — which, solving
/// for `x`, is **any 32-bit value the prover likes**. This test picks the tag
/// `"XXXX"`, derives the `x` that produces it, and shows the constraint set
/// ACCEPTS the resulting row.
///
/// Two mechanisms stop a real prover doing this, and neither is in this file's
/// constraint set:
///
/// - the mode columns are **preprocessed**, fixed by the row's position in a
///   trace whose commitment is folded into `lfm_program_id`; and
/// - the admission validator's one-hot check rejects any program whose
///   `LFM_HASH` group carries a non-boolean selector.
///
/// Both are asserted below, so this is a live demonstration of *why* they are
/// load-bearing rather than a latent hole.
#[test]
fn m5_m6_the_mode_columns_must_be_preprocessed_or_the_tag_is_prover_chosen() {
    const FORGED_TAG: u32 = u32::from_le_bytes(*b"XXXX");
    let (a, b) = ([9u32, 8, 7, 6], [5u32, 4, 3, 2]);

    // x such that x·LFMC + (1−x)·LFMT = FORGED_TAG.
    let lfmc = FE::from(u64::from(TAG_LFMC));
    let lfmt = FE::from(u64::from(blake3_socket::TAG_LFMT));
    let x = (FE::from(u64::from(FORGED_TAG)) - &lfmt)
        * (&lfmc - &lfmt).inv().expect("the two tags differ");

    let mut row = vec![FE::zero(); cols::NUM_COLUMNS];
    row[cols::MODE_C] = x;
    row[cols::MODE_T] = FE::one() - x;
    row[cols::IN0..cols::IN0 + 4].copy_from_slice(&word_of(&a));
    row[cols::IN0 + 4..cols::IN0 + 8].copy_from_slice(&word_of(&b));
    for (k, iv) in BLAKE3_IV.iter().take(4).enumerate() {
        row[cols::S8 + k] = FE::from(u64::from(*iv));
    }
    let digest = blake3_socket::socket_digest_rounds_tagged(&a, &b, SOCKET_ROUNDS, FORGED_TAG);
    row[cols::OUT0..cols::OUT0 + 4].copy_from_slice(&word_of(&digest));
    blake3_socket::fill_socket_witness_tagged(&mut row, FORGED_TAG);

    assert_eq!(
        violations(&row),
        Vec::<usize>::new(),
        "⚠ the constraint set alone accepts a prover-chosen domain tag — the \
         mode columns being preprocessed is what stops this"
    );
    assert_ne!(
        digest,
        socket_digest(&a, &b),
        "the forged domain really is a different function"
    );

    // MECHANISM 1: the mode columns are inside the preprocessed prefix, so a
    // prover supplies neither.
    const { assert!(cols::MODE_C < cols::PREP_WIDTH) };
    const { assert!(cols::MODE_T < cols::PREP_WIDTH) };
    const { assert!(cols::MODE_P < cols::PREP_WIDTH) };

    // MECHANISM 2: the admission validator rejects a non-one-hot selector, so
    // the program above cannot be registered even if a prover could write it.
    let mut program = compress_program();
    let g = &mut program.groups.hash;
    let row0 = 0;
    g.data[row0 * g.width + super::layout::hash::MODE_C] = x;
    g.data[row0 * g.width + super::layout::hash::MODE_T] = FE::one() - x;
    assert!(
        matches!(
            super::validator::validate(&program),
            Err(super::validator::LfmViolation::NonOneHotSelector {
                chip: "LFM_HASH",
                ..
            })
        ),
        "the registrar must reject a fractional mode selector"
    );

    // HONEST CONTROL: the untouched program is admissible, so the rejection
    // above is about the tampering and not about the program.
    assert!(super::validator::validate(&compress_program()).is_ok());
}

/// **M7 — the capacity constraints (idx 0–3) bite on a transcript row.**
///
/// A transcript row is still a compress, so its capacity prefix is still the
/// IV — the selector widened to `MODE_C + MODE_T` and nothing else did. If the
/// widening had been forgotten, a transcript row's `S` would be pinned to zero
/// instead of to the IV, and this is the test that would have said so.
#[test]
fn m7_the_capacity_constraints_bite_on_a_transcript_row() {
    let (a, b) = ([9u32, 8, 7, 6], [5u32, 4, 3, 2]);
    for k in 0..4 {
        let mut row = hash_row_mode(HashMode::Transcript, a, b);
        row[cols::S8 + k] += FE::one();
        assert_eq!(
            violations(&row),
            vec![k],
            "a wrong capacity lane must violate exactly constraint {k}"
        );
    }

    // A transcript row's capacity is the IV, same as a compress row's — the two
    // differ in `m[8]` and in nothing else.
    let transcript = hash_row_mode(HashMode::Transcript, a, b);
    let compress = hash_row_mode(HashMode::Compress, a, b);
    for k in 0..4 {
        assert_eq!(transcript[cols::S8 + k], compress[cols::S8 + k]);
        assert_eq!(transcript[cols::S8 + k], FE::from(u64::from(BLAKE3_IV[k])));
    }
    assert_eq!(violations(&transcript), Vec::<usize>::new());
}

/// The degree bound survives the mode-selected tag: `m[8]` went from degree 0
/// to degree 1, and the wrap's blowup 2 depends on the maximum staying 3.
#[test]
fn the_mode_selected_tag_does_not_raise_the_degree() {
    let set = HashConstraints::BLAKE3;
    assert_eq!(ConstraintSet::<F, E>::max_degree(&set), 3);
    // Measured, not declared — `the_declared_degree_bound_is_respected` walks
    // the captured IR; this asserts the declaration it checks against.
}

// =========================================================================
// Prove and verify — rule 2: this is what makes the numbers measurements
// =========================================================================

/// A compress-only program: two leaf merges and a parent merge.
///
/// This is the shape the socket exists for — `edsl::merkle_walk`'s parent
/// compression — and it exercises obligation O2 as well, since `d0` and `d1` are
/// socket outputs fed straight back in as inputs. `trivial_program` cannot be
/// used: it contains a `permute`, which BLAKE3 has no socket for.
fn compress_program_source() -> LfmProgramSource {
    let mut b = LfmBuilder::new();
    let arena = b.declare_arena(4);
    let h: Vec<Cell> = (0..4).map(|i| b.hint_word(arena, i)).collect();
    let d0 = b.compress(h[0].as_digest(), h[1].as_digest());
    let d1 = b.compress(h[2].as_digest(), h[3].as_digest());
    let root = b.compress(d0, d1);
    b.public(root.as_cell());
    b.finish()
}

fn compress_program() -> LfmProgram {
    compile(compress_program_source())
}

/// Four arena words whose lanes are `u32`s — the socket's domain (O1).
fn arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u32)
            .map(|i| word_of(&[0x1000_0000 * (i + 1), 0x0BAD_F00D ^ i, i, 0xFFFF_FFFF - i]))
            .collect(),
    ]
}

/// `permute_coverage_program`'s arenas — three state cells.
fn permute_arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..3u64)
            .map(|i| core::array::from_fn(|j| FE::from(500 * (i + 1) + j as u64)))
            .collect(),
    ]
}

/// `trivial_program`'s arenas — arbitrary felts, which is exactly why BLAKE3
/// cannot take them.
fn trivial_arenas() -> Vec<Vec<LfmWord>> {
    vec![
        (0..4u64)
            .map(|i| core::array::from_fn(|j| FE::from(1_000 * (i + 1) + j as u64)))
            .collect(),
    ]
}

/// ★ The production prover builds this AIR, proves a program through it, and
/// the production verifier accepts.
#[test]
fn the_blake3_socket_proves_and_verifies() {
    let opts = options();
    let program = compress_program();
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let proved = lfm_prove_with_hasher(&program, &artifacts, &arenas(), &opts, KIND)
        .expect("proving under BLAKE3 must succeed");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "an honest BLAKE3-configured proof must verify"
    );

    // The public output is the Merkle root the socket computed, recomputed here
    // from the vectors' own reference function — so the proof's public words
    // are checked against the specification, not against the executor.
    let a = arenas();
    let lanes = |i: usize| lanes_of(&a[0][i]).expect("u32 lanes");
    let d0 = socket_digest(&lanes(0), &lanes(1));
    let d1 = socket_digest(&lanes(2), &lanes(3));
    let root = socket_digest(&d0, &d1);
    assert_eq!(proved.public_words, vec![(0u32, word_of(&root))]);
}

/// A proof is bound to the hasher it was produced under, in both directions.
#[test]
fn a_blake3_proof_does_not_verify_under_another_hasher() {
    let opts = options();
    let program = compress_program();
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let proved =
        lfm_prove_with_hasher(&program, &artifacts, &arenas(), &opts, KIND).expect("prove");

    for other in [HasherKind::Test, HasherKind::Poseidon] {
        // The digest stays the proved-under one: this isolates the AIR-set
        // mismatch rather than passing because the statement also moved.
        assert!(
            !verify_against(
                &artifacts.roots,
                &artifacts.program_id,
                artifacts.keccak_rnd_chunks,
                &proved.proof,
                &proved.public_words,
                &opts,
                other,
                artifacts.chip_set,
            ),
            "a BLAKE3 proof must not verify under {other:?}"
        );
    }
}

/// The hasher tag moves the program digest and no preprocessed root — the
/// Phase-3 binding, now with a third candidate in it.
///
/// The third candidate is the point: with only two, a width coincidence was
/// enough to separate them by accident. The tag is what separates them on
/// purpose.
#[test]
fn the_blake3_choice_moves_the_program_digest_and_no_root() {
    let opts = options();
    let program = compress_program();
    let test = build_artifacts_with_hasher(&program, &opts, HasherKind::Test);
    let blake = build_artifacts_with_hasher(&program, &opts, KIND);
    let pos = build_artifacts_with_hasher(&program, &opts, HasherKind::Poseidon);

    assert_eq!(build_artifacts(&program, &opts).program_id, test.program_id);
    assert_eq!(test.roots, blake.roots, "no root may move with the hasher");
    assert_eq!(test.log_heights, blake.log_heights);
    assert_eq!(test.keccak_rnd_chunks, blake.keccak_rnd_chunks);
    assert_ne!(test.program_id, blake.program_id);
    assert_ne!(pos.program_id, blake.program_id);
    assert_eq!(KIND.as_tag(), 2, "the wire tag is written out, not derived");
}

/// Prove the program with `mutate` applied to the hash trace, and report
/// whether the proof was ACCEPTED. A prover refusal and a verifier rejection
/// are both real rejections and this chip produces both.
fn round_trip(mutate: impl FnOnce(&mut TraceTable<F, E>)) -> Result<bool, String> {
    let opts = options();
    let program = compress_program();
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let exec = execute(&program, &arenas(), &KIND).expect("execute");
    let mut traces = build_traces_with_hasher(&program, &exec.records, KIND);
    mutate(&mut traces.hash);
    match prove_traces_with_hasher(
        &artifacts,
        &mut traces,
        &exec.public_words,
        &opts,
        KIND,
        stark::residency_mode::ResidencyMode::Retain,
    ) {
        Ok(proof) => Ok(verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proof,
            &exec.public_words,
            &opts,
            KIND,
            artifacts.chip_set,
        )),
        Err(e) => Err(format!("{e:?}")),
    }
}

fn assert_not_accepted(what: &str, mutate: impl FnOnce(&mut TraceTable<F, E>)) {
    if let Ok(true) = round_trip(mutate) {
        panic!("{what} must not produce an accepted proof, but the proof verified");
    }
}

/// Tamper rejection, one cell at a time, across the three column families the
/// arm adds: a lane byte, a mixing-core carry, and a digest byte.
///
/// The honest control is `the_blake3_socket_proves_and_verifies` above: without
/// it these would pass just as well if the AIR rejected everything.
#[test]
fn tampering_with_the_witness_is_not_accepted() {
    assert_not_accepted("a moved input lane byte", |t| {
        let v = t.main_table.get_row(0)[cols::lane_byte(0, 0)];
        t.main_table.set_fe(0, cols::lane_byte(0, 0), v + FE::one());
    });
    assert_not_accepted("a flipped add3 carry bit", |t| {
        let c = cols::g_base(0) + cols::G_A1_C;
        let v = t.main_table.get_row(0)[c];
        t.main_table.set_fe(0, c, v + FE::one());
    });
    assert_not_accepted("a moved digest byte", |t| {
        let v = t.main_table.get_row(0)[cols::out_byte(0, 0)];
        t.main_table.set_fe(0, cols::out_byte(0, 0), v + FE::one());
    });
    assert_not_accepted("a real flag on a padding row", |t| {
        t.main_table.set_fe(3, cols::MODE_C, FE::one());
    });
}

// =========================================================================
// F3.4 — what B1 retired, and the ONE thing it did not
// =========================================================================

/// ★ `TrivialV0` — a REGISTERED program — proves and verifies under BLAKE3.
///
/// It could not while it ended on a raw `permute`; option B1 replaced that with
/// a third `compress`, and this is the milestone that states.
#[test]
fn the_trivial_program_proves_and_verifies_under_blake3() {
    let opts = options();
    let program = trivial_program();
    let arenas = vec![
        (0..4u32)
            .map(|i| word_of(&[0x1000_0000 * (i + 1), 0x0BAD_F00D ^ i, i, 0xFFFF_FFFF - i]))
            .collect(),
    ];
    let artifacts = build_artifacts_with_hasher(&program, &opts, KIND);
    let proved = lfm_prove_with_hasher(&program, &artifacts, &arenas, &opts, KIND)
        .expect("TrivialV0 must prove under BLAKE3");
    assert!(
        verify_against(
            &artifacts.roots,
            &artifacts.program_id,
            artifacts.keccak_rnd_chunks,
            &proved.proof,
            &proved.public_words,
            &opts,
            artifacts.hasher,
            artifacts.chip_set,
        ),
        "an honest BLAKE3 proof of TrivialV0 must verify"
    );
}

// The O1 tripwire that used to live here is GONE, and that is the deliverable.
//
// It asserted that `FriToyV0` was refused under BLAKE3 for obligation O1, and
// its own doc said it must be replaced by a prove+verify when O1 closed. Option
// C closed it: `leaf_tests::fri_toy_proves_and_verifies_under_blake3` is the
// replacement, and it carries the negative leg the tripwire's criteria asked
// for.

/// ★ H1 guard — every `LFM_HASH` candidate emits each constraint index exactly
/// once, checked in RELEASE.
///
/// `EmitTracker`'s duplicate assert is `#[cfg(debug_assertions)]` and this
/// workspace declares no `[profile.release]` override, so under the house
/// convention `cargo test --release` it is a no-op: a second
/// `emit_base(idx, ..)` silently overwrites the first. Nothing else notices,
/// because a body that emits one index twice and another never still fills the
/// declared number of slots — `num_constraints`, `predicted_constraints` and
/// `assert_complete` all still pass while a constraint has been deleted.
///
/// This runs the real body through `ConstraintSet::meta` (no `cfg`) and demands
/// the emitted index multiset be exactly `0..num_constraints`. The checker's own
/// ability to fail — on this exact shape, a widened lane block overrunning the
/// pins after it — is established in
/// `stark::tests::constraint_index_tests::the_widened_lane_block_collides_and_the_checker_says_so`.
///
/// Required by COMMIT.md §1.4.4 H1. It guards the chip as it stands today, and
/// it is what would catch the `NUM_LANES` widening if that lands before the
/// framing indices stop being written as literals.
#[test]
fn every_hash_candidate_emits_each_constraint_index_exactly_once() {
    for (set, kind) in [
        (HashConstraints::TEST, HasherKind::Test),
        (HashConstraints::POSEIDON, HasherKind::Poseidon),
        (HashConstraints::BLAKE3, HasherKind::Blake3),
    ] {
        let declared = HashConstraints::num_constraints(kind);
        let meta = <HashConstraints as ConstraintSet<F, E>>::meta(&set);
        check_dense_index_set(&meta, declared)
            .unwrap_or_else(|e| panic!("{kind:?} LFM_HASH constraint body: {e}"));
    }
}
