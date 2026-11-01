//! Prove + verify the LFM-hosted BLAKE3 compression chip standalone.
//!
//! The `keccak_probe` pattern, one hash later: [`super::blake3_chip`] carries
//! the chip's real bus interactions and its real constraints, and this module
//! closes both of its buses — `BusId::ByteAlu` / `BusId::AreBytes` against the
//! UNCHANGED production `BITWISE` table, and `BusId::LfmMem` against a mirror
//! AIR standing in for the machine's memory. The preprocessed prefix is
//! committed for real, so the addresses and multiplicities the chip reads are
//! program data here exactly as they would be in the machine.
//!
//! Standing-decisions rule 2 is why this exists: an execute-only test would
//! prove nothing about the chip, because [`super::blake3::blake3_compress_6round`]
//! and the chip's `ValueFlow` would simply agree with each other. Only a
//! prove+verify makes the chip's constraints and interactions load
//! bearing, which is what turns the measured width into a *column*.
//!
//! # What this probe cannot see
//!
//! - **Whether the machine can drive the chip.** The mirror AIR is a synthetic
//!   memory: it sends whatever the ops say the inputs are. Nothing here checks
//!   that an LFM program can produce those words at those addresses, that the
//!   admission validator would accept the address assignment, or that the
//!   multiplicities match real read counts. Those are registrar obligations and
//!   they are exactly what registering the chip would exercise.
//! - **The epoch verifier's blake bill.** This measures cells per compression.
//!   The permutation count comes from wave 8's rate-parameterised closed form
//!   and is inherited, not re-established here.
//! - **Anything cryptographic about the 6-round variant** (assumption A6R).
use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::config::Commitment;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintSet, EmptyConstraints, RootKind, num_base_from_meta,
};
use stark::lookup::{
    AirWithBuses, AuxiliaryTraceBuildData, BusInteraction, BusValue, Multiplicity,
    NullBoundaryConstraintBuilder, Packing,
};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::view::MultiProofView;
use stark::prover::IsStarkProver;
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::IsStarkVerifier;

use crate::tables::bitwise;
use crate::tables::types::{BusId, FE, FEE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::test_utils::create_bitwise_air;

use super::blake3::{BLAKE3_ROUNDS, CANONICAL_VECTORS, canonical_expected_out};
use super::blake3_chip::{
    self, Blake3LfmConstraints, Blake3Operation, Blake3Values, IN_WORDS, MAIN_COLUMNS,
    NUM_CONSTRAINTS, NUM_G, OUT_WORDS, cols,
};
use super::commit::commit_columns;

type F = GoldilocksField;
type E = GoldilocksExtension;
type DynAir<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;
type ChipAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), Blake3LfmConstraints>;
type MirrorAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;

const PROBE_TAG: &[u8] = b"LFM_BLAKE3_PROBE_V1";
/// Compressions in the probe. Three real rows in a height-4 table leaves one
/// padding row, which `padding_row_turned_real_rejects` needs.
const NUM_OPS: usize = 3;

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("probe options")
}

fn transcript() -> crate::hash_pin::BlockTranscript {
    let mut t = crate::hash_pin::block_transcript(&[]);
    t.append_bytes(PROBE_TAG);
    t
}

// =========================================================================
// The ops
// =========================================================================

/// Three compressions taken from the canonical 6-round vectors, at disjoint
/// addresses.
///
/// Using the pinned vectors rather than fresh randomness means the trace's own
/// OUT columns are checkable against a constant that came from outside this
/// repository's Rust (see [`super::blake3`]'s provenance note).
fn probe_ops() -> Vec<Blake3Operation> {
    (0..NUM_OPS)
        .map(|i| {
            let v = &CANONICAL_VECTORS[i];
            let base = 1_000 + (i as u64) * 100;
            Blake3Operation {
                in_addr: core::array::from_fn(|j| base + j as u64),
                out_addr: core::array::from_fn(|j| base + 50 + j as u64),
                // Distinct nonzero read counts: a uniform 1 would not notice a
                // multiplicity mixed up between output words.
                read_counts: core::array::from_fn(|j| 1 + j as u64),
                values: Blake3Values {
                    h: v.h,
                    m: v.m,
                    t: v.t,
                    block_len: v.block_len,
                    flags: v.flags,
                },
            }
        })
        .collect()
}

/// The value halves of a probe op list — what the BITWISE feed is computed from.
fn probe_values(ops: &[Blake3Operation]) -> Vec<Blake3Values> {
    ops.iter().map(|op| op.values).collect()
}

// =========================================================================
// The AIRs
// =========================================================================

/// The preprocessed prefix, column-major and padded — what the program would
/// supply and what the chip's addresses and multiplicities are read from.
fn prep_columns(ops: &[Blake3Operation], num_rows: usize) -> Vec<Vec<FE>> {
    let mut columns = vec![vec![FE::zero(); num_rows]; cols::PREP_WIDTH];
    for (row, op) in ops.iter().enumerate() {
        for j in 0..IN_WORDS {
            columns[cols::in_addr(j)][row] = FE::from(op.in_addr[j]);
        }
        for j in 0..OUT_WORDS {
            columns[cols::out_addr(j)][row] = FE::from(op.out_addr[j]);
            columns[cols::mult(j)][row] = FE::from(op.read_counts[j]);
        }
        columns[cols::MU][row] = FE::one();
    }
    columns
}

fn chip_air(prep_root: Commitment, opts: &ProofOptions) -> ChipAir {
    AirWithBuses::new(
        cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: blake3_chip::bus_interactions(),
        },
        opts,
        1,
        Blake3LfmConstraints,
    )
    .with_name("LFM_BLAKE3")
    .with_preprocessed(prep_root, cols::PREP_WIDTH)
}

/// A synthetic `LfmMem` counterparty: `[ADDR, V0..V3, SEND_MULT, RECV_MULT]`.
///
/// One row per word the chip touches. Input words are SENT here (the chip
/// receives them); output words are RECEIVED here `read_counts` times (the chip
/// sends them once with that multiplicity). Nothing constrains the values — the
/// mirror is memory, and in the machine the `LfmMem` multiset IS the semantics.
mod mirror {
    pub const ADDR: usize = 0;
    pub const V0: usize = 1; // ..V3
    pub const SEND_MULT: usize = 5;
    pub const RECV_MULT: usize = 6;
    pub const NUM_COLUMNS: usize = 7;
}

fn mirror_token() -> Vec<BusValue> {
    let mut v = vec![BusValue::Packed {
        start_column: mirror::ADDR,
        packing: Packing::Direct,
    }];
    v.extend((0..4).map(|l| BusValue::Packed {
        start_column: mirror::V0 + l,
        packing: Packing::Direct,
    }));
    v
}

fn mirror_air(opts: &ProofOptions) -> MirrorAir {
    let interactions = vec![
        BusInteraction::sender(
            BusId::LfmMem,
            Multiplicity::Column(mirror::SEND_MULT),
            mirror_token(),
        ),
        BusInteraction::receiver(
            BusId::LfmMem,
            Multiplicity::Column(mirror::RECV_MULT),
            mirror_token(),
        ),
    ];
    AirWithBuses::new(
        mirror::NUM_COLUMNS,
        AuxiliaryTraceBuildData { interactions },
        opts,
        1,
        EmptyConstraints,
    )
    .with_name("LFM_MEM_MIRROR")
}

/// Four `u32` lanes of machine word `word` out of a flat `u32` array.
fn lanes(words: &[u32], word: usize) -> [u64; 4] {
    core::array::from_fn(|l| words[4 * word + l] as u64)
}

fn mirror_trace(ops: &[Blake3Operation]) -> TraceTable<F, E> {
    let rows = (ops.len() * (IN_WORDS + OUT_WORDS))
        .next_power_of_two()
        .max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(rows * mirror::NUM_COLUMNS),
        mirror::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;
    let mut row = 0usize;
    for op in ops {
        let inputs = op.values.input_words();
        for j in 0..IN_WORDS {
            table.set_u64(row, mirror::ADDR, op.in_addr[j]);
            for (l, v) in lanes(&inputs, j).into_iter().enumerate() {
                table.set_u64(row, mirror::V0 + l, v);
            }
            table.set_fe(row, mirror::SEND_MULT, FE::one());
            row += 1;
        }
        let outputs = op.values.output_words();
        for j in 0..OUT_WORDS {
            table.set_u64(row, mirror::ADDR, op.out_addr[j]);
            for (l, v) in lanes(&outputs, j).into_iter().enumerate() {
                table.set_u64(row, mirror::V0 + l, v);
            }
            table.set_u64(row, mirror::RECV_MULT, op.read_counts[j]);
            row += 1;
        }
    }
    trace
}

fn bitwise_trace(ops: &[Blake3Operation]) -> TraceTable<F, E> {
    let mut hist = bitwise::BitwiseHistogram::new();
    hist.add_ops(&blake3_chip::bitwise_ops_for(&probe_values(ops)));
    let mut bw = bitwise::generate_bitwise_trace();
    hist.fill_multiplicities(&mut bw);
    bw
}

/// The three traces, in AIR order: chip, mirror, BITWISE.
fn build_traces(ops: &[Blake3Operation]) -> [TraceTable<F, E>; 3] {
    [
        blake3_chip::generate_blake3_trace(ops),
        mirror_trace(ops),
        bitwise_trace(ops),
    ]
}

fn prove_traces(
    opts: &ProofOptions,
    chip: &ChipAir,
    traces: &mut [TraceTable<F, E>; 3],
) -> Result<stark::proof::stark::MultiProof<F, E, ()>, stark::prover::ProvingError> {
    let mirror = mirror_air(opts);
    let bw_air = create_bitwise_air(opts).with_preprocessed(
        bitwise::preprocessed_commitment(opts),
        bitwise::NUM_PRECOMPUTED_COLS,
    );
    let [t0, t1, t2] = traces;
    let pairs: Vec<(DynAir, &mut TraceTable<F, E>, &())> =
        vec![(chip, t0, &()), (&mirror, t1, &()), (&bw_air, t2, &())];
    let mut t = transcript();
    crate::hash_pin::BlockProver::multi_prove(
        pairs,
        &mut t,
        stark::residency_mode::ResidencyMode::Retain,
    )
}

fn verify_proof(
    opts: &ProofOptions,
    chip: &ChipAir,
    proof: &stark::proof::stark::MultiProof<F, E, ()>,
) -> bool {
    let mirror = mirror_air(opts);
    let bw_air = create_bitwise_air(opts).with_preprocessed(
        bitwise::preprocessed_commitment(opts),
        bitwise::NUM_PRECOMPUTED_COLS,
    );
    let refs: Vec<DynAir> = vec![chip, &mirror, &bw_air];
    let mut vt = transcript();
    crate::hash_pin::BlockVerifier::multi_verify_views(
        &refs,
        MultiProofView::Owned(proof),
        &mut vt,
        &FEE::zero(),
    )
}

/// Prove + verify, optionally corrupting the chip trace in between.
///
/// `Err` means the prover refused — which for this chip is the *expected*
/// outcome of most tampering, because unlike the keccak adapter it carries 769
/// polynomial constraints that a wrong cell violates locally.
fn round_trip(mutate: impl FnOnce(&mut TraceTable<F, E>)) -> Result<bool, String> {
    let opts = options();
    let ops = probe_ops();
    let num_rows = ops.len().next_power_of_two().max(4);
    let root = commit_columns(&prep_columns(&ops, num_rows), &opts);
    let chip = chip_air(root, &opts);
    let mut traces = build_traces(&ops);
    mutate(&mut traces[0]);
    match prove_traces(&opts, &chip, &mut traces) {
        Ok(proof) => Ok(verify_proof(&opts, &chip, &proof)),
        Err(e) => Err(format!("{e:?}")),
    }
}

/// Assert a mutation does not end in an accepted proof.
///
/// A refusal by the prover and a rejection by the verifier are both real
/// rejections and this chip produces both: a cell that violates one of its 769
/// constraints is caught locally, while a cell that only breaks a bus reaches
/// the verifier. Each caller records which it observed in its own doc comment.
fn assert_not_accepted(what: &str, mutate: impl FnOnce(&mut TraceTable<F, E>)) {
    if let Ok(true) = round_trip(mutate) {
        panic!("{what} must not produce an accepted proof, but the proof verified");
    }
}

// =========================================================================
// The measurement
// =========================================================================

/// The blake column's per-compression cell law, on our stack, at BOTH round
/// counts.
///
/// `main + 3·aux` with `aux = ceil(interactions / 2)` is `airs.rs`'s census
/// formula — the same instrument that produced the keccak and Poseidon columns,
/// so the three are comparable by construction rather than by argument.
///
/// The closed forms are written out as functions of the round count and the
/// literals for both are pinned, so the A6R price stays visible whichever way
/// the build is compiled; the built layout is then asserted to equal the
/// prediction at the compiled count. Two statements that can disagree.
#[test]
fn the_hosted_chip_cell_budget_at_both_round_counts() {
    // 112 input bytes + `8·rounds` G-blocks of 60 + 64 feed-forward bytes.
    const fn predicted_main(rounds: usize) -> usize {
        112 + 60 * (8 * rounds) + 64
    }
    // 13 `LfmMem` tokens (7 read + 4 written + 2 reversed-digest);
    // `ByteAlu[XOR]` over `4·8·rounds` mixing words and 16 feed-forward words;
    // `AreBytes` over `2·8·rounds` rotations; 32 message.
    const fn predicted_interactions(rounds: usize) -> usize {
        13 + 4 * (4 * (8 * rounds) + 16) + 4 * (2 * (8 * rounds)) + 32
    }
    const fn predicted_cells(rounds: usize) -> usize {
        predicted_main(rounds) + 3 * predicted_interactions(rounds).div_ceil(2)
    }

    // 6 rounds — the A6R variant. These four literals are #903's and were the
    // measured figures before the round count became a knob.
    assert_eq!(predicted_main(6), 3_056);
    assert_eq!(predicted_interactions(6), 1_261);
    assert_eq!(predicted_interactions(6).div_ceil(2), 631);
    assert_eq!(predicted_cells(6), 4_949);
    // Group by group at 6 rounds, so a layout change cannot move the total
    // silently: 13 LfmMem + 832 ByteAlu + 384 shift AreBytes + 32 message.
    assert_eq!(predicted_interactions(6), 13 + 832 + 384 + 32);

    // 7 rounds — standard BLAKE3, the default. PLAN §7 predicted exactly these
    // on paper; this is the same arithmetic against the built layout.
    assert_eq!(predicted_main(7), 3_536);
    assert_eq!(predicted_interactions(7), 1_453);
    assert_eq!(predicted_interactions(7).div_ceil(2), 727);
    assert_eq!(predicted_cells(7), 5_717);

    // The built layout IS the prediction at the compiled round count.
    let interactions = blake3_chip::bus_interactions().len();
    let aux = interactions.div_ceil(2);
    assert_eq!(cols::PREP_WIDTH, 20, "preprocessed prefix");
    assert_eq!(cols::G - cols::IN, 112, "input bytes");
    assert_eq!(cols::OUT - cols::G, 60 * NUM_G, "G-blocks × 60 cells");
    assert_eq!(
        cols::NUM_COLUMNS - cols::OUT,
        64,
        "feed-forward output bytes"
    );
    assert_eq!(MAIN_COLUMNS, predicted_main(BLAKE3_ROUNDS));
    assert_eq!(cols::NUM_COLUMNS, MAIN_COLUMNS + cols::PREP_WIDTH);
    assert_eq!(
        IN_WORDS + OUT_WORDS + cols::DIGEST_WORDS,
        13,
        "LfmMem tokens"
    );
    assert_eq!(interactions, predicted_interactions(BLAKE3_ROUNDS));
    assert_eq!(
        MAIN_COLUMNS + 3 * aux,
        predicted_cells(BLAKE3_ROUNDS),
        "base-field-equivalent cells"
    );

    // #903's syscall variant at 6 rounds, for the delta the hosting buys: 3,219
    // main and 1,397 interactions (699 aux) = 5,316. The difference is all I/O.
    assert_eq!(3_219 + 3 * 1_397usize.div_ceil(2), 5_316);

    // ★ For the comparison this chip exists to support: the `LFM_HASH` BLAKE3
    // socket arm is cheaper at BOTH round counts — a constant initial state, a
    // constant message tail, and twelve of the sixteen output words never built.
    //
    // Asserted against `blake3_socket_tests`' own census formula rather than
    // against a transcription of its output. What stood here was "4,741 at 6
    // rounds and 5,509 at 7": those predated the leaf mode's canonicity block,
    // nothing recomputed them, and they were wrong by 8 for as long as they
    // stood — then wrong by 36 once the leaf RATE widened the socket. A cost
    // figure no test derives is a comment, not a claim.
    for rounds in [6, 7] {
        let socket = super::blake3_socket_tests::predicted_cells(rounds);
        assert!(
            socket < predicted_cells(rounds),
            "hosting must stay cheaper than the standalone chip at {rounds} \
             rounds: socket {socket}, standalone {}",
            predicted_cells(rounds)
        );
    }
    assert_eq!(super::blake3_socket_tests::predicted_cells(6), 4_777);
    assert_eq!(super::blake3_socket_tests::predicted_cells(7), 5_545);
}

/// Every constraint index is emitted exactly once, and the count is the one the
/// module documents. The syscall chip's single-compression core emits 814; the
/// 45 address-derivation constraints have no counterpart here.
#[test]
fn the_chip_emits_its_constraints_at_degree_3() {
    // 16 per G-instance — two add3s (a sum identity and two carry booleanities
    // each), two add2 carry booleanities, two rotations of four — plus the
    // ungated `IS_BIT(MU)`. 769 at 6 rounds, 897 at 7; both written out.
    assert_eq!(16 * (8 * 6) + 1, 769);
    assert_eq!(16 * (8 * 7) + 1, 897);
    assert_eq!(NUM_CONSTRAINTS, 16 * NUM_G + 1);
    assert_eq!(
        NUM_CONSTRAINTS,
        3 * (NUM_G * 2) + NUM_G * 2 + 4 * (NUM_G * 2) + 1
    );
    // The syscall variant's single-compression core emits 814 at 6 rounds (its
    // absorb mode adds 34 more, none of them per-G); the 45 address-derivation
    // constraints have no counterpart here.
    if BLAKE3_ROUNDS == 6 {
        assert_eq!(814 - 45, NUM_CONSTRAINTS, "vs the syscall variant's core");
    }

    let set = Blake3LfmConstraints;
    let meta = ConstraintSet::<F, E>::meta(&set);
    assert_eq!(meta.len(), NUM_CONSTRAINTS, "constraints emitted");
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i, "meta must be dense and idx-ordered");
        assert_eq!(m.kind, RootKind::Base, "every blake constraint is base");
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
    // Not merely `<=`: the μ-gated carry booleanities really are cubic, so a
    // set that quietly topped out at 2 would mean the carries had stopped being
    // constrained.
    assert_eq!(
        degrees.iter().map(|&(_, d)| d).max(),
        Some(3),
        "some constraint must actually reach degree 3"
    );
}

// =========================================================================
// The round trip
// =========================================================================

#[test]
fn the_hosted_chip_proves_and_verifies() {
    let ops = probe_ops();
    let traces = build_traces(&ops);

    // The chip's OUT columns are the real compression, byte for byte, against
    // the canonical vectors.
    for (row, op) in ops.iter().enumerate() {
        let expected = canonical_expected_out(row);
        assert_eq!(
            expected,
            op.values.output_words(),
            "op {row}'s output must be the primitive's at the compiled round count"
        );
        for (i, &word) in expected.iter().enumerate() {
            for b in 0..4 {
                assert_eq!(
                    traces[0].main_table.get_row(row)[cols::out_word(i, b)],
                    FE::from(u64::from((word >> (8 * b)) as u8)),
                    "OUT byte ({i}, {b}) of row {row}"
                );
            }
        }
    }

    // The BITWISE feed is exactly the senders' count, with no address-shaped
    // lookups: 1,248 per compression at 6 rounds and 1,440 at 7. Both literals
    // are written out so the flip cannot quietly move the feed.
    const fn predicted_bitwise(rounds: usize) -> usize {
        4 * (4 * (8 * rounds) + 16) + 4 * (2 * (8 * rounds)) + 32
    }
    assert_eq!(predicted_bitwise(6), 1_248);
    assert_eq!(predicted_bitwise(7), 1_440);
    assert_eq!(
        blake3_chip::bitwise_ops_for(&probe_values(&ops)).len(),
        ops.len() * predicted_bitwise(BLAKE3_ROUNDS),
        "per-compression BITWISE lookup count"
    );
    // And it is the interaction list less the 13 `LfmMem` tokens — the mirror
    // property, which is what stops the feed and the senders from drifting.
    // 13, not 11: registration added the two reversed-digest sends, which carry
    // no BITWISE lookup because they are a second `Linear` over columns the
    // plain digest send already covers.
    assert_eq!(
        predicted_bitwise(BLAKE3_ROUNDS) + IN_WORDS + OUT_WORDS + cols::DIGEST_WORDS,
        blake3_chip::bus_interactions().len()
    );

    assert_eq!(round_trip(|_| {}), Ok(true), "honest proof must verify");
}

/// The padding rows carry nothing at all — no pointer pad, no witness.
#[test]
fn padding_rows_are_all_zero() {
    let ops = probe_ops();
    let traces = build_traces(&ops);
    let row = traces[0].main_table.get_row(NUM_OPS);
    assert!(
        row.iter().all(|c| *c == FE::zero()),
        "the padding row must be entirely zero, so that μ = 0 is the only thing \
         standing between it and the constraint set"
    );
}

// =========================================================================
// Falsification (rule 1)
// =========================================================================

/// CONTROL. If this ever fails, every mutation below is reporting on a broken
/// harness rather than on the chip — check it first (rule 7's corollary).
#[test]
fn falsification_control_the_untampered_proof_verifies() {
    assert_eq!(round_trip(|_| {}), Ok(true), "control must be green");
}

/// A flipped OUT byte: it is a feed-forward XOR *result*, so the XOR lookup
/// finds no BITWISE row, and the `LfmMem` word the mirror receives no longer
/// matches either.
#[test]
fn a_tampered_output_byte_rejects() {
    assert_not_accepted("a flipped OUT byte", |t| {
        let old = t.main_table.get_row(1)[cols::out_word(5, 2)];
        t.main_table
            .set_fe(1, cols::out_word(5, 2), old + FE::one());
    });
}

/// A flipped message byte. `m` is never XORed, so the only things that see this
/// are its explicit `AreBytes` send, the add3 sum identity and the `LfmMem`
/// read. This is the test that would go green if the 32 message range checks
/// were ever dropped as "redundant" *and* the sum identity were loosened.
#[test]
fn a_tampered_message_byte_rejects() {
    assert_not_accepted("a flipped message byte", |t| {
        let old = t.main_table.get_row(0)[cols::in_word(8 + 3, 1)];
        t.main_table
            .set_fe(0, cols::in_word(8 + 3, 1), old + FE::one());
    });
}

/// A padding row turned real. `MU` is preprocessed, so the prover recommits the
/// prefix and refuses before the constraint set is ever consulted — which IS
/// the point: an is-real flag a prover can choose is exactly what preprocessing
/// prevents, and the keccak adapter's `padding_row_multiplicity_rejects`
/// documents the weaker witness-side version.
#[test]
fn a_padding_row_turned_real_rejects() {
    assert_not_accepted("an is-real padding row", |t| {
        t.main_table.set_fe(NUM_OPS, cols::MU, FE::one())
    });
}

/// A bumped output-word read count, in the witness. Same shape: program data.
#[test]
fn a_tampered_read_multiplicity_rejects() {
    assert_not_accepted("a bumped output-word read count", |t| {
        let old = t.main_table.get_row(0)[cols::mult(2)];
        t.main_table.set_fe(0, cols::mult(2), old + FE::one());
    });
}

/// A carry bit flipped on an add3. The sum identity `a + b + m − s − 2^32·(c1+c2)`
/// is the only thing that sees it, and it is exactly the constraint #903's
/// "two summed committed carry bits" decision exists to keep at degree 3.
#[test]
fn a_tampered_add3_carry_bit_rejects() {
    assert_not_accepted("a flipped add3 carry bit", |t| {
        let col = cols::g_base(7) + cols::G_A1_C;
        let old = t.main_table.get_row(2)[col];
        t.main_table.set_fe(2, col, old + FE::one());
    });
}
