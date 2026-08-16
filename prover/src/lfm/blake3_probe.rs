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

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
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
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

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

fn transcript() -> DefaultTranscript<E> {
    let mut t = DefaultTranscript::<E>::new(&[]);
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
    Prover::multi_prove(
        pairs,
        &mut t,
        #[cfg(feature = "disk-spill")]
        Default::default(),
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
    Verifier::multi_verify_views(&refs, MultiProofView::Owned(proof), &mut vt, &FEE::zero())
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
/// module documents. #903 emits 814; the 45 address-derivation constraints have
/// no counterpart here.
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
    // #903's syscall variant emits 814 at 6 rounds; the 45 address-derivation
    // constraints have no counterpart here.
    if BLAKE3_ROUNDS == 6 {
        assert_eq!(814 - 45, NUM_CONSTRAINTS, "vs #903's syscall variant");
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

// =========================================================================
// The column, at the production epoch shape
// =========================================================================

/// Two-term peak-RSS model (wave 7, after the one-parameter 33.7 B/cell fit was
/// falsified): `27 B` per base-field-equivalent cell plus `190 MB` per
/// sub-proof.
///
/// ⚠ Both coefficients were calibrated on KECCAK-SHAPED runs — a machine whose
/// biggest tables are a 1,480-column round chip and a 2^20-row lookup table.
/// Nothing has checked that 27 B/cell survives a machine whose widest table is
/// a 3,056-column single-row chip, let alone one with no lookup table at all,
/// so every GiB below is a projection carrying that caveat and not a
/// measurement.
const BYTES_PER_CELL: f64 = 27.0;
const BYTES_PER_SUB_PROOF: f64 = 190_000_000.0;
const GIB: f64 = (1u64 << 30) as f64;

fn projected_gib(cells: u64, sub_proofs: usize) -> f64 {
    (cells as f64 * BYTES_PER_CELL + sub_proofs as f64 * BYTES_PER_SUB_PROOF) / GIB
}

/// ★ The blake column, the residue split and the re-derived matrix, on the real
/// epoch verifier.
///
/// `#[ignore]`d for the same reason `wrap_tests::the_wrap_census_at_blowup_8`
/// is: it proves a real inner epoch at blowup 8 and then emits ~2.25M
/// instructions. Run with
/// `cargo test -p lambda-vm-prover --lib the_blake_column -- --ignored --nocapture`.
///
/// Every line labels its basis. Three MEASURED inputs feed it — the 4,946
/// cells per compression proved above, this run's own census, and this run's
/// own permutation closed form — and everything else is arithmetic over them.
#[test]
#[ignore]
fn the_blake_column_and_the_residue_split() {
    use super::airs::{lfm_cell_counts, lfm_chip_census};
    use super::epoch_verify::{query_permutations, query_permutations_at_rate};
    use super::instr::Instr;
    use super::layout::padded_rows;

    let inner = crate::recursion::Preset::Blowup8.options();
    let e = super::epoch_tests::real_epoch_with(inner.clone());
    let profile = super::wrap_tests::epoch_profile(&e);
    let program = super::epoch_tests::epoch_program(&e, true);
    let spine = super::epoch_tests::epoch_program(&e, false);

    let census = lfm_chip_census(&program);
    let (main, aux) = lfm_cell_counts(&program);
    let total = main + 3 * aux;
    let sub_proofs = census.len();

    println!(
        "\n★ EPOCH {profile}, inner blowup {}, {} queries, {sub_proofs} sub-proofs",
        inner.blowup_factor, inner.fri_number_of_queries
    );
    println!(
        "   {:>12} {:>12} {:>7} {:>6} {:>16} {:>8}",
        "chip", "rows", "main", "aux", "base-equiv", "% total"
    );
    // KECCAK_RND reports once per chunk; fold the chunks so the table reads as
    // one line per chip class, which is what the matrix rows are about.
    let mut folded: Vec<(&str, u64, usize, usize, u64)> = Vec::new();
    for c in &census {
        let cells = c.main_cells() + 3 * c.aux_cells();
        match folded.iter_mut().find(|f| f.0 == c.name) {
            Some(f) => {
                f.1 += c.rows;
                f.4 += cells;
            }
            None => folded.push((c.name, c.rows, c.main_cols, c.aux_cols, cells)),
        }
    }
    for (name, rows, m, a, cells) in &folded {
        println!(
            "   {name:>12} {rows:>12} {m:>7} {a:>6} {cells:>16} {:>7.2}%",
            100.0 * *cells as f64 / total as f64
        );
    }
    println!("   {:>12} {:>50}", "TOTAL", total);

    let cells_of = |names: &[&str]| -> u64 {
        folded
            .iter()
            .filter(|f| names.contains(&f.0))
            .map(|f| f.4)
            .sum()
    };
    // The keccak permutation itself, and the 2^20-row lookup table it shares
    // with anything byte-oriented. Split because a field-native hash deletes
    // BOTH while blake deletes only the first.
    let keccak_perm = cells_of(&["LFM_KECCAK", "KECCAK_RND", "KECCAK_RC"]);
    let bitwise = cells_of(&["BITWISE"]);
    let residue = total - keccak_perm - bitwise;

    // ---- permutations, at both rates, from the closed form over the shapes.
    let legs_17: usize = e.legs.iter().map(|l| query_permutations(&l.verify)).sum();
    let legs_8: usize = e
        .legs
        .iter()
        .map(|l| query_permutations_at_rate(&l.verify, 8))
        .sum();
    let emitted = super::wrap_tests::permutations(&program);
    let spine_perms = super::wrap_tests::permutations(&spine);
    assert_eq!(
        emitted - spine_perms,
        legs_17,
        "the rate-17 closed form must reproduce the emitted legs"
    );
    // Rate 8 is BLAKE3's own: its socket absorbs two cells of message per
    // compression, and option B1 did not change that. It is NOT the
    // field-native chain's rate — that is `epoch_verify::LFM_HASH_RATE_FELTS`,
    // which is 4 because the chain absorbs one cell per step. The two were the
    // same number while the sponge was a three-cell duplex, and this line used
    // to say "blake and field-native" on that basis; they have since diverged.
    //
    // The spine is absorption-bound, so at rate 8 it lies between 1.0x and
    // 2.125x its rate-17 cost — wave 8's interval, restated on this run's own
    // spine count rather than quoted.
    let p_lo = legs_8 + spine_perms;
    let p_hi = legs_8 + (spine_perms as f64 * 17.0 / 8.0).ceil() as usize;
    println!(
        "\n   PERMUTATIONS  emitted {emitted} = spine {spine_perms} + legs {legs_17} (MEASURED)\n   \
         closed form   legs @ rate 17 (keccak) {legs_17}, @ rate 8 (BLAKE3's socket) {legs_8}\n   \
         P at rate 8 in [{p_lo}, {p_hi}] — legs exact, spine bounded"
    );

    // ---- the byteswap gadget: exactly the 64-bit decompositions.
    //
    // `sample_u64_pow2` asserts nbits <= 32 and every other production
    // `bit_dec` site passes 32 or a Merkle depth, so a 64-bit decomposition in
    // this program IS a `felt_be_halves` and nothing else. Counted rather than
    // reasoned about, with the whole histogram printed so that a new 64-bit
    // caller would show up instead of being silently folded in.
    let mut hist = std::collections::BTreeMap::<usize, usize>::new();
    for i in &program.instrs {
        if let Instr::BitDec { bits, .. } = i {
            *hist.entry(bits.len()).or_default() += 1;
        }
    }
    println!("\n   BitDec width histogram: {hist:?}");
    let swaps = hist.get(&64).copied().unwrap_or(0);

    let width = |name: &str| -> (u64, u64) {
        let f = folded.iter().find(|f| f.0 == name).expect("chip in census");
        (f.2 as u64, f.3 as u64)
    };
    let (bitdec_m, bitdec_a) = width("LFM_BITDEC");
    let (balu_m, balu_a) = width("LFM_BALU");
    let cell_law = |m: u64, a: u64, rows: u64| rows * (m + 3 * a);

    let bd_rows = program.groups.bitdec.real_rows;
    let ba_rows = program.groups.balu.real_rows;
    let before = cell_law(bitdec_m, bitdec_a, padded_rows(bd_rows) as u64)
        + cell_law(balu_m, balu_a, padded_rows(ba_rows) as u64);
    let after = cell_law(bitdec_m, bitdec_a, padded_rows(bd_rows - swaps) as u64)
        + cell_law(balu_m, balu_a, padded_rows(ba_rows - 64 * swaps) as u64);
    let unpadded =
        cell_law(bitdec_m, bitdec_a, swaps as u64) + cell_law(balu_m, balu_a, 64 * swaps as u64);
    let byteswap = before - after;

    println!(
        "\n   BYTESWAP GADGET  {swaps} felts x (1 BitDec + 64 BALU)\n   \
         LFM_BITDEC {bd_rows} real rows -> {} padded, {bitdec_m} main / {bitdec_a} aux\n   \
         LFM_BALU   {ba_rows} real rows -> {} padded, {balu_m} main / {balu_a} aux \
         ({:.2}% of all BALU rows)\n   \
         unpadded closed form {unpadded}\n   \
         padding-aware delta  {byteswap}  (the two chips together: before {before}, after {after})",
        padded_rows(bd_rows),
        padded_rows(ba_rows),
        100.0 * (64 * swaps) as f64 / ba_rows as f64,
    );

    // ---- the three residues the matrix needs.
    let residue_field_native = residue - byteswap;
    println!(
        "\n   RESIDUE (everything that is not the hash chip or its lookup table)\n   \
         keccak permutation chips  {keccak_perm:>16}  {:>6.2}%\n   \
         BITWISE (2^20 fixed)      {bitwise:>16}  {:>6.2}%\n   \
         residue                   {residue:>16}  {:>6.2}%\n   \
         \x20 of which byteswap       {byteswap:>16}  {:>6.2}% OF THE RESIDUE\n   \
         residue, byte-oriented    {residue:>16}  (blake keeps the gadget AND BITWISE)\n   \
         residue, field-native     {residue_field_native:>16}  (gadget deleted, BITWISE deleted)",
        100.0 * keccak_perm as f64 / total as f64,
        100.0 * bitwise as f64 / total as f64,
        100.0 * residue as f64 / total as f64,
        100.0 * byteswap as f64 / residue as f64,
    );
    println!(
        "   ⚠ the field-native line is DERIVED by subtraction from a keccak-shaped\n   \
         emission, not measured on a re-emitted field-native verifier. It is an\n   \
         UPPER bound on that residue: a field-native absorb also deletes the\n   \
         Pack/Unpack traffic around the gadget, and LFM_LANES still costs {} here.",
        cells_of(&["LFM_LANES"])
    );

    // ---- the matrix, re-derived.
    //
    // Hash-chip cells at P permutations: `rows x (main + 3 x aux)`, with rows
    // either padded to the next power of two (one AIR instance) or chunked the
    // way KECCAK_RND is (several instances, ~1.9% waste). Both are printed
    // because the choice is a policy, not a property of the hash.
    // ★ P is the instrument's OWN measured interval, not a constant.
    //
    // This line was a hardcoded `192_000` while the same function computed
    // `[p_lo, p_hi]` two dozen lines above and only PRINTED it — and on this
    // run that interval is roughly [287k, 291k], so every ratio below was
    // being taken at a hash count about a third under what the run measured,
    // in the direction that flatters every non-keccak column. Reading the
    // measurement is the whole fix; the assertion is what stops it drifting
    // back to a literal.
    //
    // Both ends are priced. The spine is absorption-bound and only bounded, so
    // a single number would be a choice about which end to quote — and the two
    // ends bracket the answer rather than approximating it.
    assert!(
        p_lo <= p_hi && p_lo > 0,
        "the permutation interval must be non-degenerate, got [{p_lo}, {p_hi}]"
    );
    let chunked = |perms: u64| (perms as f64 * 1.01871).ceil() as u64;
    let unchunked = |perms: u64| perms.next_power_of_two();
    let row = |name: &str, cells_per_perm: u64, resid: u64, table: u64| {
        for (how, rows) in [
            ("chunked@lo", chunked(p_lo as u64)),
            ("chunked@hi", chunked(p_hi as u64)),
            ("padded@lo", unchunked(p_lo as u64)),
            ("padded@hi", unchunked(p_hi as u64)),
        ] {
            let hash_cells = rows * cells_per_perm;
            let t = resid + table + hash_cells;
            println!(
                "   {name:>28} {how:>10}  hash {hash_cells:>13}  total {t:>13}  \
                 {:>6.2}x under keccak  ~{:.0} GiB",
                total as f64 / t as f64,
                projected_gib(t, sub_proofs),
            );
        }
    };
    println!(
        "\n★ THE MATRIX, RE-DERIVED (P in [{p_lo}, {p_hi}] MEASURED, {sub_proofs} \
         sub-proofs, two-term RSS {BYTES_PER_CELL} B/cell + {} MB/sub-proof)",
        BYTES_PER_SUB_PROOF / 1e6
    );
    println!(
        "   {:>28} {:>10}  keccak {:>11}  total {:>13}  {:>6.2}x  ~{:.0} GiB",
        "keccak (MEASURED, ours)",
        "n/a",
        keccak_perm + bitwise,
        total,
        1.0,
        projected_gib(total, sub_proofs),
    );
    // Blake keeps the byte-oriented residue AND the BITWISE table it looks up in.
    // The label and the figure follow the compiled round count, so a sweep
    // cannot leave this row naming one variant and pricing another.
    let blake_cells = (MAIN_COLUMNS + 3 * blake3_chip::bus_interactions().len().div_ceil(2)) as u64;
    let blake_label = if BLAKE3_ROUNDS == 6 {
        "BLAKE3-6r (MEASURED chip)"
    } else {
        "BLAKE3-7r (MEASURED chip)"
    };
    row(blake_label, blake_cells, residue, bitwise);
    // Field-native candidates delete both. Poseidon-original's 621 is wave 9's
    // measured column; RPO's 152 and Monolith's ~850 stay INHERITED estimates.
    row("Poseidon-orig (w9 MEASURED)", 621, residue_field_native, 0);
    row("RPO (INHERITED estimate)", 152, residue_field_native, 0);
    row("Monolith (INHERITED est.)", 850, residue_field_native, 0);
}

// =========================================================================
// The delegation topology, priced
// =========================================================================

/// ★ In-machine hosting vs an Airbender-style delegation circuit.
///
/// The question (user request): instead of the epoch verifier carrying an
/// `LFM_BLAKE3` AIR, put the compressions in a SEPARATE specialized circuit and
/// verify that circuit's proof — Airbender's blake2s delegation circuit does
/// ~19 proofs' Merkle work in one 2^20 instance.
///
/// The whole comparison is arithmetic over the same closed form the epoch's own
/// permutation count comes from ([`super::epoch_verify::blocks_at_rate`] and
/// the leaf/path/FRI decomposition), applied to the delegation proof's shape.
/// Every substituted input is named in the printout.
///
/// ⚠ What this CANNOT see: prover wall time, proof size on the wire, and the
/// engineering cost of a second circuit and its glue. It prices cells only.
#[test]
#[ignore]
fn the_delegation_topology_priced_against_in_machine_hosting() {
    use super::epoch_verify::{blocks_at_rate, group_leaf_felts, query_permutations_at_rate};
    use super::sub_proof::GroupShape;

    let inner = crate::recursion::Preset::Blowup8.options();
    let e = super::epoch_tests::real_epoch_with(inner.clone());

    // The epoch's widest leg supplies the two shape inputs this calculation
    // does not derive: how many composition parts a sub-proof carries, and what
    // one query's FRI leg costs. Both are INHERITED from a real proof rather
    // than assumed.
    let widest = e
        .legs
        .iter()
        .max_by_key(|l| l.verify.sub.deep.log2_trace_length)
        .expect("the epoch has legs");
    let parts = widest.verify.sub.deep.num_composition_parts;
    let queries = widest.verify.num_queries;
    let log2_blowup = inner.blowup_factor.trailing_zeros();

    /// Compressions to verify ONE sub-proof of the given geometry, at BLAKE3's
    /// rate 8 (two cells of message per compression — not the field-native
    /// chain's 4, see `epoch_verify::LFM_HASH_RATE_FELTS`).
    ///
    /// `Σ_groups blocks_at_rate(leaf felts) + groups × merkle_depth + FRI`, the
    /// same three terms `query_permutations_at_rate` sums, per query.
    fn verify_cost(
        groups: &[GroupShape],
        log2_trace: u32,
        log2_blowup: u32,
        fri_per_query: usize,
        queries: usize,
    ) -> (usize, usize) {
        let merkle_depth = (log2_trace + log2_blowup) as usize - 1;
        let leaves: usize = groups
            .iter()
            .map(|g| blocks_at_rate(group_leaf_felts(g), 8))
            .sum();
        let per_query = leaves + groups.len() * merkle_depth + fri_per_query;
        (per_query, per_query * queries)
    }

    let fri_per_query = widest.verify.fri.permutations_per_query();

    // --- the delegation circuit's two AIRs, at the epoch's own compression count.
    // The same quantity the matrix instrument brackets, and it was the same
    // hardcoded `192_000` here — independently unasserted, so the two could
    // have drifted apart as well as away from the truth. Derived from the
    // epoch's own shapes at BLAKE3's rate, which is what the closed form is
    // for.
    // The LEGS' term only — exact, where the spine's is merely bounded (it is
    // absorption-bound, so its rate-8 cost sits between 1.0x and 2.125x its
    // rate-17 one). Excluding the spine understates the compression count, so
    // every "extra cost of delegating" figure below is a LOWER bound and the
    // verdict is if anything understated.
    let compressions: usize = e
        .legs
        .iter()
        .map(|l| query_permutations_at_rate(&l.verify, 8))
        .sum();
    let log2_blake_trace = (compressions as u32).next_power_of_two().trailing_zeros(); // 18
    let blake_groups = vec![
        GroupShape {
            num_columns: cols::PREP_WIDTH,
            is_ext: false,
        },
        GroupShape {
            num_columns: MAIN_COLUMNS,
            is_ext: false,
        },
        GroupShape {
            num_columns: blake3_chip::bus_interactions().len().div_ceil(2),
            is_ext: true,
        },
        GroupShape {
            num_columns: parts,
            is_ext: true,
        },
    ];
    let bitwise_groups = vec![
        GroupShape {
            num_columns: crate::tables::bitwise::NUM_PRECOMPUTED_COLS,
            is_ext: false,
        },
        GroupShape {
            num_columns: 10,
            is_ext: false,
        },
        GroupShape {
            num_columns: 5,
            is_ext: true,
        },
        GroupShape {
            num_columns: parts,
            is_ext: true,
        },
    ];
    let (blake_pq, blake_total) = verify_cost(
        &blake_groups,
        log2_blake_trace,
        log2_blowup,
        fri_per_query,
        queries,
    );
    let (bw_pq, bw_total) = verify_cost(&bitwise_groups, 20, log2_blowup, fri_per_query, queries);

    let blake_aux = blake3_chip::bus_interactions().len().div_ceil(2);
    let cells_per_compression = MAIN_COLUMNS as u64 + 3 * blake_aux as u64;
    let delegation_trace =
        (compressions as f64 * 1.01871).ceil() as u64 * cells_per_compression + 26_214_400; // its own BITWISE table

    println!(
        "\n★ DELEGATION TOPOLOGY, at the epoch's {compressions} compressions\n\
         \x20 shared inputs (INHERITED from the epoch's 2^{} leg): {parts} composition parts, \
         {queries} queries, {fri_per_query} FRI compressions per query, blowup {}\n\n\
         \x20 IN-MACHINE   the epoch verifier carries LFM_BLAKE3 as one more AIR of its\n\
         \x20              multi-proof, {compressions} rows x {cells_per_compression} cells = {} cells.\n\
         \x20              Nothing else changes: chip heights ARE program shape here, so the\n\
         \x20              hash competes with nothing for space.\n\n\
         \x20 DELEGATED    (a) the delegation proof's own trace, LFM_BLAKE3 + BITWISE  {delegation_trace:>12} cells\n\
         \x20              (b) verifying it inside the epoch verifier:\n\
         \x20                  LFM_BLAKE3 AIR (2^{log2_blake_trace} rows, {MAIN_COLUMNS} main + {blake_aux} aux) \
         {blake_pq:>6}/query x {queries} = {blake_total:>8} compressions\n\
         \x20                  BITWISE AIR    (2^20 rows, 10 main + 5 aux)          \
         {bw_pq:>6}/query x {queries} = {bw_total:>8} compressions\n\
         \x20                  = {} extra compressions, i.e. {} extra cells in the\n\
         \x20                    epoch verifier's OWN blake AIR, on top of (a).\n",
        widest.verify.sub.deep.log2_trace_length,
        inner.blowup_factor,
        (compressions as f64 * 1.01871).ceil() as u64 * cells_per_compression,
        blake_total + bw_total,
        (blake_total + bw_total) as u64 * cells_per_compression,
    );
    println!(
        "\x20 VERDICT  delegation costs (a) + (b) where in-machine costs (a) alone, so it is a\n\
         \x20          net LOSS of {:.0}M cells ({:.0}% on top) at these shapes. The reason is\n\
         \x20          structural, not a tuning accident: the thing Airbender's delegation\n\
         \x20          circuit buys is moving hash work out of a FIXED-SIZE main circuit (a\n\
         \x20          2^20-cycle RISC-V trace). The LFM has no fixed-size box — every chip's\n\
         \x20          height is program shape — so its multi-AIR proof already IS the\n\
         \x20          delegation pattern, and a second proof only adds a verification.\n\
         \x20          The leaf term is what makes (b) large: a {MAIN_COLUMNS}-column AIR has a\n\
         \x20          {}-felt main leaf, {} compressions to absorb, {} times per query.",
        (blake_total + bw_total) as f64 * cells_per_compression as f64 / 1e6,
        100.0 * (blake_total + bw_total) as f64 * cells_per_compression as f64
            / delegation_trace as f64,
        group_leaf_felts(&blake_groups[1]),
        blocks_at_rate(group_leaf_felts(&blake_groups[1]), 8),
        queries,
    );
}
