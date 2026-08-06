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
//! prove+verify makes the 769 constraints and the 1,259 interactions load
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

use super::blake3::{CANONICAL_VECTORS, blake3_compress_6round};
use super::blake3_chip::{
    self, Blake3LfmConstraints, Blake3Operation, IN_WORDS, MAIN_COLUMNS, NUM_CONSTRAINTS,
    OUT_WORDS, cols,
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
                h: v.h,
                m: v.m,
                t: v.t,
                block_len: v.block_len,
                flags: v.flags,
            }
        })
        .collect()
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
        let inputs = op.input_words();
        for j in 0..IN_WORDS {
            table.set_u64(row, mirror::ADDR, op.in_addr[j]);
            for (l, v) in lanes(&inputs, j).into_iter().enumerate() {
                table.set_u64(row, mirror::V0 + l, v);
            }
            table.set_fe(row, mirror::SEND_MULT, FE::one());
            row += 1;
        }
        let outputs = op.output_words();
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
    hist.add_ops(&blake3_chip::bitwise_ops_for(ops));
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

/// The blake column's per-compression cell law, on our stack.
///
/// `main + 3·aux` with `aux = ceil(interactions / 2)` is `airs.rs`'s census
/// formula — the same instrument that produced the keccak and Poseidon columns,
/// so the three are comparable by construction rather than by argument.
#[test]
fn the_hosted_chip_costs_4946_base_field_equivalent_cells_per_compression() {
    let interactions = blake3_chip::bus_interactions().len();
    let aux = interactions.div_ceil(2);

    // Column budget, block by block, so a layout change cannot move the total
    // silently.
    assert_eq!(cols::PREP_WIDTH, 16, "preprocessed prefix");
    assert_eq!(cols::G - cols::IN, 112, "input bytes");
    assert_eq!(cols::OUT - cols::G, 2_880, "48 G-blocks × 60 cells");
    assert_eq!(
        cols::NUM_COLUMNS - cols::OUT,
        64,
        "feed-forward output bytes"
    );
    assert_eq!(cols::NUM_COLUMNS, 3_072);
    assert_eq!(MAIN_COLUMNS, 3_056);

    // Interaction budget, group by group.
    assert_eq!(IN_WORDS + OUT_WORDS, 11, "LfmMem tokens");
    assert_eq!(interactions, 11 + 832 + 384 + 32);
    assert_eq!(interactions, 1_259);
    assert_eq!(aux, 630);

    assert_eq!(MAIN_COLUMNS + 3 * aux, 4_946, "base-field-equivalent cells");

    // #903's syscall variant, for the delta the hosting buys: 3,219 main and
    // 1,397 interactions (699 aux) = 5,316. The difference is entirely I/O.
    assert_eq!(3_219 + 3 * 1_397usize.div_ceil(2), 5_316);
}

/// Every constraint index is emitted exactly once, and the count is the one the
/// module documents. #903 emits 814; the 45 address-derivation constraints have
/// no counterpart here.
#[test]
fn the_chip_emits_769_constraints_at_degree_3() {
    assert_eq!(NUM_CONSTRAINTS, 769);
    assert_eq!(814 - 45, NUM_CONSTRAINTS, "vs #903's syscall variant");
    assert_eq!(NUM_CONSTRAINTS, 3 * 96 + 96 + 4 * 96 + 1);

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
        let expected = blake3_compress_6round(&op.h, &op.m, op.t, op.block_len, op.flags);
        assert_eq!(expected, CANONICAL_VECTORS[row].out, "op {row} is a vector");
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

    // The BITWISE feed is exactly the senders' count: 1,248 per compression,
    // with no address-shaped lookups.
    assert_eq!(
        blake3_chip::bitwise_ops_for(&ops).len(),
        ops.len() * 1_248,
        "per-compression BITWISE lookup count"
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

/// ★ The blake column and the residue split, on the real epoch verifier.
///
/// `#[ignore]`d for the same reason `wrap_tests::the_wrap_census_at_blowup_8`
/// is: it proves a real inner epoch at blowup 8 and then emits ~2.25M
/// instructions. Run with
/// `cargo test -p lambda-vm-prover --lib the_blake_column -- --ignored --nocapture`.
///
/// Everything printed here is arithmetic over three inputs, each labelled:
/// the per-compression cost MEASURED above, the permutation count from wave 8's
/// rate-parameterised closed form (re-run here rather than quoted), and the
/// residue read off this run's own census.
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

    let share = |names: &[&str]| -> u64 {
        census
            .iter()
            .filter(|c| names.contains(&c.name))
            .map(|c| c.main_cells() + 3 * c.aux_cells())
            .sum()
    };
    let hash = share(&["LFM_KECCAK", "KECCAK_RND", "KECCAK_RC", "BITWISE"]);
    let residue = total - hash;

    println!(
        "\n★ EPOCH {profile}, inner blowup {}, {} queries",
        inner.blowup_factor, inner.fri_number_of_queries
    );
    println!("   total   {total:>16}  base-field-equivalent cells (MEASURED, this run)");
    println!(
        "   keccak  {hash:>16}  ({:.2}%)",
        100.0 * hash as f64 / total as f64
    );
    println!(
        "   residue {residue:>16}  ({:.2}%)",
        100.0 * residue as f64 / total as f64
    );

    // ---- permutations, at both rates, from the closed form over the shapes.
    let legs_17: usize = e.legs.iter().map(|l| query_permutations(&l.verify)).sum();
    let legs_8: usize = e
        .legs
        .iter()
        .map(|l| query_permutations_at_rate(&l.verify, 8))
        .sum();
    let emitted = super::wrap_tests::permutations(&program);
    let spine_perms = super::wrap_tests::permutations(&spine);
    println!(
        "\n   permutations: emitted {emitted} = spine {spine_perms} + legs {}\n   \
         closed form  legs @ rate 17 (keccak) {legs_17}, @ rate 8 (blake / field-native) {legs_8}",
        emitted - spine_perms
    );
    assert_eq!(
        emitted - spine_perms,
        legs_17,
        "the rate-17 closed form must reproduce the emitted legs"
    );
    // The spine is absorption-bound, so at rate 8 it lies between 1.0x and
    // 2.125x its rate-17 cost — wave 8's interval, restated on this run's own
    // spine count rather than quoted.
    let p_lo = legs_8 + spine_perms;
    let p_hi = legs_8 + (spine_perms as f64 * 17.0 / 8.0).ceil() as usize;
    println!("   P at rate 8 in [{p_lo}, {p_hi}]  (spine bounded, legs exact)");

    // ---- the byteswap gadget: exactly the 64-bit decompositions.
    //
    // `sample_u64_pow2` asserts nbits <= 32 and every other production
    // `bit_dec` site is 32 bits or a Merkle depth, so a 64-bit decomposition in
    // this program IS a `felt_be_halves` and nothing else. Counted rather than
    // reasoned about, with the whole histogram printed so a new 64-bit caller
    // would be visible instead of silently folded in.
    let mut hist = std::collections::BTreeMap::<usize, usize>::new();
    for i in &program.instrs {
        if let Instr::BitDec { bits, .. } = i {
            *hist.entry(bits.len()).or_default() += 1;
        }
    }
    println!("\n   BitDec width histogram: {hist:?}");
    let swaps = hist.get(&64).copied().unwrap_or(0);

    let width = |name: &str| -> (u64, u64) {
        let c = census
            .iter()
            .find(|c| c.name == name)
            .expect("chip in census");
        (c.main_cols as u64, c.aux_cols as u64)
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

    println!(
        "\n   byteswap gadget: {swaps} felts x (1 BitDec + 64 BALU)\n   \
         LFM_BITDEC {bd_rows} real rows ({} padded), width {bitdec_m} main / {bitdec_a} aux\n   \
         LFM_BALU   {ba_rows} real rows ({} padded), width {balu_m} main / {balu_a} aux\n   \
         gadget cells, unpadded closed form : {unpadded}\n   \
         gadget cells, padding-aware delta  : {}  (the two chips' padded totals, before {before} after {after})",
        padded_rows(bd_rows),
        padded_rows(ba_rows),
        before - after,
    );

    println!(
        "\n   residue, byte-oriented (blake keeps the gadget) : {residue}\n   \
         residue, field-native (gadget deleted)          : {}  (-{:.2}%)",
        residue - (before - after),
        100.0 * (before - after) as f64 / residue as f64,
    );
}
