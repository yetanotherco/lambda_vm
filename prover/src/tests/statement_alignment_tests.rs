//! ★★ Every value a WHIR proof absorbs starts on a field element boundary.
//!
//! ```text
//! cargo test -p lambda-vm-prover --lib statement_alignment
//! ```
//!
//! # The property, and what it is NOT
//!
//! The transcript hashes BYTES. The algebraic configuration's sponge re-slices
//! everything absorbed since its last squeeze into field elements every 8 bytes
//! (`crypto::hash::rpx::sponge_leaf_bytes`), so a value absorbed at an offset
//! that is not a multiple of 8 straddles two of them — which a field-machine
//! verifier replaying the transcript can only reproduce by decomposing bits.
//!
//! The property is therefore about OFFSETS, over every shape: *every absorb
//! after the statement lands at a window offset that is a multiple of 8*. It is
//! **not** "the statement's fixed prefix is a multiple of 8": the roots do not
//! follow the fixed prefix, they follow two variable-length fields, and a
//! statement padded to a round fixed prefix leaves them at
//! `(|public_output| + |table_num_vars|) mod 8` — 2 mod 8 at the shape this
//! system runs. That arithmetic is why the pad is computed from an accumulated
//! length rather than written as a constant, and this file is where the claim
//! can fail.
//!
//! # Two tests, because one of them cannot see the other's failure
//!
//! * [`the_statement_ends_on_a_field_element_boundary`] sweeps the shapes
//!   against a recording transcript. It sees every shape and no protocol.
//! * [`every_absorb_of_a_real_prove_is_field_element_aligned`] drives a real
//!   `multi_prove` and sees one shape and the whole protocol. It is the one that
//!   can report a misalignment the statement padding does not fix.
//!
//! # Why the recorder is validated rather than trusted
//!
//! [`WindowRecorder`] has to know where a window ENDS, and only a squeeze ends
//! one — so it mirrors `DefaultTranscript`'s duplex output buffer instead of
//! guessing. A mirror that drifts would draw different challenges, so the
//! end-to-end test first proves the same fixture twice, once through the
//! production transcript and once through the recorder, and requires the two
//! proofs to serialise to the same bytes. The offsets it reports are then the
//! production stream's offsets, not a model's.

use digest::Digest;
use math::field::element::FieldElement;
use math::field::traits::HasDefaultTranscript;
use math::traits::AsBytes;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::fiat_shamir::transcript_hash::{
    HasTranscriptHash, KeccakTranscriptHash, TranscriptHash,
};
use multilinear::whir_chain::{ChainConfig, GrindBits};
use multilinear::whir_hash::{KeccakWhir, RpxWhir, WhirHash};
use stark::multilinear_air::Uniforms;
use stark::multilinear_table::{self, CommittedTable, CommittedTables, TableLayout};
use stark::proof::options::ProofOptions;
use stark::traits::AIR;

use crate::statement::{FELT_BYTES, NUM_TABLE_KINDS, statement_padding};
use crate::tables::eq::{EqConstraints, EqOperation, generate_eq_trace};
use crate::test_utils::{ConcreteVmAir, E, F, create_eq_air};
use crate::{RuntimePageRange, TableCounts};

/// Bytes one squeeze hands the transcript, and therefore the offset a fresh
/// window opens at. `DefaultTranscript::sample` finalize-resets the sponge and
/// re-absorbs its own 32-byte output, so a window never opens empty.
///
/// It is private to `default_transcript`, so it is mirrored here rather than
/// imported — and a wrong value here does not silently weaken this file: the
/// duplex buffer would refill at the wrong time, the challenge stream would
/// diverge, and
/// [`every_absorb_of_a_real_prove_is_field_element_aligned`]'s byte comparison
/// against the production transcript would fail.
const SQUEEZE_LEN: usize = 32;

// -------------------------------------------------------------------------
// The recording transcript
// -------------------------------------------------------------------------

/// A transcript that records the window offset of every absorb.
///
/// Absorption is DELEGATED — the inner `DefaultTranscript` is what hashes, so
/// this is the production sponge with a tape attached. Only the duplex output
/// buffer is mirrored, because that is the only part that tells the recorder
/// when a window ends.
struct WindowRecorder<T: TranscriptHash> {
    inner: DefaultTranscript<E, T>,
    out_buf: [u8; SQUEEZE_LEN],
    out_pos: usize,
    /// Bytes absorbed since the sponge was last reset by a squeeze.
    window: usize,
    /// `(window offset, length)` of every absorb, in call order.
    absorbs: Vec<(usize, usize)>,
}

/// Of the absorbs recorded after the first `from` — which is how a caller says
/// "everything after the statement" — the ones that do not start on a field
/// element boundary.
fn misaligned(absorbs: &[(usize, usize)], from: usize) -> Vec<(usize, usize)> {
    absorbs[from..]
        .iter()
        .copied()
        .filter(|(offset, _)| !offset.is_multiple_of(FELT_BYTES))
        .collect()
}

impl<T: TranscriptHash> WindowRecorder<T> {
    fn new() -> Self {
        Self {
            inner: DefaultTranscript::<E, T>::new(&[]),
            out_buf: [0u8; SQUEEZE_LEN],
            out_pos: SQUEEZE_LEN,
            window: 0,
            absorbs: Vec::new(),
        }
    }

    fn record(&mut self, len: usize) {
        self.absorbs.push((self.window, len));
        self.window += len;
        // Same invalidation the inner transcript performs: a challenge drawn
        // after an absorb must depend on it.
        self.out_pos = SQUEEZE_LEN;
    }

    /// `DefaultTranscript::next_sample_u64`, mirrored. The squeeze itself is the
    /// inner transcript's, so the bytes are production's; what is duplicated is
    /// only the bookkeeping that says when one happens.
    fn next_u64(&mut self) -> u64 {
        if self.out_pos + 8 > SQUEEZE_LEN {
            self.out_buf = self.inner.sample();
            self.out_pos = 0;
            // A squeeze finalize-resets the sponge and re-absorbs its own
            // output, so the new window opens holding those 32 bytes.
            self.window = SQUEEZE_LEN;
        }
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&self.out_buf[self.out_pos..self.out_pos + 8]);
        self.out_pos += 8;
        u64::from_be_bytes(bytes)
    }
}

impl<T: TranscriptHash> HasTranscriptHash for WindowRecorder<T> {
    type Hash = T;
}

impl<T: TranscriptHash> IsTranscript<E> for WindowRecorder<T> {
    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.record(new_bytes.len());
        self.inner.append_bytes(new_bytes);
    }

    fn append_field_element(&mut self, element: &FieldElement<E>) {
        // Counted per `update`, the same unit the inner transcript counts, so a
        // serialisation that streamed an element in pieces would show up here
        // as several absorbs rather than one.
        let mut chunks: Vec<usize> = Vec::new();
        element.stream_bytes(&mut |b| chunks.push(b.len()));
        for len in chunks {
            self.record(len);
        }
        self.inner.append_field_element(element);
    }

    fn state(&self) -> [u8; 32] {
        // Finalizes a clone: no reset, no re-absorb, so the window does not move.
        self.inner.state()
    }

    fn sample_field_element(&mut self) -> FieldElement<E> {
        E::sample_field_element_from(|| self.next_u64())
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        assert!(upper_bound > 0, "upper_bound must be greater than 0");
        let threshold = upper_bound.wrapping_neg() % upper_bound;
        loop {
            let candidate = self.next_u64();
            if candidate >= threshold {
                return candidate % upper_bound;
            }
        }
    }
}

// -------------------------------------------------------------------------
// The closed forms, written from the field lists rather than from the code
// -------------------------------------------------------------------------

const DIGEST: usize = 32;
const U64: usize = 8;
/// `log_blowup`, `log_folding`, `num_queries`.
const CONFIG: usize = 3 * U64;
/// `grind.folding`, `grind.ood`, `grind.query`, absorbed as one 3-byte value.
const GRIND_TRAILER: usize = 3;

/// What a statement's fields add up to, what it therefore absorbs in total, and
/// how many times it calls the transcript.
#[derive(Clone, Copy, Debug)]
struct Expected {
    /// The statement's own fields, before any padding.
    body: usize,
    /// `body` plus the pad, which is what the transcript should have taken.
    total: usize,
    /// Absorb calls, padding included.
    calls: usize,
}

impl Expected {
    fn new(body: usize, calls: usize) -> Self {
        Self {
            body,
            total: body + statement_padding(body),
            calls,
        }
    }
}

/// What an epoch statement absorbs, field by field.
///
/// Deliberately a SECOND derivation: the production function accumulates its
/// length beside its own absorbs, and this adds up the fields it is supposed to
/// have. The two can only be compared here, and a field added to one and not
/// the other fails here.
fn epoch_expected(public_output: usize, table_num_vars: usize) -> Expected {
    let body = crate::multilinear_continuation::MULTILINEAR_EPOCH_TAG.len()
        + DIGEST
        + U64 // epoch_label
        + U64 // |public_output|
        + public_output
        + NUM_TABLE_KINDS * U64
        + U64 // |table_num_vars|
        + table_num_vars
        + CONFIG
        + GRIND_TRAILER;
    let calls = 1 // tag
        + 1 // elf digest
        + 1 // epoch label
        + 1 // |public_output|
        + 1 // public_output
        + NUM_TABLE_KINDS
        + 1 // |table_num_vars|
        + 1 // table_num_vars
        + 3 // config
        + 1 // grind trailer
        + 1; // padding, ALWAYS
    Expected::new(body, calls)
}

/// The same for the cross-epoch statement. `page_bases` is eight bytes an entry
/// and cannot move the alignment; `table_num_vars` can.
fn global_expected(page_bases: usize, table_num_vars: usize) -> Expected {
    let body = crate::multilinear_continuation::MULTILINEAR_GLOBAL_TAG.len()
        + DIGEST
        + U64 // num_epochs
        + U64 // num_private_input_pages
        + U64 // |page_bases|
        + page_bases * U64
        + U64 // |table_num_vars|
        + table_num_vars
        + CONFIG
        + GRIND_TRAILER;
    let calls = 1 + 1 + 1 + 1 + 1 + page_bases + 1 + 1 + 3 + 1 + 1;
    Expected::new(body, calls)
}

/// And for the monolithic multilinear statement. Its ranges are sixteen bytes
/// an entry, so they too are alignment-neutral.
fn monolithic_expected(
    public_output: usize,
    runtime_page_ranges: usize,
    table_num_vars: usize,
) -> Expected {
    let body = crate::statement::MULTILINEAR_TAG.len()
        + DIGEST
        + U64 // |public_output|
        + public_output
        + NUM_TABLE_KINDS * U64
        + U64 // num_private_input_pages
        + U64 // |runtime_page_ranges|
        + runtime_page_ranges * 2 * U64
        + U64 // |table_num_vars|
        + table_num_vars
        + CONFIG
        + GRIND_TRAILER;
    let calls =
        1 + 1 + 1 + 1 + NUM_TABLE_KINDS + 1 + 1 + 2 * runtime_page_ranges + 1 + 1 + 3 + 1 + 1;
    Expected::new(body, calls)
}

fn config() -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 2,
        num_queries: 3,
        grind: GrindBits::default(),
    }
}

fn counts() -> TableCounts {
    TableCounts {
        cpu: 1,
        lt: 2,
        memw: 3,
        memw_aligned: 4,
        load: 5,
        mul: 6,
        dvrm: 7,
        shift: 8,
        branch: 9,
        memw_register: 10,
        eq: 11,
        bytewise: 12,
        store: 13,
        cpu32: 14,
    }
}

/// The shapes the sweep runs, as `(|public_output|, |table_num_vars|)`.
///
/// `(0, 34)` is what this system actually proves — and the shape a constant pad
/// would have left at 2 mod 8. The others are chosen so the residue of
/// `|public_output| + |table_num_vars|` differs: a sweep in which every shape
/// needed the same pad would pass with the pad hard-coded to that value.
const SHAPES: &[(usize, usize)] = &[(0, 34), (7, 34), (32, 34), (33, 35), (2, 1), (0, 1)];

// -------------------------------------------------------------------------
// (1) The statement, over every shape
// -------------------------------------------------------------------------

#[test]
fn the_statement_ends_on_a_field_element_boundary() {
    let mut residues_seen = std::collections::BTreeSet::new();

    for &(po_len, tnv_len) in SHAPES {
        let public_output = vec![0xABu8; po_len];
        let table_num_vars = vec![20u8; tnv_len];

        let mut rec = WindowRecorder::<KeccakTranscriptHash>::new();
        crate::multilinear_continuation::absorb_epoch(
            &mut rec,
            &[7u8; 32],
            &public_output,
            &counts(),
            3,
            &table_num_vars,
            &config(),
        );

        let expected = epoch_expected(po_len, tnv_len);
        residues_seen.insert(expected.body % FELT_BYTES);

        // ★ The property first, so a mutation reports the property. The two
        // assertions under it are corroborating derivations, not the claim.
        assert!(
            rec.window.is_multiple_of(FELT_BYTES),
            "epoch statement at (po {po_len}, tnv {tnv_len}) ended at byte \
             {} = {} mod {FELT_BYTES}: whatever is absorbed next straddles two \
             field elements",
            rec.window,
            rec.window % FELT_BYTES,
        );
        assert_eq!(
            rec.window, expected.total,
            "epoch statement at (po {po_len}, tnv {tnv_len}) absorbed a different \
             number of BYTES than its field list implies",
        );
        assert_eq!(
            rec.absorbs.len(),
            expected.calls,
            "epoch statement at (po {po_len}, tnv {tnv_len}) absorbed a different \
             number of times than its field list implies — a padding absorb that \
             is skipped when the pad is empty shows up here",
        );

        // And the thing that actually follows a statement: the roots.
        let before = rec.absorbs.len();
        for _ in 0..4 {
            rec.append_bytes(&[0u8; 32]);
        }
        assert_eq!(
            misaligned(&rec.absorbs, before),
            Vec::new(),
            "a root absorbed after the epoch statement at (po {po_len}, tnv \
             {tnv_len}) does not start on a field element boundary",
        );
    }

    // ⚠ A sweep whose shapes all need the same pad would pass with the pad
    // written as that constant. This says the sweep is not that sweep.
    assert!(
        residues_seen.len() > 1,
        "every shape in the sweep has the same residue, so the sweep cannot \
         tell a computed pad from a constant one",
    );
}

#[test]
fn the_cross_epoch_statement_ends_on_a_field_element_boundary() {
    for &(pages, tnv_len) in &[(0usize, 15usize), (35, 50), (3, 18), (1, 16), (35, 51)] {
        let page_bases: Vec<u64> = (0..pages as u64).map(|i| i * 4096).collect();
        let table_num_vars = vec![21u8; tnv_len];

        let mut rec = WindowRecorder::<KeccakTranscriptHash>::new();
        crate::multilinear_continuation::absorb_global(
            &mut rec,
            &[9u8; 32],
            15,
            0,
            &page_bases,
            &table_num_vars,
            &config(),
        );

        let expected = global_expected(pages, tnv_len);
        assert!(
            rec.window.is_multiple_of(FELT_BYTES),
            "cross-epoch statement at (pages {pages}, tnv {tnv_len}) ended at \
             byte {} = {} mod {FELT_BYTES}",
            rec.window,
            rec.window % FELT_BYTES,
        );
        assert_eq!(
            (rec.absorbs.len(), rec.window),
            (expected.calls, expected.total),
            "cross-epoch statement at (pages {pages}, tnv {tnv_len})",
        );
    }
}

#[test]
fn the_monolithic_statement_ends_on_a_field_element_boundary() {
    for &(po_len, ranges, tnv_len) in &[
        (0usize, 0usize, 34usize),
        (5, 2, 34),
        (33, 1, 35),
        (4, 3, 7),
    ] {
        let public_output = vec![0x5Au8; po_len];
        let runtime_page_ranges: Vec<RuntimePageRange> = (0..ranges)
            .map(|i| RuntimePageRange {
                base: i as u64 * 4096,
                count: 1,
            })
            .collect();
        let table_num_vars = vec![22u8; tnv_len];

        let mut rec = WindowRecorder::<KeccakTranscriptHash>::new();
        crate::multilinear_prove::absorb(
            &mut rec,
            &[4u8; 32],
            &public_output,
            &counts(),
            2,
            &runtime_page_ranges,
            &table_num_vars,
            &config(),
        );

        let expected = monolithic_expected(po_len, ranges, tnv_len);
        assert!(
            rec.window.is_multiple_of(FELT_BYTES),
            "monolithic statement at (po {po_len}, ranges {ranges}, tnv \
             {tnv_len}) ended at byte {} = {} mod {FELT_BYTES}",
            rec.window,
            rec.window % FELT_BYTES,
        );
        assert_eq!(
            (rec.absorbs.len(), rec.window),
            (expected.calls, expected.total),
            "monolithic statement at (po {po_len}, ranges {ranges}, tnv {tnv_len})",
        );
    }
}

/// ★ The pad is a function of the WHOLE length, not of the fixed prefix.
///
/// This is the arithmetic that killed the constant-pad design, written as an
/// assertion so it cannot be forgotten: at the shape this system runs, padding
/// the fixed prefix to a multiple of 8 leaves the roots at 2 mod 8.
#[test]
fn padding_only_the_fixed_prefix_would_not_align_the_roots() {
    // The shape this system runs: no public output, 34 tables.
    let (po_len, tnv_len) = (0usize, 34usize);
    assert!(
        epoch_expected(po_len, tnv_len)
            .total
            .is_multiple_of(FELT_BYTES)
    );

    // The fixed prefix is the body with both variable fields empty.
    let fixed = epoch_expected(0, 0).body;
    assert_eq!(fixed, 237, "the epoch statement's fixed prefix");
    let fixed_rounded = fixed + statement_padding(fixed);
    assert_eq!(fixed_rounded, 240);

    assert_eq!(
        (fixed_rounded + po_len + tnv_len) % FELT_BYTES,
        2,
        "a pad computed from the fixed prefix alone leaves the roots at 2 mod \
         {FELT_BYTES} at the shape we run: it would move every pinned constant \
         and align nothing",
    );
}

// -------------------------------------------------------------------------
// (2) A real prove, every absorb
// -------------------------------------------------------------------------

fn eq_table() -> (
    &'static ConcreteVmAir<EqConstraints>,
    Vec<Vec<FieldElement<F>>>,
) {
    let ops = vec![
        EqOperation::new(7, 7, false),
        EqOperation::new(7, 9, false),
        EqOperation::new(3, 3, true),
        EqOperation::new(3, 5, true),
    ];
    let columns: Vec<Vec<FieldElement<F>>> = generate_eq_trace(&ops).columns_main();
    // Leaked so the layout borrows nothing from a temporary; this is a test
    // binary and the leak is one AIR.
    let air: &'static ConcreteVmAir<EqConstraints> = Box::leak(Box::new(create_eq_air(
        &ProofOptions::default_test_options(),
    )));
    (air, columns)
}

fn committed<H: WhirHash>(
    air: &'static ConcreteVmAir<EqConstraints>,
    columns: &[Vec<FieldElement<F>>],
    cfg: &ChainConfig,
) -> CommittedTables<'static, F, E, H> {
    let layout = TableLayout::<F, E>::new(
        air.constraint_program(),
        air.constraints_meta(),
        air.bus_interactions(),
        columns.len(),
        columns[0].len().trailing_zeros() as usize,
        Uniforms::default(),
    )
    .expect("layout");
    let table =
        CommittedTable::from_layout(layout, |col| columns[col as usize].clone()).expect("table");
    CommittedTables::<_, _, H>::commit(vec![table], cfg).expect("commit")
}

/// The epoch statement this fixture proves under, absorbed into `t`, and how
/// many absorbs it took.
fn seed_with_statement(t: &mut impl IsTranscript<E>, table_num_vars: &[u8], po: &[u8]) -> usize {
    crate::multilinear_continuation::absorb_epoch(
        t,
        &[1u8; 32],
        po,
        &counts(),
        0,
        table_num_vars,
        &config(),
    );
    epoch_expected(po.len(), table_num_vars.len()).calls
}

/// ⚠ The trace is generated ONCE and handed to both proves.
///
/// `generate_eq_trace` emits its rows in `HashMap` iteration order, so two
/// calls produce two row orders, two commitments and two proofs — W1's finding
/// F2, and the reason the byte gate sorts its rows canonically. Regenerating it
/// per prove would make this comparison fail for a reason that has nothing to do
/// with the recorder, which is exactly what it did on the first run of this test.
fn prove_through_recorder<H: WhirHash>(
    air: &'static ConcreteVmAir<EqConstraints>,
    columns: &[Vec<FieldElement<F>>],
    po: &[u8],
) -> (Vec<u8>, WindowRecorder<H::Transcript>)
where
    WindowRecorder<H::Transcript>: IsTranscript<E>,
{
    let cfg = config();
    let num_vars = columns[0].len().trailing_zeros() as u8;
    let committed = committed::<H>(air, columns, &cfg);

    let mut rec = WindowRecorder::<H::Transcript>::new();
    let statement_absorbs = seed_with_statement(&mut rec, &[num_vars], po);
    assert_eq!(rec.absorbs.len(), statement_absorbs);

    let proof = multilinear_table::multi_prove(&committed, &cfg, &mut rec).expect("prove");
    let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .expect("serialize")
        .to_vec();
    (bytes, rec)
}

fn prove_through_production<H: WhirHash>(
    air: &'static ConcreteVmAir<EqConstraints>,
    columns: &[Vec<FieldElement<F>>],
    po: &[u8],
) -> Vec<u8> {
    let cfg = config();
    let num_vars = columns[0].len().trailing_zeros() as u8;
    let committed = committed::<H>(air, columns, &cfg);

    let mut t = DefaultTranscript::<E, H::Transcript>::new(&[]);
    seed_with_statement(&mut t, &[num_vars], po);

    let proof = multilinear_table::multi_prove(&committed, &cfg, &mut t).expect("prove");
    rkyv::to_bytes::<rkyv::rancor::Error>(&proof)
        .expect("serialize")
        .to_vec()
}

/// ★★ The end-to-end claim, on the production stream.
///
/// Run under both configurations. The offsets are a property of the CALL
/// SEQUENCE rather than of the sponge, so the two arms are expected to agree —
/// which is exactly why running both is worth its seconds: a disagreement would
/// mean one configuration absorbs something the other does not.
#[test]
fn every_absorb_of_a_real_prove_is_field_element_aligned() {
    // Two public-output lengths, so the statement ends at a different length in
    // each and the pad differs: at `[]` the pad is 2, at `[0, 0]` it is 0 — the
    // shape where the padding absorb is empty and must still be made.
    let (air, columns) = eq_table();

    for po in [vec![], vec![0u8, 0u8]] {
        for arm in ["keccak", "rpx"] {
            let (recorded, rec, produced) = match arm {
                "keccak" => {
                    let (b, r) = prove_through_recorder::<KeccakWhir>(air, &columns, &po);
                    (
                        b,
                        r.absorbs.clone(),
                        prove_through_production::<KeccakWhir>(air, &columns, &po),
                    )
                }
                _ => {
                    let (b, r) = prove_through_recorder::<RpxWhir>(air, &columns, &po);
                    (
                        b,
                        r.absorbs.clone(),
                        prove_through_production::<RpxWhir>(air, &columns, &po),
                    )
                }
            };

            // The recorder is production's sponge with a tape attached, and this
            // is what says so. A mirrored duplex buffer that drifted would draw
            // different challenges and land here.
            assert_eq!(
                keccak_line(&recorded),
                keccak_line(&produced),
                "the {arm} arm's recorded prove is not the production prove \
                 (po {} bytes): the recorder's mirror of the duplex buffer has \
                 drifted, so its offsets describe some other transcript",
                po.len(),
            );

            let statement_absorbs = epoch_expected(po.len(), 1).calls;
            let bad = misaligned(&rec, statement_absorbs);
            assert!(
                bad.is_empty(),
                "the {arm} arm absorbed {} of {} values at an offset that is not \
                 a multiple of {FELT_BYTES} (po {} bytes). First few \
                 (offset, len): {:?}",
                bad.len(),
                rec.len() - statement_absorbs,
                po.len(),
                &bad[..bad.len().min(8)],
            );

            // A read that can fail in the other direction: a prove that absorbed
            // nothing after the statement would satisfy the emptiness above.
            // "None misaligned" is a statement only beside the count it is out of.
            assert!(
                rec.len() > statement_absorbs + 8,
                "the {arm} arm absorbed only {} values in total, so the \
                 alignment assertion above is about almost nothing",
                rec.len(),
            );
            println!(
                "ALIGNED {arm} po={} {} absorbs after the statement, 0 misaligned",
                po.len(),
                rec.len() - statement_absorbs,
            );
        }
    }
}

fn keccak_line(bytes: &[u8]) -> String {
    crypto::hash::platform_keccak::PlatformKeccak256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
