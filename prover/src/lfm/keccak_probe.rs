//! R1a probe: prove real `keccak-f[1600]` permutations through the UNCHANGED
//! production `KECCAK_RND` + `KECCAK_RC` + `BITWISE` AIRs, driven by
//! [`super::keccak_adapter`] instead of the VM-coupled `KECCAK` core chip.
//!
//! This is the entry gate for hosting the keccak table family inside the LFM
//! recursion machine's AIR set: it establishes that the family's only coupling
//! to the VM is the core chip's two `Keccak` bus tokens, and that a chip owning
//! nothing but those tokens is a sufficient driver.

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use stark::constraints::builder::EmptyConstraints;
use stark::lookup::{AirWithBuses, AuxiliaryTraceBuildData, NullBoundaryConstraintBuilder};
use stark::proof::options::{GoldilocksCubicProofOptions, ProofOptions};
use stark::proof::view::MultiProofView;
use stark::prover::{IsStarkProver, Prover};
use stark::trace::TraceTable;
use stark::traits::AIR;
use stark::verifier::{IsStarkVerifier, Verifier};

use crate::tables::types::{FE, FEE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::tables::{bitwise, keccak_rc, keccak_rnd};
use crate::test_utils::{create_bitwise_air, create_keccak_rc_air, create_keccak_rnd_air};

use super::keccak_adapter::{self, KeccakAdapterOperation, cols};

type F = GoldilocksField;
type E = GoldilocksExtension;
type AdapterAir = AirWithBuses<F, E, NullBoundaryConstraintBuilder, (), EmptyConstraints>;
type DynAir<'a> = &'a dyn AIR<Field = F, FieldExtension = E, PublicInputs = ()>;

const PROBE_TAG: &[u8] = b"LFM_R1A_KECCAK_PROBE_V1";

fn options() -> ProofOptions {
    GoldilocksCubicProofOptions::with_blowup(2).expect("probe options")
}

fn transcript() -> DefaultTranscript<E> {
    let mut t = DefaultTranscript::<E>::new(&[]);
    t.append_bytes(PROBE_TAG);
    t
}

fn adapter_air(opts: &ProofOptions) -> AdapterAir {
    AirWithBuses::new(
        cols::NUM_COLUMNS,
        AuxiliaryTraceBuildData {
            interactions: keccak_adapter::bus_interactions(),
        },
        opts,
        1,
        EmptyConstraints,
    )
    .with_name("KECCAK_ADAPTER")
}

/// Three permutations, distinct nontrivial inputs, distinct tags.
///
/// Tags are `row + 1` here. The production LFM adapter will source them from
/// preprocessed program data — see the tag-uniqueness note on
/// [`super::keccak_adapter`] and `duplicate_tag_output_swap_accepts_demonstrating_hazard`.
fn probe_ops() -> Vec<KeccakAdapterOperation> {
    (0..3u64)
        .map(|i| {
            let mut input = [0u64; 25];
            for (lane, slot) in input.iter_mut().enumerate() {
                *slot = (lane as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i.wrapping_mul(0xD1B5_4A32_D192_ED03))
                    ^ 0x0123_4567_89AB_CDEF;
            }
            KeccakAdapterOperation { tag: i + 1, input }
        })
        .collect()
}

/// The four traces, in AIR order: adapter, KECCAK_RND, KECCAK_RC, BITWISE.
fn build_traces(ops: &[KeccakAdapterOperation]) -> [TraceTable<F, E>; 4] {
    let adapter = keccak_adapter::generate_adapter_trace(ops);
    let rnd = keccak_rnd::generate_keccak_rnd_trace(&keccak_adapter::round_operations(ops));

    let mut rc = keccak_rc::generate_keccak_rc_trace();
    keccak_rc::update_multiplicities(&mut rc, ops.len());

    let mut hist = bitwise::BitwiseHistogram::new();
    hist.add_ops(&keccak_adapter::bitwise_ops_for(ops));
    let mut bw = bitwise::generate_bitwise_trace();
    hist.fill_multiplicities(&mut bw);

    [adapter, rnd, rc, bw]
}

/// Prove the four-AIR set over `traces`, which the caller may have tampered
/// with after generation.
fn prove_traces(
    opts: &ProofOptions,
    adapter: &AdapterAir,
    traces: &mut [TraceTable<F, E>; 4],
) -> Result<stark::proof::stark::MultiProof<F, E, ()>, stark::prover::ProvingError> {
    let rnd_air = create_keccak_rnd_air(opts);
    let rc_air = create_keccak_rc_air(opts).with_preprocessed(
        keccak_rc::preprocessed_commitment(opts),
        keccak_rc::NUM_PRECOMPUTED_COLS,
    );
    let bw_air = create_bitwise_air(opts).with_preprocessed(
        bitwise::preprocessed_commitment(opts),
        bitwise::NUM_PRECOMPUTED_COLS,
    );

    let [t0, t1, t2, t3] = traces;
    let pairs: Vec<(DynAir, &mut TraceTable<F, E>, &())> = vec![
        (adapter, t0, &()),
        (&rnd_air, t1, &()),
        (&rc_air, t2, &()),
        (&bw_air, t3, &()),
    ];
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
    adapter: &AdapterAir,
    proof: &stark::proof::stark::MultiProof<F, E, ()>,
) -> bool {
    let rnd_air = create_keccak_rnd_air(opts);
    let rc_air = create_keccak_rc_air(opts).with_preprocessed(
        keccak_rc::preprocessed_commitment(opts),
        keccak_rc::NUM_PRECOMPUTED_COLS,
    );
    let bw_air = create_bitwise_air(opts).with_preprocessed(
        bitwise::preprocessed_commitment(opts),
        bitwise::NUM_PRECOMPUTED_COLS,
    );
    let refs: Vec<DynAir> = vec![adapter, &rnd_air, &rc_air, &bw_air];
    let mut vt = transcript();
    Verifier::multi_verify_views(&refs, MultiProofView::Owned(proof), &mut vt, &FEE::zero())
}

/// Prove + verify, optionally corrupting the adapter trace in between.
///
/// `Err` means the prover refused. The adapter carries no constraints, so
/// tampering with its values is expected to reach the verifier and be caught
/// there (`Ok(false)`); the reject tests assert that stronger outcome rather
/// than accepting either failure, so a prover-side refusal would show up as a
/// change in behavior instead of hiding behind a passing test.
fn round_trip(mutate: impl FnOnce(&mut TraceTable<F, E>)) -> Result<bool, String> {
    let opts = options();
    let adapter = adapter_air(&opts);
    let ops = probe_ops();
    let mut traces = build_traces(&ops);
    mutate(&mut traces[0]);
    match prove_traces(&opts, &adapter, &mut traces) {
        Ok(proof) => Ok(verify_proof(&opts, &adapter, &proof)),
        Err(e) => Err(format!("{e:?}")),
    }
}

/// Assert the prover accepted the tampered trace and the verifier rejected it.
fn assert_proves_but_fails_verification(what: &str, mutate: impl FnOnce(&mut TraceTable<F, E>)) {
    match round_trip(mutate) {
        Ok(true) => panic!("{what} must break the Keccak bus balance, but the proof verified"),
        Ok(false) => {}
        Err(e) => panic!("{what} should reach the verifier, but the prover refused first: {e}"),
    }
}

#[test]
fn adapter_probe_proves_real_permutations() {
    let ops = probe_ops();
    let traces = build_traces(&ops);

    // The adapter's OUT columns must be the real permutation, byte for byte.
    for (row, op) in ops.iter().enumerate() {
        let expected = keccak_adapter::permute(op.input);
        for (lane, &value) in expected.iter().enumerate() {
            for b in 0..8 {
                assert_eq!(
                    traces[0].main_table.get_row(row)[cols::OUT + lane * 8 + b],
                    FE::from(u64::from((value >> (b * 8)) as u8)),
                    "OUT byte ({lane}, {b}) of row {row}"
                );
            }
        }
    }

    // Known-answer vector: keccak-f[1600] of the all-zero state. Same constant
    // the executor pins in `executor/src/tests/keccak_tests.rs`.
    let zero_out = keccak_adapter::permute([0u64; 25]);
    assert_eq!(
        zero_out[0], 0xF1258F7940E1DDE7,
        "keccak_f1600(0) lane 0 must match the published vector"
    );

    // The BITWISE feed is exactly the per-round half of the production
    // collector: 1028 lookups per round, no address-shaped lookups and no HWSL
    // (the θ/ρ shifts are inline μ-gated identities on the round chip).
    assert_eq!(
        keccak_adapter::bitwise_ops_for(&ops).len(),
        ops.len() * 24 * 1028,
        "per-permutation BITWISE lookup count"
    );

    // The round-trip through the AIRs is the real check: KECCAK_RND enforces
    // all 24 rounds, and the bus only balances if the adapter's OUT state is
    // what those rounds actually produce from its IN state.
    assert_eq!(
        round_trip(|_| {}),
        Ok(true),
        "honest keccak adapter proof must verify"
    );
}

#[test]
fn tampered_output_byte_rejects() {
    assert_proves_but_fails_verification("a flipped OUT byte", |t| {
        let old = t.main_table.get_row(1)[cols::out_byte(2, 3, 4)];
        t.main_table
            .set_fe(1, cols::out_byte(2, 3, 4), old + FE::one());
    });
}

#[test]
fn tampered_input_byte_rejects() {
    assert_proves_but_fails_verification("a flipped IN byte", |t| {
        let old = t.main_table.get_row(0)[cols::in_byte(4, 1, 7)];
        t.main_table
            .set_fe(0, cols::in_byte(4, 1, 7), old + FE::one());
    });
}

#[test]
fn padding_row_multiplicity_rejects() {
    // Row 3 is padding (3 real ops, height 4). Turning it real makes the
    // adapter send a (tag=0, round=0, all-zero state) request and receive a
    // (tag=0, round=24, all-zero state) reply that no KECCAK_RND row answers.
    assert_proves_but_fails_verification("an is-real padding row", |t| {
        t.main_table.set_fe(3, cols::MU, FE::one())
    });
}

/// DOCUMENTS A HAZARD — this test asserts that a forgery SUCCEEDS.
///
/// Nothing in the bus contract binds a request token to its reply token except
/// the tag. Given two permutations sharing a tag, a prover can hand back each
/// one's output as the other's: the reply multiset `{(tag, 24, A_out),
/// (tag, 24, B_out)}` is unchanged by the swap, so the `Keccak` bus still
/// balances and the proof verifies even though neither adapter row states a
/// true permutation.
///
/// This is why [`super::keccak_adapter`] requires unique tags, and why the
/// production LFM adapter must carry them as preprocessed program data with
/// registrar-vouched uniqueness rather than as prover-chosen witness. If this
/// test ever starts FAILING, something began binding request to reply and the
/// tag-uniqueness obligation should be re-derived before it is relaxed.
#[test]
fn duplicate_tag_output_swap_accepts_demonstrating_hazard() {
    let opts = options();
    let adapter = adapter_air(&opts);

    let mut ops = probe_ops();
    ops.truncate(2);
    ops[1].tag = ops[0].tag; // the whole point: duplicate tag

    let mut traces = build_traces(&ops);
    assert_ne!(
        keccak_adapter::permute(ops[0].input),
        keccak_adapter::permute(ops[1].input),
        "the two outputs must differ or the swap is a no-op"
    );

    // Swap the two rows' 200 OUT bytes.
    for col in cols::OUT..cols::MU {
        let a = traces[0].main_table.get_row(0)[col];
        let b = traces[0].main_table.get_row(1)[col];
        traces[0].main_table.set_fe(0, col, b);
        traces[0].main_table.set_fe(1, col, a);
    }

    let proof = prove_traces(&opts, &adapter, &mut traces).expect("locally consistent");
    assert!(
        verify_proof(&opts, &adapter, &proof),
        "documents the tag-uniqueness obligation: with duplicate tags the swapped \
         outputs still balance the bus, so the verifier cannot catch the forgery"
    );
}
