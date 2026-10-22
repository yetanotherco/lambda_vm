//! The coset leaf order under one chain per run of a group's polynomials
//! (`LeafOrder::Coset` with `GroupFold::Indexed`) on the card: a grouped run
//! keeps its round-0 codewords on the card and its successor is ENCODED from
//! the run's combined bound message — the bytes of the same run proved on the
//! host alone, where the host folds — or the prove is refused, named by its
//! cause. No path folds a grouped run's card codewords under the coset order.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_coset_group_device -- --test-threads=1
//! ```
//!
//! Needs a GPU. Each path is asserted taken by its counter. The tests flip
//! process-wide knobs, so they run one at a time.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{Claimed, StackedCommitment, StackedProof, prove, verify};
use multilinear::stacking::{StackedLayout, borrow};
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, LeafOrder, RoundOpenings,
    UnencodedSuccessor, WhirFolds,
};
use multilinear::whir_commit::TreeDrop;
use multilinear::whir_hash::{P1Whir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn sponge() -> <P1Whir as WhirHash>::Sponge {
    let mut t = P1Whir::sponge();
    t.append_bytes(b"whir-coset-group-device");
    t
}

/// First6 then uniform4, one chain per run of up to four, no grind (its nonce
/// is not reproducible across two searches).
fn config(leaves: LeafOrder) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 33,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            arity4_cap: CapPolicy::Auto,
            group: GroupFold::Indexed,
            leaves,
            ..ChainFormat::DEFAULT
        },
    }
}

fn column(num_vars: usize, seed: u64) -> Mle<F> {
    Mle::new(
        (0..(1u64 << num_vars))
            .map(|i| {
                FE::from(
                    i.wrapping_add(seed)
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(17)
                        >> 11,
                )
            })
            .collect(),
    )
    .unwrap()
}

fn bytes(proof: &StackedProof<F, Ext>) -> Vec<u8> {
    rkyv::to_bytes::<rkyv::rancor::Error>(proof)
        .unwrap()
        .to_vec()
}

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A stack of three polynomials over `2^18` — one grouped chain — committed
/// over resident columns and opened whole, or retired and revived.
struct Stack {
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
}

type Proved = (Vec<[u8; 32]>, StackedProof<F, Ext>);

impl Stack {
    fn new() -> Self {
        let heights = [18usize, 17, 17, 16, 15];
        let columns: Vec<Mle<F>> = heights
            .iter()
            .enumerate()
            .map(|(i, &h)| column(h, 11 + 7 * i as u64))
            .collect();
        let points: Vec<Vec<EE>> = heights
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                (0..h)
                    .map(|j| EE::from(5 + 13 * i as u64 + 3 * j as u64))
                    .collect()
            })
            .collect();
        let values = columns
            .iter()
            .zip(&points)
            .map(|(c, p)| c.evaluate_in(p).unwrap())
            .collect();
        let layout = StackedLayout::build(&heights, 18).unwrap();
        assert_eq!(layout.num_polys(), 3, "one run of three");
        Self {
            layout,
            columns,
            points,
            values,
        }
    }

    fn try_prove(
        &self,
        cfg: &ChainConfig,
        drop: Option<usize>,
    ) -> Result<Proved, multilinear::Error> {
        let refs: Vec<&Mle<F>> = self.columns.iter().collect();
        let store = gpu::upload_columns(&refs).expect("resident columns (a GPU)");
        let committed = StackedCommitment::<F, P1Whir>::commit(
            self.layout.clone(),
            &borrow(&self.columns),
            Some((&store, 0)),
            cfg,
        )?;
        let roots = committed.roots();
        let stacked = match drop {
            None => committed,
            Some(drop) => committed
                .retire(TreeDrop::uniform(drop), cfg)?
                .revive::<P1Whir, _>(
                    &borrow(&self.columns),
                    Some((&store, multilinear::stacked_eval::ColumnsAt::From(0))),
                    cfg,
                )?,
        };
        let proof = prove::<F, Ext, _, P1Whir, _>(
            &stacked,
            &borrow(&self.columns),
            Some((&store, 0)),
            &Claimed::PerColumn(&self.points),
            &self.values,
            cfg,
            &mut sponge(),
        )?;
        Ok((roots, proof))
    }

    fn verify(
        &self,
        roots: &[[u8; 32]],
        proof: &StackedProof<F, Ext>,
        cfg: &ChainConfig,
        label: &str,
    ) {
        assert_eq!(proof.polys.len(), 1, "{label}: one chain");
        assert!(
            matches!(
                proof.polys[0].rounds[0].openings,
                RoundOpenings::BaseGrouped(_)
            ),
            "{label}: a grouped round 0"
        );
        let domain = Domain::<F>::new(self.layout.n_stack() + cfg.log_blowup).unwrap();
        verify::<F, Ext, _, P1Whir>(
            proof,
            &self.layout,
            roots,
            &Claimed::PerColumn(&self.points),
            &self.values,
            &domain,
            cfg,
            &mut sponge(),
        )
        .unwrap_or_else(|e| panic!("{label}: the verifier refused: {e:?}"));
    }
}

/// ★ A grouped run under the coset order on the card — committed and opened
/// whole, and retired and revived (its codewords re-encoded on the card) —
/// encodes its successor and proves the bytes of the same run proved on the
/// host alone (the host's codewords, factors and fold). Its roots are not the
/// natural order's.
#[test]
fn a_grouped_coset_run_on_the_card_proves_the_host_runs_bytes() {
    let _serial = serial();
    let coset = config(LeafOrder::Coset);
    let stack = Stack::new();
    gpu::force_host_whir_commit(true);
    gpu::force_shared_open_declined(true);
    let host = stack.try_prove(&coset, None);
    gpu::force_shared_open_declined(false);
    gpu::force_host_whir_commit(false);
    let (host_roots, host_proof) = host.unwrap_or_else(|e| panic!("the host run: {e:?}"));
    stack.verify(&host_roots, &host_proof, &coset, "the host run");
    for drop in [None, Some(2), Some(4)] {
        let label = format!("drop {drop:?}");
        let (encoded, chains) = (gpu::encoded_successors(), gpu::grouped_card_chains());
        let (roots, proof) = stack
            .try_prove(&coset, drop)
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        assert_eq!(
            gpu::encoded_successors() - encoded,
            1,
            "{label}: the successor encoded"
        );
        assert_eq!(
            gpu::grouped_card_chains() - chains,
            1,
            "{label}: round 0 on the card"
        );
        assert_eq!(roots, host_roots, "{label}: roots");
        stack.verify(&roots, &proof, &coset, &label);
        assert!(
            bytes(&proof) == bytes(&host_proof),
            "{label}: the host run's bytes"
        );
    }
    let (natural_roots, _) = stack.try_prove(&config(LeafOrder::Natural), None).unwrap();
    assert_ne!(natural_roots, host_roots, "the coset order moves the roots");
}

/// ★ Under the coset order a grouped run's successor that is not encoded
/// refuses the prove, named by its cause, on each path — the knob off, the
/// run's factors built on the host (the card declined an opening), the card
/// declining the encode — committed and opened whole and revived. Each path is
/// shown taken by its counter where it has one. The natural order under the
/// same forcing folds and proves its unforced bytes.
#[test]
fn an_unencoded_grouped_successor_refuses_the_coset_prove_by_cause() {
    let _serial = serial();
    let coset = config(LeafOrder::Coset);
    let natural = config(LeafOrder::Natural);
    let stack = Stack::new();
    let (_, natural_whole) = stack.try_prove(&natural, None).unwrap();
    type Force = fn(bool);
    let paths: [(&str, Force, UnencodedSuccessor); 3] = [
        (
            "the knob off",
            |on| gpu::force_encode_successor(on.then_some(false)),
            UnencodedSuccessor::KnobOff,
        ),
        (
            "the run's factors on the host",
            gpu::force_shared_open_declined,
            UnencodedSuccessor::HostFactors,
        ),
        (
            "the card declining the encode",
            gpu::force_encode_decline,
            UnencodedSuccessor::CardDeclined,
        ),
    ];
    for (name, force, cause) in paths {
        for drop in [None, Some(4)] {
            let label = format!("{name}, drop {drop:?}");
            let (declines, host) = (gpu::encode_declines(), gpu::grouped_host_chains());
            force(true);
            let refused = stack.try_prove(&coset, drop);
            let natural_proof = stack.try_prove(&natural, drop);
            force(false);
            assert_eq!(
                refused.err(),
                Some(multilinear::Error::CosetSuccessorNotEncoded { cause }),
                "{label}"
            );
            let (_, natural_proof) =
                natural_proof.unwrap_or_else(|e| panic!("{label}: the natural order folds: {e:?}"));
            assert!(
                bytes(&natural_proof) == bytes(&natural_whole),
                "{label}: the natural order's bytes"
            );
            match cause {
                UnencodedSuccessor::CardDeclined => assert!(
                    gpu::encode_declines() > declines,
                    "{label}: the decline counted"
                ),
                UnencodedSuccessor::HostFactors => assert!(
                    gpu::grouped_host_chains() > host,
                    "{label}: the grouped host fallback counted"
                ),
                UnencodedSuccessor::KnobOff => {}
            }
        }
    }
    // Every forcing undone: the coset run proves again.
    let (roots, again) = stack.try_prove(&coset, Some(4)).unwrap();
    stack.verify(&roots, &again, &coset, "after the forcings");
}
