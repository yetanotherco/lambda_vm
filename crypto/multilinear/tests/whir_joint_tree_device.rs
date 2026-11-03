//! One round-0 tree a grouped run (`Round0Trees::One`) on the card: the joint
//! tree committed in one leaf launch over the run's codewords, opened whole,
//! retired and revived over its stacks, and re-encoded when the stack path is
//! off — every path the bytes of the same group proved on the host alone, and
//! the leaf permutations the commits paid counted.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test whir_joint_tree_device -- --test-threads=1
//! LAMBDA_VM_P1_LEAN=1 cargo test --release -p multilinear --features cuda,parallel --test whir_joint_tree_device -- --test-threads=1
//! ```
//!
//! Needs a GPU. The tests flip process-wide knobs, so they run one at a time.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::gpu;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{
    Claimed, ColumnsAt, StackedCommitment, StackedProof, prove, round0_trees, verify,
};
use multilinear::stacking::{StackedLayout, borrow};
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, LeafOrder, Round0Trees,
    RoundOpenings, WhirFolds,
};
use multilinear::whir_commit::TreeDrop;
use multilinear::whir_hash::{P1Whir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn sponge() -> <P1Whir as WhirHash>::Sponge {
    let mut t = P1Whir::sponge();
    t.append_bytes(b"whir-joint-tree-device");
    t
}

/// First6 then uniform4, one chain and one round-0 tree per run of up to
/// four, the coset order, no grind (its nonce is not reproducible across two
/// searches).
fn config(round0: Round0Trees) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 33,
        grind: GrindBits::default(),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            arity4_cap: CapPolicy::Auto,
            group: GroupFold::Indexed,
            leaves: LeafOrder::Coset,
            round0,
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

/// The roots, the proof, and the round-0 leaf permutations its commit paid.
type Proved = (Vec<[u8; 32]>, StackedProof<F, Ext>, u64);

struct Stack {
    name: &'static str,
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
}

impl Stack {
    fn new(name: &'static str, heights: &[usize], n_stack: usize) -> Self {
        let columns: Vec<Mle<F>> = heights
            .iter()
            .enumerate()
            .map(|(i, &h)| column(h, 41 + 5 * i as u64))
            .collect();
        let points: Vec<Vec<EE>> = heights
            .iter()
            .enumerate()
            .map(|(i, &h)| {
                (0..h)
                    .map(|j| EE::from(7 + 11 * i as u64 + 3 * j as u64))
                    .collect()
            })
            .collect();
        let values = columns
            .iter()
            .zip(&points)
            .map(|(c, p)| c.evaluate_in(p).unwrap())
            .collect();
        let layout = StackedLayout::build(heights, n_stack).unwrap();
        Self {
            name,
            layout,
            columns,
            points,
            values,
        }
    }

    /// One run of three polynomials over `2^18`: one joint tree, 192-felt leaves.
    fn run_of_three() -> Self {
        let stack = Self::new("a run of three", &[18, 17, 17, 16, 15], 18);
        assert_eq!(stack.layout.num_polys(), 3, "one run of three");
        stack
    }

    /// Six polynomials over `2^18`: a run of four (256-felt leaves) and a run
    /// of two (128-felt leaves).
    fn runs_of_four_and_two() -> Self {
        let stack = Self::new("runs [4, 2]", &[18; 6], 18);
        assert_eq!(stack.layout.num_polys(), 6, "six polynomials");
        stack
    }

    /// Five polynomials: a run of four and a polynomial alone (its own tree).
    fn runs_of_four_and_one() -> Self {
        let stack = Self::new("runs [4, 1]", &[18; 5], 18);
        assert_eq!(stack.layout.num_polys(), 5, "five polynomials");
        stack
    }

    fn trees(&self, cfg: &ChainConfig) -> u64 {
        round0_trees(self.layout.num_polys(), cfg).len() as u64
    }

    /// The round-0 leaf permutations its commit pays: `⌈felts / 12⌉` a leaf
    /// of every tree, `2^(n + 2 − 6)` leaves a tree.
    fn leaf_perms(&self, cfg: &ChainConfig) -> u64 {
        let leaves = 1u64 << (self.layout.n_stack() + cfg.log_blowup - 6);
        round0_trees(self.layout.num_polys(), cfg)
            .iter()
            .map(|tree| leaves * (64 * tree.len() as u64).div_ceil(12))
            .sum()
    }

    fn try_prove(
        &self,
        cfg: &ChainConfig,
        drop: Option<usize>,
    ) -> Result<Proved, multilinear::Error> {
        let refs: Vec<&Mle<F>> = self.columns.iter().collect();
        let store = gpu::upload_columns(&refs).expect("resident columns (a GPU)");
        let perms = gpu::commit_leaf_perms();
        let committed = StackedCommitment::<F, P1Whir>::commit(
            self.layout.clone(),
            &borrow(&self.columns),
            Some((&store, 0)),
            cfg,
        )?;
        let perms = gpu::commit_leaf_perms() - perms;
        let roots = committed.roots();
        let stacked = match drop {
            None => committed,
            Some(drop) => committed
                .retire(TreeDrop::uniform(drop), cfg)?
                .revive::<P1Whir, _>(
                    &borrow(&self.columns),
                    Some((&store, ColumnsAt::From(0))),
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
        Ok((roots, proof, perms))
    }

    fn verify(
        &self,
        roots: &[[u8; 32]],
        proof: &StackedProof<F, Ext>,
        cfg: &ChainConfig,
        label: &str,
    ) {
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

    /// The same group proved on the host alone.
    fn on_the_host(&self, cfg: &ChainConfig) -> Proved {
        gpu::force_host_whir_commit(true);
        gpu::force_shared_open_declined(true);
        let host = self.try_prove(cfg, None);
        gpu::force_shared_open_declined(false);
        gpu::force_host_whir_commit(false);
        host.unwrap_or_else(|e| panic!("{}: the host run: {e:?}", self.name))
    }
}

/// ★ A joint run committed on the card — one leaf launch over its codewords —
/// has the host's roots, pays exactly its leaves' permutations, and proves the
/// host's bytes opened whole, retired and revived over its stacks at two kept
/// depths (one stack opening a tree, nothing re-encoded), and re-encoded with
/// the stack path off.
#[test]
fn a_joint_run_on_the_card_proves_the_hosts_bytes() {
    let _serial = serial();
    let cfg = config(Round0Trees::One);
    for stack in [
        Stack::run_of_three(),
        Stack::runs_of_four_and_two(),
        Stack::runs_of_four_and_one(),
    ] {
        let (host_roots, host_proof, _) = stack.on_the_host(&cfg);
        stack.verify(&host_roots, &host_proof, &cfg, stack.name);
        assert_eq!(
            host_roots.len() as u64,
            stack.trees(&cfg),
            "{}: one root a tree",
            stack.name
        );
        assert!(
            host_proof
                .polys
                .iter()
                .any(|chain| matches!(chain.rounds[0].openings, RoundOpenings::BaseJoint(_))),
            "{}: a joint round 0",
            stack.name
        );

        let fallbacks = gpu::host_fallbacks();
        let (roots, whole, perms) = stack
            .try_prove(&cfg, None)
            .unwrap_or_else(|e| panic!("{}, whole: {e:?}", stack.name));
        assert_eq!(
            perms,
            stack.leaf_perms(&cfg),
            "{}: the commit's leaf permutations",
            stack.name
        );
        assert_eq!(
            gpu::host_fallbacks(),
            fallbacks,
            "{}: no host commit",
            stack.name
        );
        assert_eq!(roots, host_roots, "{}, whole: roots", stack.name);
        assert!(
            bytes(&whole) == bytes(&host_proof),
            "{}, whole: the host's bytes",
            stack.name
        );

        for drop in [2, 4] {
            let label = format!("{}, revived at drop {drop}", stack.name);
            let from_stack = gpu::coset_openings();
            let reencoded = gpu::coset_fallbacks();
            let (roots, proof, _) = stack
                .try_prove(&cfg, Some(drop))
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            assert_eq!(
                gpu::coset_openings() - from_stack,
                stack.trees(&cfg),
                "{label}: one stack opening a tree"
            );
            assert_eq!(
                gpu::coset_fallbacks(),
                reencoded,
                "{label}: nothing re-encoded"
            );
            assert_eq!(roots, host_roots, "{label}: roots");
            stack.verify(&roots, &proof, &cfg, &label);
            assert!(
                bytes(&proof) == bytes(&host_proof),
                "{label}: the host's bytes"
            );
        }

        let label = format!("{}, revived at drop 4, the stack path off", stack.name);
        gpu::force_coset_openings(Some(false));
        let from_stack = gpu::coset_openings();
        let reencoded = stack.try_prove(&cfg, Some(4));
        let moved = gpu::coset_openings() - from_stack;
        gpu::force_coset_openings(None);
        let (_, reencoded, _) = reencoded.unwrap_or_else(|e| panic!("{label}: {e:?}"));
        assert_eq!(moved, 0, "{label}: no tree from its stack");
        assert!(
            bytes(&reencoded) == bytes(&host_proof),
            "{label}: the host's bytes"
        );
    }
}

/// ★ At the production block's shape — a run of three stacked polynomials of
/// `2^27`, First6, blowup 2, the coset order: `2^22` leaves a tree — a joint
/// run committed on the card has the host's roots (the verifiers' commit,
/// `commit_on_host`) and pays exactly its leaves' permutations, with no host
/// fallback. The other tests' shapes are `2^14` leaves.
#[test]
fn a_production_shaped_joint_run_on_the_card_has_the_hosts_roots() {
    let _serial = serial();
    let cfg = ChainConfig {
        log_blowup: 1,
        ..config(Round0Trees::One)
    };
    let columns: Vec<Mle<F>> = (0..3).map(|i| column(27, 97 + 13 * i)).collect();
    let layout = StackedLayout::build(&[27; 3], 27).unwrap();
    assert_eq!(layout.num_polys(), 3, "one run of three");
    assert_eq!(round0_trees(3, &cfg), vec![0..3], "one joint tree");
    let refs: Vec<&Mle<F>> = columns.iter().collect();
    let store = gpu::upload_columns(&refs).expect("resident columns (a GPU)");
    let (fallbacks, perms) = (gpu::host_fallbacks(), gpu::commit_leaf_perms());
    let card = StackedCommitment::<F, P1Whir>::commit(
        layout.clone(),
        &borrow(&columns),
        Some((&store, 0)),
        &cfg,
    )
    .unwrap();
    let perms = gpu::commit_leaf_perms() - perms;
    assert_eq!(gpu::host_fallbacks(), fallbacks, "no host commit");
    // 192 felts a leaf: 16 permutations, over 2^(27 + 1 - 6) leaves.
    assert_eq!(perms, 16 << 22, "the joint leaves' permutations");
    drop(store);
    let host =
        StackedCommitment::<F, P1Whir>::commit_on_host(layout, &borrow(&columns), &cfg).unwrap();
    assert_eq!(card.roots().len(), 1, "one root");
    assert_eq!(host.roots(), card.roots(), "the card's root is the host's");
}

/// The same stack under one tree a polynomial pays 18 permutations a leaf
/// position of a run of three where one tree a run pays 16; its proof is the
/// per-tree format's, refused by the joint verifier and the reverse.
#[test]
fn one_tree_a_run_saves_two_permutations_a_leaf_of_a_run_of_three() {
    let _serial = serial();
    let stack = Stack::run_of_three();
    let one = config(Round0Trees::One);
    let each = config(Round0Trees::Each);
    let (each_roots, each_proof, each_perms) = stack.try_prove(&each, None).unwrap();
    let (one_roots, one_proof, one_perms) = stack.try_prove(&one, None).unwrap();
    let leaves = 1u64 << (18 + 2 - 6);
    assert_eq!((each_perms, one_perms), (18 * leaves, 16 * leaves));
    stack.verify(&each_roots, &each_proof, &each, "each");
    stack.verify(&one_roots, &one_proof, &one, "one");
    let domain = Domain::<F>::new(stack.layout.n_stack() + one.log_blowup).unwrap();
    for (roots, proof, cfg, label) in [
        (
            &one_roots,
            &one_proof,
            &each,
            "a joint proof under one tree a polynomial",
        ),
        (
            &each_roots,
            &each_proof,
            &one,
            "a per-tree proof under one tree a run",
        ),
    ] {
        assert!(
            verify::<F, Ext, _, P1Whir>(
                proof,
                &stack.layout,
                roots,
                &Claimed::PerColumn(&stack.points),
                &stack.values,
                &domain,
                cfg,
                &mut sponge(),
            )
            .is_err(),
            "{label}: refused"
        );
    }
}

/// A verifier's host commit (`StackedCommitment::commit_on_host`) has the
/// card's roots under either round-0 format, at a size the card commits, and
/// runs no card commit and no card leaf pass.
#[test]
fn a_host_commit_has_the_cards_roots_and_runs_nothing_on_the_card() {
    let _serial = serial();
    for round0 in [Round0Trees::Each, Round0Trees::One] {
        let cfg = config(round0);
        for stack in [Stack::run_of_three(), Stack::runs_of_four_and_one()] {
            let refs: Vec<&Mle<F>> = stack.columns.iter().collect();
            let store = gpu::upload_columns(&refs).expect("resident columns (a GPU)");
            let card = StackedCommitment::<F, P1Whir>::commit(
                stack.layout.clone(),
                &borrow(&stack.columns),
                Some((&store, 0)),
                &cfg,
            )
            .unwrap();
            let (calls, perms) = (gpu::commit_calls(), gpu::commit_leaf_perms());
            let host = StackedCommitment::<F, P1Whir>::commit_on_host(
                stack.layout.clone(),
                &borrow(&stack.columns),
                &cfg,
            )
            .unwrap();
            assert_eq!(
                (gpu::commit_calls(), gpu::commit_leaf_perms()),
                (calls, perms),
                "{} {round0:?}: nothing on the card",
                stack.name
            );
            assert_eq!(
                host.roots(),
                card.roots(),
                "{} {round0:?}: the card's roots",
                stack.name
            );
        }
    }
}
