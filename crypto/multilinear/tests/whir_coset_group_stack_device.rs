//! The coset leaf order under one chain per run (`LeafOrder::Coset` with
//! `GroupFold::Indexed`) on the card, a group retired and revived: every tree
//! of a run is held as its stack and served from it (each opening evaluates
//! the cosets its queries fall in), no round-0 codeword is re-encoded, and the
//! run's successor is encoded from its combined bound message — the bytes of
//! the same group proved on the host alone. A successor not encoded still
//! refuses the prove, by cause.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_coset_group_stack_device -- --test-threads=1
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
use multilinear::stacked_eval::{
    Claimed, ColumnsAt, StackedCommitment, StackedProof, index_chains, prove, verify,
};
use multilinear::stacking::{StackedLayout, borrow};
use multilinear::whir::Domain;
use multilinear::whir_chain::{
    ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, LeafOrder, UnencodedSuccessor,
    WhirFolds,
};
use multilinear::whir_commit::TreeDrop;
use multilinear::whir_hash::{P1Whir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn sponge() -> <P1Whir as WhirHash>::Sponge {
    let mut t = P1Whir::sponge();
    t.append_bytes(b"whir-coset-group-stack-device");
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

type Proved = (Vec<[u8; 32]>, StackedProof<F, Ext>);

/// A group's columns, stacked at `n_stack`, committed over resident columns
/// and opened whole or retired and revived.
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
            .map(|(i, &h)| column(h, 23 + 5 * i as u64))
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

    /// One run of three polynomials over `2^18`.
    fn run_of_three() -> Self {
        let stack = Self::new("a run of three", &[18, 17, 17, 16, 15], 18);
        assert_eq!(stack.layout.num_polys(), 3, "one run of three");
        stack
    }

    /// Five polynomials over `2^18`: a run of four and a polynomial alone.
    fn runs_of_four_and_one() -> Self {
        let stack = Self::new("runs [4, 1]", &[18; 5], 18);
        assert_eq!(stack.layout.num_polys(), 5, "five polynomials");
        stack
    }

    fn trees(&self) -> u64 {
        self.layout.num_polys() as u64
    }

    fn chains(&self) -> u64 {
        index_chains(self.layout.num_polys(), GroupFold::Indexed).len() as u64
    }

    fn grouped_runs(&self) -> u64 {
        index_chains(self.layout.num_polys(), GroupFold::Indexed)
            .iter()
            .filter(|run| run.len() > 1)
            .count() as u64
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
        Ok((roots, proof))
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

    /// The same group proved on the host alone: host codewords, host factors,
    /// host folds.
    fn on_the_host(&self, cfg: &ChainConfig) -> Proved {
        gpu::force_host_whir_commit(true);
        gpu::force_shared_open_declined(true);
        let host = self.try_prove(cfg, None);
        gpu::force_shared_open_declined(false);
        gpu::force_host_whir_commit(false);
        host.unwrap_or_else(|e| panic!("{}: the host run: {e:?}", self.name))
    }
}

/// The counters a revived group's openings move.
#[derive(Clone, Copy)]
struct Counts {
    from_stack: u64,
    reencoded: u64,
    encoded: u64,
    grouped: u64,
}

impl Counts {
    fn now() -> Self {
        Self {
            from_stack: gpu::coset_openings(),
            reencoded: gpu::coset_fallbacks(),
            encoded: gpu::encoded_successors(),
            grouped: gpu::grouped_card_chains(),
        }
    }

    fn since(self) -> Self {
        let now = Self::now();
        Self {
            from_stack: now.from_stack - self.from_stack,
            reencoded: now.reencoded - self.reencoded,
            encoded: now.encoded - self.encoded,
            grouped: now.grouped - self.grouped,
        }
    }
}

/// ★ A group retired and revived under the coset order and one chain per run
/// serves EVERY tree from its stack — the openings counter rises by the
/// group's tree count and nothing is re-encoded — and encodes each chain's
/// successor, the run's from its combined bound message: the bytes of the same
/// group proved on the host alone, at two kept depths. Opened whole, its trees
/// are held codewords and the counter does not move. With the stack path
/// switched off the revived group re-encodes (the counter does not move) and
/// proves the same bytes.
#[test]
fn a_revived_grouped_coset_run_serves_every_tree_from_its_stack() {
    let _serial = serial();
    let coset = config(LeafOrder::Coset);
    for stack in [Stack::run_of_three(), Stack::runs_of_four_and_one()] {
        let (host_roots, host_proof) = stack.on_the_host(&coset);
        stack.verify(&host_roots, &host_proof, &coset, stack.name);

        let before = Counts::now();
        let (roots, whole) = stack
            .try_prove(&coset, None)
            .unwrap_or_else(|e| panic!("{}, whole: {e:?}", stack.name));
        let moved = before.since();
        assert_eq!(roots, host_roots, "{}, whole: roots", stack.name);
        assert!(
            bytes(&whole) == bytes(&host_proof),
            "{}, whole: the host's bytes",
            stack.name
        );
        assert_eq!(moved.from_stack, 0, "{}, whole: held codewords", stack.name);
        assert_eq!(
            moved.encoded,
            stack.chains(),
            "{}, whole: every successor encoded",
            stack.name
        );

        for drop in [2, 4] {
            let label = format!("{}, revived at drop {drop}", stack.name);
            let before = Counts::now();
            let (roots, proof) = stack
                .try_prove(&coset, Some(drop))
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            let moved = before.since();
            assert_eq!(
                moved.from_stack,
                stack.trees(),
                "{label}: every tree from its stack"
            );
            assert_eq!(moved.reencoded, 0, "{label}: nothing re-encoded");
            assert_eq!(
                moved.encoded,
                stack.chains(),
                "{label}: every successor encoded"
            );
            assert_eq!(
                moved.grouped,
                stack.grouped_runs(),
                "{label}: the runs on the card"
            );
            assert_eq!(roots, host_roots, "{label}: roots");
            stack.verify(&roots, &proof, &coset, &label);
            assert!(
                bytes(&proof) == bytes(&host_proof),
                "{label}: the host's bytes"
            );
        }

        let label = format!("{}, revived at drop 4, the stack path off", stack.name);
        gpu::force_coset_openings(Some(false));
        let before = Counts::now();
        let reencoded = stack.try_prove(&coset, Some(4));
        let moved = before.since();
        gpu::force_coset_openings(None);
        let (_, reencoded) = reencoded.unwrap_or_else(|e| panic!("{label}: {e:?}"));
        assert_eq!(moved.from_stack, 0, "{label}: no tree from its stack");
        assert_eq!(
            moved.encoded,
            stack.chains(),
            "{label}: every successor encoded"
        );
        assert!(
            bytes(&reencoded) == bytes(&host_proof),
            "{label}: the host's bytes"
        );
    }
}

/// ★ Under the coset order a revived run's successor that is not encoded
/// refuses the prove, named by its cause, on each path — the knob off, the
/// run's factors built on the host (counted as a grouped host fallback: a
/// tree held as its stack is a card codeword), the card declining the encode
/// (counted). The natural order under the same forcing re-encodes, folds and
/// proves its unforced bytes.
#[test]
fn a_revived_grouped_coset_run_refuses_an_unencoded_successor_by_cause() {
    let _serial = serial();
    let coset = config(LeafOrder::Coset);
    let natural = config(LeafOrder::Natural);
    let stack = Stack::run_of_three();
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
        let label = format!("{name}, revived at drop 4");
        let (declines, host) = (gpu::encode_declines(), gpu::grouped_host_chains());
        let before = Counts::now();
        force(true);
        let refused = stack.try_prove(&coset, Some(4));
        let natural_proof = stack.try_prove(&natural, Some(4));
        force(false);
        let moved = before.since();
        assert_eq!(
            refused.err(),
            Some(multilinear::Error::CosetSuccessorNotEncoded { cause }),
            "{label}"
        );
        assert_eq!(moved.reencoded, 0, "{label}: nothing re-encoded");
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
                gpu::grouped_host_chains() >= host + 2,
                "{label}: the grouped host fallback counted for the coset run and the natural one"
            ),
            UnencodedSuccessor::KnobOff => {}
        }
    }
    // Every forcing undone: the revived coset run proves again, from its stack.
    let before = Counts::now();
    let (roots, again) = stack.try_prove(&coset, Some(4)).unwrap();
    assert_eq!(
        before.since().from_stack,
        stack.trees(),
        "after the forcings: from the stack"
    );
    stack.verify(&roots, &again, &coset, "after the forcings");
}

/// A revived group under the natural order and one chain per run is untouched:
/// no tree is served from a stack, and it proves the bytes it proves whole.
#[test]
fn a_revived_natural_grouped_run_is_untouched() {
    let _serial = serial();
    let natural = config(LeafOrder::Natural);
    for stack in [Stack::run_of_three(), Stack::runs_of_four_and_one()] {
        let (_, whole) = stack.try_prove(&natural, None).unwrap();
        for drop in [2, 4] {
            let label = format!("{}, natural, revived at drop {drop}", stack.name);
            let before = Counts::now();
            let (roots, proof) = stack
                .try_prove(&natural, Some(drop))
                .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            assert_eq!(
                before.since().from_stack,
                0,
                "{label}: no tree from a stack"
            );
            stack.verify(&roots, &proof, &natural, &label);
            assert!(
                bytes(&proof) == bytes(&whole),
                "{label}: the whole group's bytes"
            );
        }
    }
}
