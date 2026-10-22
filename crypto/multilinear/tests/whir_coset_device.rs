//! The coset leaf order on the card: a round-0 tree the card commits in the
//! coset order is the host's, and a stack retired and revived over its
//! resident columns proves the whole tree's bytes whether its openings are
//! evaluated coset by coset from the stack or read from a re-encoded codeword.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda --test whir_coset_device
//! ```
//!
//! Needs a GPU. The device paths are asserted TAKEN (the commit stayed on the
//! card; the stack-held openings were served from the stack and nothing fell
//! back), so a card that declined could not turn this into a host-against-host
//! comparison.
#![cfg(feature = "cuda")]

use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as F;
use multilinear::mle::Mle;
use multilinear::stacked_eval::{Claimed, StackedCommitment, StackedProof, prove, verify};
use multilinear::stacking::{StackedLayout, borrow};
use multilinear::whir::{Domain, encode, lift_coefficients};
use multilinear::whir_chain::{
    self, ChainConfig, ChainFormat, FirstFold, GrindBits, LeafOrder, UnencodedSuccessor, WhirFolds,
};
use multilinear::whir_commit::{CodewordCommitment, TreeDrop};
use multilinear::whir_hash::{P1Whir, WhirHash};

type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn sponge() -> <P1Whir as WhirHash>::Sponge {
    let mut t = P1Whir::sponge();
    t.append_bytes(b"whir-coset-device");
    t
}

fn config(leaves: LeafOrder, cap: CapPolicy) -> ChainConfig {
    ChainConfig {
        log_blowup: 2,
        log_folding: 4,
        num_queries: 33,
        grind: GrindBits::uniform(8),
        format: ChainFormat {
            folds: WhirFolds::First(FirstFold::new(6).unwrap()),
            arity4_cap: cap,
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

macro_rules! bytes {
    ($proof:expr) => {
        rkyv::to_bytes::<rkyv::rancor::Error>($proof)
            .unwrap()
            .to_vec()
    };
}

/// The tests flip process-wide knobs (the stack-held openings, the encoded
/// successor), so they run one at a time.
static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// ★ A coset-ordered chain committed on the card proves the bytes of the
/// same chain over a host tree in the coset order, and the host verifier
/// accepts it; its root is not the natural tree's.
#[test]
fn a_coset_chain_on_the_card_proves_the_host_chains_bytes() {
    let _serial = serial();
    for cap in [CapPolicy::Off, CapPolicy::Fixed(2), CapPolicy::Auto] {
        let num_vars = 16;
        let cfg = config(LeafOrder::Coset, cap);
        let f = column(num_vars, 3);
        let z: Vec<EE> = (0..num_vars).map(|i| EE::from(301 + i as u64)).collect();
        let y = f.evaluate_in(&z).unwrap();
        let (device, domain) = whir_chain::commit::<F, P1Whir>(&f, &cfg, true).unwrap();
        assert!(
            device.codeword().device().is_some(),
            "{cap}: the commit must have stayed on the card"
        );
        let host_domain = Domain::<F>::new(num_vars + cfg.log_blowup).unwrap();
        let host = CodewordCommitment::<F, P1Whir>::from_codeword_on_host_in(
            encode::<F, F>(&lift_coefficients(&f), &host_domain).unwrap(),
            cfg.schedule(num_vars)[0],
            LeafOrder::Coset,
        )
        .unwrap();
        assert_eq!(host.root(), device.root(), "{cap}: roots");
        let (natural, _) =
            whir_chain::commit::<F, P1Whir>(&f, &config(LeafOrder::Natural, cap), true).unwrap();
        assert_ne!(
            natural.root(),
            device.root(),
            "{cap}: the order moves the root"
        );
        let device_proof =
            whir_chain::prove::<F, Ext, _, P1Whir>(&f, &z, &device, &domain, &cfg, &mut sponge())
                .unwrap();
        let host_proof = whir_chain::prove::<F, Ext, _, P1Whir>(
            &f,
            &z,
            &host,
            &host_domain,
            &cfg,
            &mut sponge(),
        )
        .unwrap();
        assert_eq!(
            bytes!(&device_proof),
            bytes!(&host_proof),
            "{cap}: proof bytes"
        );
        whir_chain::verify::<F, Ext, _, P1Whir>(
            &device_proof,
            &device.root(),
            &z,
            y,
            &domain,
            &cfg,
            &mut sponge(),
        )
        .unwrap_or_else(|e| panic!("{cap}: the verifier refused the card's proof: {e:?}"));
    }
}

/// A stack's roots and its proof.
type Proved = (Vec<[u8; 32]>, StackedProof<F, Ext>);

/// A stack committed over resident columns and opened: the whole tree's
/// proof, then the same stack retired to `drop` 4-ary levels and revived.
struct Stack {
    layout: StackedLayout,
    columns: Vec<Mle<F>>,
    points: Vec<Vec<EE>>,
    values: Vec<EE>,
}

impl Stack {
    fn new(n_stack: usize) -> Self {
        let heights = [n_stack, n_stack - 1, n_stack - 1, n_stack - 2, n_stack - 3];
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
        Self {
            layout: StackedLayout::build(&heights, n_stack).unwrap(),
            columns,
            points,
            values,
        }
    }

    /// The proof, whole (`drop` None) or retired and revived, under `cfg`.
    fn prove(
        &self,
        cfg: &ChainConfig,
        drop: Option<usize>,
    ) -> (Vec<[u8; 32]>, StackedProof<F, Ext>) {
        self.try_prove(cfg, drop).unwrap()
    }

    /// [`Self::prove`], its refusal returned.
    fn try_prove(
        &self,
        cfg: &ChainConfig,
        drop: Option<usize>,
    ) -> Result<Proved, multilinear::Error> {
        let refs: Vec<&Mle<F>> = self.columns.iter().collect();
        let store = multilinear::gpu::upload_columns(&refs).expect("resident columns (a GPU)");
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

    fn verify(&self, roots: &[[u8; 32]], proof: &StackedProof<F, Ext>, cfg: &ChainConfig) {
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
        .unwrap_or_else(|e| panic!("the verifier refused a stack's proof: {e:?}"));
    }
}

/// ★ THE LEVER'S PARITY. A coset-ordered stack retired to a kept top of 16 or
/// 256 leaves and revived over its resident columns proves the whole tree's
/// bytes: with its openings evaluated coset by coset from the stack (served
/// from the stack, none fallen back) and with them read from a re-encoded
/// codeword. At stacks whose cosets fold 4 and 6 bits, uncapped and under the
/// production cap.
#[test]
fn a_revived_coset_stack_proves_the_whole_trees_bytes_from_the_stack_or_a_reencode() {
    let _serial = serial();
    for (n_stack, cap) in [
        (18usize, CapPolicy::Auto),
        (18, CapPolicy::Off),
        (20, CapPolicy::Auto),
    ] {
        let cfg = config(LeafOrder::Coset, cap);
        let stack = Stack::new(n_stack);
        let (roots, whole) = stack.prove(&cfg, None);
        stack.verify(&roots, &whole, &cfg);
        for drop in [2usize, 4] {
            let label = format!("2^{n_stack} {cap} drop {drop}");
            multilinear::gpu::force_coset_openings(Some(true));
            let served = multilinear::gpu::coset_openings();
            let fell_back = multilinear::gpu::coset_fallbacks();
            let (again, from_stack) = stack.prove(&cfg, Some(drop));
            assert_eq!(again, roots, "{label}: roots");
            assert!(
                multilinear::gpu::coset_openings() > served,
                "{label}: the openings must have been served from the stack"
            );
            assert_eq!(
                multilinear::gpu::coset_fallbacks(),
                fell_back,
                "{label}: nothing re-encoded"
            );
            assert_eq!(
                bytes!(&from_stack),
                bytes!(&whole),
                "{label}: from the stack"
            );
            multilinear::gpu::force_coset_openings(Some(false));
            let served = multilinear::gpu::coset_openings();
            let (_, reencoded) = stack.prove(&cfg, Some(drop));
            assert_eq!(
                multilinear::gpu::coset_openings(),
                served,
                "{label}: off is off"
            );
            assert_eq!(
                bytes!(&reencoded),
                bytes!(&whole),
                "{label}: from a re-encode"
            );
            multilinear::gpu::force_coset_openings(None);
        }
    }
}

/// ★ Under the coset order a successor that is not encoded from the bound
/// message refuses the prove, named by its cause, on each path — the knob off,
/// the factors on the host (the card declined the opening), the card declining
/// the encode — for a stack committed on the card and one revived over its
/// columns. Each path is shown taken by its own counter where it has one. The
/// natural order under the same forcing folds and proves its whole tree's bytes.
#[test]
fn an_unencoded_successor_refuses_the_coset_prove_by_cause() {
    let _serial = serial();
    let coset = config(LeafOrder::Coset, CapPolicy::Auto);
    let natural = config(LeafOrder::Natural, CapPolicy::Auto);
    let stack = Stack::new(18);
    let (_, natural_whole) = stack.prove(&natural, None);
    type Force = fn(bool);
    let paths: [(&str, Force, UnencodedSuccessor); 3] = [
        (
            "the knob off",
            |on| multilinear::gpu::force_encode_successor(on.then_some(false)),
            UnencodedSuccessor::KnobOff,
        ),
        (
            "the factors on the host",
            multilinear::gpu::force_shared_open_declined,
            UnencodedSuccessor::HostFactors,
        ),
        (
            "the card declining the encode",
            multilinear::gpu::force_encode_decline,
            UnencodedSuccessor::CardDeclined,
        ),
    ];
    for (name, force, cause) in paths {
        for drop in [None, Some(4)] {
            let label = format!("{name}, drop {drop:?}");
            let (declines, host) = (
                multilinear::gpu::encode_declines(),
                multilinear::gpu::open_host_fallbacks(),
            );
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
            assert_eq!(
                bytes!(&natural_proof),
                bytes!(&natural_whole),
                "{label}: natural bytes"
            );
            match cause {
                UnencodedSuccessor::CardDeclined => assert!(
                    multilinear::gpu::encode_declines() > declines,
                    "{label}: the decline counted"
                ),
                UnencodedSuccessor::HostFactors => assert!(
                    multilinear::gpu::open_host_fallbacks() > host,
                    "{label}: the host fallback counted"
                ),
                UnencodedSuccessor::KnobOff => {}
            }
        }
    }
    // Every forcing undone: the coset stack proves again, retired as whole.
    let (_, again) = stack.prove(&coset, Some(4));
    let (_, whole) = stack.prove(&coset, None);
    assert_eq!(bytes!(&again), bytes!(&whole));
}

/// The natural order on the card is untouched: a retired natural stack still
/// proves its whole tree's bytes, and nothing is served from a stack.
#[test]
fn a_revived_natural_stack_is_untouched() {
    let _serial = serial();
    let cfg = config(LeafOrder::Natural, CapPolicy::Auto);
    let stack = Stack::new(18);
    let (roots, whole) = stack.prove(&cfg, None);
    let served = multilinear::gpu::coset_openings();
    let (again, revived) = stack.prove(&cfg, Some(4));
    assert_eq!(again, roots);
    assert_eq!(multilinear::gpu::coset_openings(), served);
    assert_eq!(bytes!(&revived), bytes!(&whole));
    stack.verify(&roots, &whole, &cfg);
}
