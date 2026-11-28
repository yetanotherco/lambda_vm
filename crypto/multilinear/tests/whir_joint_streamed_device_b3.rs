//! A grouped run's joint BLAKE3 tree built one codeword at a time
//! (`multilinear::gpu::streamed_joint_leaves`): each codeword encoded, absorbed
//! into every leaf's chain and let go before the next. The roots and the proof
//! are the batched commit's and the host's, for every tree shape, and the
//! commit's ledger holds one codeword of the run instead of all of them.
//!
//! ```text
//! cargo test --release -p multilinear --features cuda,parallel --test whir_joint_streamed_device_b3 -- --test-threads=1
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
    ChainConfig, ChainFormat, FirstFold, GrindBits, GroupFold, LeafOrder, Round0Trees, WhirFolds,
};
use multilinear::whir_commit::TreeDrop;
use multilinear::whir_hash::{Blake3Whir, WhirHash};

type H = Blake3Whir;
type FE = FieldElement<F>;
type EE = FieldElement<Ext>;

fn sponge() -> <H as WhirHash>::Sponge {
    let mut t = H::sponge();
    t.append_bytes(b"whir-joint-tree-device-b3");
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

/// The roots and the proof.
type Proved = (Vec<[u8; 32]>, StackedProof<F, Ext>);

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

    /// One run of three polynomials over `2^18`: one joint tree, 1,536-byte
    /// leaves.
    fn run_of_three() -> Self {
        let stack = Self::new("a run of three", &[18, 17, 17, 16, 15], 18);
        assert_eq!(stack.layout.num_polys(), 3, "one run of three");
        stack
    }

    /// Six polynomials over `2^18`: a run of four (2,048-byte leaves) and a
    /// run of two (1,024 bytes).
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

    fn try_prove(
        &self,
        cfg: &ChainConfig,
        drop: Option<usize>,
    ) -> Result<Proved, multilinear::Error> {
        let refs: Vec<&Mle<F>> = self.columns.iter().collect();
        let store = gpu::upload_columns(&refs).expect("resident columns (a GPU)");
        // Opened whole: committed as any commitment. Retired first: committed
        // to be retired (`commit_to_retire`), which may stream a joint run.
        let (roots, stacked) = match drop {
            None => {
                let committed = StackedCommitment::<F, H>::commit(
                    self.layout.clone(),
                    &borrow(&self.columns),
                    Some((&store, 0)),
                    cfg,
                )?;
                (committed.roots(), committed)
            }
            Some(drop) => {
                let committed = StackedCommitment::<F, H>::commit_to_retire(
                    self.layout.clone(),
                    &borrow(&self.columns),
                    Some((&store, ColumnsAt::From(0))),
                    cfg,
                    &|| {},
                )?;
                let roots = committed.roots();
                let revived = committed
                    .retire(TreeDrop::uniform(drop), cfg)?
                    .revive::<H, _>(
                        &borrow(&self.columns),
                        Some((&store, ColumnsAt::From(0))),
                        cfg,
                    )?;
                (roots, revived)
            }
        };
        let proof = prove::<F, Ext, _, H, _>(
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
        verify::<F, Ext, _, H>(
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

/// Joint runs (two codewords or more) under `cfg` for `stack`.
fn joint_runs(stack: &Stack, cfg: &ChainConfig) -> u64 {
    round0_trees(stack.layout.num_polys(), cfg)
        .iter()
        .filter(|run| run.len() >= 2)
        .count() as u64
}

/// The joint runs streamed: every one, when a codeword's fold block is at
/// least 8 values (a whole number of 64-byte blocks of the leaf's chain);
/// none below, where the batched commit runs.
fn streamed_runs(stack: &Stack, cfg: &ChainConfig) -> u64 {
    let first = cfg
        .schedule(stack.layout.n_stack())
        .first()
        .copied()
        .unwrap_or(0);
    if first >= 3 {
        joint_runs(stack, cfg)
    } else {
        0
    }
}

/// The configurations the trees are built under: First6 at blowups 2 and 4,
/// uniform folds of 4 (a 16-value block), each under one tree a run and one
/// tree a polynomial; and under one tree a run, first folds of 3 (8 values,
/// the least that streams: one 64-byte block a codeword) and 2 (4 values:
/// never streamed, the batched commit runs), and the natural leaf order.
fn configs() -> Vec<(String, ChainConfig)> {
    let mut out = Vec::new();
    for first in [3u8, 2] {
        let mut cfg = config(Round0Trees::One);
        cfg.format.folds = WhirFolds::First(FirstFold::new(first as usize).unwrap());
        out.push((format!("One, first{first}, blowup 2^2"), cfg));
    }
    let mut natural = config(Round0Trees::One);
    natural.format.leaves = LeafOrder::Natural;
    out.push((
        "One, first6, natural leaf order, blowup 2^2".to_string(),
        natural,
    ));
    for round0 in [Round0Trees::One, Round0Trees::Each] {
        for log_blowup in [1, 2] {
            out.push((
                format!("{round0:?}, first6, blowup 2^{log_blowup}"),
                ChainConfig {
                    log_blowup,
                    ..config(round0)
                },
            ));
        }
        let mut uniform = config(round0);
        uniform.format.folds = WhirFolds::Uniform;
        out.push((format!("{round0:?}, uniform4, blowup 2^2"), uniform));
    }
    out
}

/// Streamed into its leaves one codeword at a time, every tree shape — a run
/// of three, runs of four and two, a run of four and a polynomial alone, under
/// one tree a run and one a polynomial, at blowups 2 and 4 and at fold blocks
/// of 64 and 16 values — has the host's roots, proves the host's bytes after
/// the retire and the revive, and verifies; each joint run is streamed once
/// and nothing falls back to a host commit. Under one tree a polynomial no run
/// is joint and nothing is streamed.
#[test]
fn a_streamed_joint_tree_has_the_hosts_roots_and_bytes_in_every_shape() {
    let _serial = serial();
    for (name, cfg) in configs() {
        for stack in [
            Stack::run_of_three(),
            Stack::runs_of_four_and_two(),
            Stack::runs_of_four_and_one(),
        ] {
            let label = format!("{}, {name}", stack.name);
            let (host_roots, host_proof) = stack.on_the_host(&cfg);
            let fallbacks = gpu::host_fallbacks();
            let streamed = gpu::joint_streamed_builds();
            let (roots, proof) = {
                let _on = gpu::force_streamed_joint_leaves(true);
                stack.try_prove(&cfg, Some(4))
            }
            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
            assert_eq!(gpu::host_fallbacks(), fallbacks, "{label}: no host commit");
            assert_eq!(
                gpu::joint_streamed_builds() - streamed,
                streamed_runs(&stack, &cfg),
                "{label}: each joint run streamed once (none under a fold block of 4)"
            );
            assert_eq!(roots, host_roots, "{label}: roots");
            stack.verify(&roots, &proof, &cfg, &label);
            assert!(
                bytes(&proof) == bytes(&host_proof),
                "{label}: the host's bytes"
            );
        }
    }
}

/// With the knob on, a commit that may be opened whole (not
/// `commit_to_retire`) is not streamed, and proves the host's bytes opened
/// whole.
#[test]
fn a_commit_opened_whole_is_never_streamed() {
    let _serial = serial();
    let cfg = config(Round0Trees::One);
    let stack = Stack::runs_of_four_and_two();
    let (host_roots, host_proof) = stack.on_the_host(&cfg);
    let streamed = gpu::joint_streamed_builds();
    let _on = gpu::force_streamed_joint_leaves(true);
    let refs: Vec<&Mle<F>> = stack.columns.iter().collect();
    let store = gpu::upload_columns(&refs).expect("resident columns (a GPU)");
    let committed = StackedCommitment::<F, H>::commit(
        stack.layout.clone(),
        &borrow(&stack.columns),
        Some((&store, 0)),
        &cfg,
    )
    .unwrap();
    assert_eq!(gpu::joint_streamed_builds(), streamed, "nothing streamed");
    assert_eq!(committed.roots(), host_roots, "roots");
    let proof = prove::<F, Ext, _, H, _>(
        &committed,
        &borrow(&stack.columns),
        Some((&store, 0)),
        &Claimed::PerColumn(&stack.points),
        &stack.values,
        &cfg,
        &mut sponge(),
    )
    .unwrap();
    assert!(bytes(&proof) == bytes(&host_proof), "the host's bytes");
}

/// The commit's ledger: a run of four streamed holds one codeword of the run
/// at a time, so its promise peaks at least three codewords under the batched
/// commit's (which holds all four), and the roots are the same.
#[test]
fn a_streamed_run_of_four_promises_three_codewords_less() {
    let _serial = serial();
    let cfg = config(Round0Trees::One);
    let stack = Stack::new("a run of four", &[18; 4], 18);
    assert_eq!(round0_trees(4, &cfg), vec![0..4], "one run of four");
    let codeword = (1u64 << (18 + cfg.log_blowup)) * 8;
    let refs: Vec<&Mle<F>> = stack.columns.iter().collect();
    let store = gpu::upload_columns(&refs).expect("resident columns (a GPU)");
    let peak = |on: bool| {
        let _knob = gpu::force_streamed_joint_leaves(on);
        gpu::reset_reserved_window();
        let base = gpu::ledger_reserved();
        let committed = StackedCommitment::<F, H>::commit_to_retire(
            stack.layout.clone(),
            &borrow(&stack.columns),
            Some((&store, ColumnsAt::From(0))),
            &cfg,
            &|| {},
        )
        .unwrap();
        let grew = gpu::reserved_window_peak().saturating_sub(base);
        (grew, committed.roots())
    };
    let streamed = gpu::joint_streamed_builds();
    let (batched, batched_roots) = peak(false);
    assert_eq!(gpu::joint_streamed_builds(), streamed, "off: not streamed");
    let (one_at_a_time, roots) = peak(true);
    assert_eq!(gpu::joint_streamed_builds(), streamed + 1, "on: streamed");
    eprintln!(
        "STREAMED LEDGER: a run of four at 2^18, blowup 2^{}: the commit grew the ledger {:.1} MiB batched, \
         {:.1} MiB streamed (a codeword {:.1} MiB)",
        cfg.log_blowup,
        batched as f64 / (1u64 << 20) as f64,
        one_at_a_time as f64 / (1u64 << 20) as f64,
        codeword as f64 / (1u64 << 20) as f64,
    );
    assert!(
        batched > 0,
        "the batched commit promised nothing: no ledger budget"
    );
    assert!(
        one_at_a_time + 3 * codeword <= batched,
        "the streamed commit grew the ledger {one_at_a_time} B, the batched {batched} B"
    );
    assert_eq!(roots, batched_roots, "roots");
}

/// What a group's streamed commit did not hold
/// (`gpu::StreamedNotHeld::add_back`, the block's
/// `commit_streamed_not_held`): for 2-, 4- and 7-polynomial groups (one run of
/// two, one of four, a run of four and one of three) every joint run streams
/// and the add-back is `(polys − 1)` codewords, and the streamed commit's
/// ledger growth plus it is the batched commit's growth to within half a
/// codeword: the planner reads the group's size, not how it was committed.
#[test]
fn a_streamed_groups_add_back_is_what_the_batched_commit_held() {
    let _serial = serial();
    let cfg = config(Round0Trees::One);
    let codeword = (1u64 << (18 + cfg.log_blowup)) * 8;
    for polys in [2usize, 4, 7] {
        let stack = Stack::new("a group", &vec![18; polys], 18);
        assert_eq!(stack.layout.num_polys(), polys, "{polys} polynomials");
        let trees = round0_trees(polys, &cfg).len();
        let refs: Vec<&Mle<F>> = stack.columns.iter().collect();
        let store = gpu::upload_columns(&refs).expect("resident columns (a GPU)");
        let commit = |on: bool| {
            let _knob = gpu::force_streamed_joint_leaves(on);
            let (refused, streamed, fallbacks) = (
                gpu::room_commit_refusals(),
                gpu::joint_streamed_builds(),
                gpu::host_fallbacks(),
            );
            let not_held = gpu::streamed_not_held();
            gpu::reset_reserved_window();
            let base = gpu::ledger_reserved();
            let committed = StackedCommitment::<F, H>::commit_to_retire(
                stack.layout.clone(),
                &borrow(&stack.columns),
                Some((&store, ColumnsAt::From(0))),
                &cfg,
                &|| {},
            )
            .unwrap();
            let grew = gpu::reserved_window_peak().saturating_sub(base);
            assert_eq!(
                gpu::room_commit_refusals(),
                refused,
                "{polys}: room promised"
            );
            assert_eq!(gpu::host_fallbacks(), fallbacks, "{polys}: no host commit");
            let runs = gpu::joint_streamed_builds() - streamed;
            let counted = gpu::streamed_not_held().since(not_held);
            assert_eq!(counted.runs, runs, "{polys}: every streamed build counted");
            let add_back = counted.add_back(trees);
            (grew, runs, add_back, committed.roots())
        };
        let (batched, runs_off, add_back_off, batched_roots) = commit(false);
        assert_eq!(
            (runs_off, add_back_off),
            (0, 0),
            "{polys}: off, nothing streamed"
        );
        let (streamed, runs, add_back, roots) = commit(true);
        eprintln!(
            "STREAMED ADD-BACK: {polys} polynomials at 2^18, blowup 2^{}: the commit grew the ledger \
             {:.1} MiB batched, {:.1} MiB streamed + {:.1} MiB added back (a codeword {:.1} MiB)",
            cfg.log_blowup,
            batched as f64 / (1u64 << 20) as f64,
            streamed as f64 / (1u64 << 20) as f64,
            add_back as f64 / (1u64 << 20) as f64,
            codeword as f64 / (1u64 << 20) as f64,
        );
        assert_eq!(runs as usize, trees, "{polys}: every joint run streamed");
        assert_eq!(
            add_back,
            (polys as u64 - 1) * codeword,
            "{polys}: every codeword but one added back"
        );
        assert!(batched > 0, "{polys}: no ledger budget");
        assert!(
            (streamed + add_back).abs_diff(batched) <= codeword / 2,
            "{polys}: streamed {streamed} B + added back {add_back} B against batched {batched} B"
        );
        assert_eq!(roots, batched_roots, "{polys}: roots");
    }
}

/// Restores the ledger's budget when dropped.
struct Budget(u64);

impl Drop for Budget {
    fn drop(&mut self) {
        gpu::set_reserve_budget(self.0);
    }
}

/// A streamed run keeps its codeword's promise between two steps: with the
/// budget at the streamed commit's own peak plus half a codeword, another
/// caller asking for a codeword between two steps
/// (`gpu::arm_streamed_gap_probe`) is refused every time, and the run streams
/// on the card with the host's roots, no host commit. Were the promise let go
/// with the codeword's values, the other caller would take that room, the next
/// encode would be refused and the run would fall off the card.
#[test]
fn a_streamed_run_keeps_its_codewords_room_between_steps() {
    let _serial = serial();
    let cfg = config(Round0Trees::One);
    let stack = Stack::new("a run of four", &[18; 4], 18);
    let codeword = (1u64 << (18 + cfg.log_blowup)) * 8;
    let (host_roots, _) = stack.on_the_host(&cfg);
    let refs: Vec<&Mle<F>> = stack.columns.iter().collect();
    let store = gpu::upload_columns(&refs).expect("resident columns (a GPU)");
    let _on = gpu::force_streamed_joint_leaves(true);
    let commit = || {
        StackedCommitment::<F, H>::commit_to_retire(
            stack.layout.clone(),
            &borrow(&stack.columns),
            Some((&store, ColumnsAt::From(0))),
            &cfg,
            &|| {},
        )
        .unwrap()
        .roots()
    };
    // The streamed commit's own peak, on a roomy ledger.
    gpu::reset_reserved_window();
    let base = gpu::ledger_reserved();
    let streamed = gpu::joint_streamed_builds();
    assert_eq!(commit(), host_roots, "roots");
    assert_eq!(gpu::joint_streamed_builds(), streamed + 1, "streamed");
    let peak = gpu::reserved_window_peak().saturating_sub(base);
    assert!(
        peak >= codeword,
        "the commit promised {peak} B: no ledger budget"
    );
    // Now the budget leaves half a codeword beside that peak.
    let held = gpu::ledger_reserved();
    let _budget = Budget(gpu::set_reserve_budget(held + peak + codeword / 2));
    let probe = gpu::arm_streamed_gap_probe(codeword);
    let (streamed, fallbacks) = (gpu::joint_streamed_builds(), gpu::host_fallbacks());
    let roots = commit();
    let (taken, refused) = probe.outcomes();
    eprintln!(
        "STREAMED GAP: a run of four at 2^18 (a codeword {:.1} MiB, the commit's peak {:.1} MiB, \
         budget {:.1} MiB free beside it): the other caller was granted {taken} and refused {refused} \
         time(s) between steps",
        codeword as f64 / (1u64 << 20) as f64,
        peak as f64 / (1u64 << 20) as f64,
        (codeword / 2) as f64 / (1u64 << 20) as f64,
    );
    assert_eq!(taken, 0, "another caller took the room between two steps");
    assert_eq!(refused, 3, "asked at each of the three gaps");
    assert_eq!(
        gpu::joint_streamed_builds(),
        streamed + 1,
        "the run streamed on the card"
    );
    assert_eq!(gpu::host_fallbacks(), fallbacks, "no host commit");
    assert_eq!(roots, host_roots, "roots");
}

/// A card that declines after the first codeword (the encode of codeword 1,
/// 2 or 3 of a run of four refused): the partial tree is let go, nothing is
/// streamed, the batched commit runs on the card (no host commit), and the
/// roots and the proof are the host's.
#[test]
fn a_streamed_run_declined_after_its_first_codeword_commits_batched_with_the_hosts_bytes() {
    let _serial = serial();
    let cfg = config(Round0Trees::One);
    let stack = Stack::new("a run of four", &[18; 4], 18);
    let (host_roots, host_proof) = stack.on_the_host(&cfg);
    for step in 1..4 {
        let label = format!("declined at codeword {step}");
        let streamed = gpu::joint_streamed_builds();
        let fallbacks = gpu::host_fallbacks();
        let (roots, proof) = {
            let _on = gpu::force_streamed_joint_leaves(true);
            let _decline = gpu::force_streamed_decline_at(Some(step));
            stack.try_prove(&cfg, Some(4))
        }
        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
        assert_eq!(
            gpu::joint_streamed_builds(),
            streamed,
            "{label}: nothing streamed"
        );
        assert_eq!(gpu::host_fallbacks(), fallbacks, "{label}: no host commit");
        assert_eq!(roots, host_roots, "{label}: roots");
        stack.verify(&roots, &proof, &cfg, &label);
        assert!(
            bytes(&proof) == bytes(&host_proof),
            "{label}: the host's bytes"
        );
    }
}
