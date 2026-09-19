//! The WHIR cross-epoch driver: the one global input a block's tree needs.
//!
//! [`crate::lfm::whir_real_epoch`] is the level-0 analogue and this reads the
//! same way on purpose. ⚠ NOT `whir_global.rs` — that name is V1's, for the
//! emitter, exactly as `whir_real_epoch.rs` is not `whir_epoch.rs`.
//!
//! # Four ways the cross-epoch proof is not an epoch
//!
//! Each is real work for whatever emits its program, and each is why this is a
//! separate driver rather than an argument to the epoch's.
//!
//! 1. **Sixteen groups, not two.** Every bookend is committed ALONE so its root
//!    can be compared against the epoch that committed it — that comparison IS
//!    the cross-epoch binding — and the global-memory tables share one group.
//! 2. **No DECODE table and no prepared opening**, so the roots block carries
//!    no derived root and this driver has no `decode_commitment` and no
//!    `prepared` parameter. The once-per-bundle derivations the epoch driver
//!    carries have nothing to carry here.
//! 3. **The bus target is a literal zero** — the cross-epoch bus has no
//!    counterparty in the statement, so there is no published-bytes term and no
//!    commit index.
//! 4. **The statement is `absorb_global`**, whose argument list is the reason
//!    for this type's middle five fields.
//!
//! # What a driver owes, and what it must refuse to take
//!
//! The same rule the epoch driver follows: everything here is a value the
//! VERIFIER computes for itself. The epoch count, the page list and the private
//! page count are the bundle's declared shape, and they are not trusted — they
//! are bound into the transcript and pinned by the bus, so a restated set
//! leaves the GlobalMemory bus unbalanced or the AIR count mismatched, which is
//! what [`crate::multilinear_continuation::verify_global`] is asked before
//! anything here is built.

use multilinear::whir_chain::ChainConfig;
use stark::config::Commitment;

use executor::elf::Elf;

use crate::lfm::block_root::GlobalLayout;
use crate::lfm::proof_arena::lanes_per_root;
use crate::multilinear_continuation::{ContinuationProof, GlobalProof, WhirGlobalAirs};

/// The cross-epoch wrap's input: the proof, everything its statement absorbs,
/// the AIR set it is argued against, and the roots the binding compares.
///
/// The statement half is exactly
/// [`crate::multilinear_continuation::absorb_global`]'s argument list, because
/// an in-guest verifier's first job is to replay that absorb and any field it
/// cannot see is a challenge it cannot reproduce.
pub struct WhirRealGlobal {
    /// The cross-epoch proof itself, cloned out of the bundle.
    pub proof: GlobalProof,
    /// `statement::elf_digest(elf_bytes)` — the program this run was of.
    pub elf_digest: [u8; 32],
    /// The run's epoch count, which is also the bookend count.
    pub num_epochs: usize,
    /// How many of the touched pages are private-input pages, which is what
    /// decides each page table's preprocessed route.
    pub num_private_input_pages: usize,
    /// The bundle's touched page list, as it travels — the canonical order the
    /// AIRs are built in is the config build's, not necessarily this.
    pub page_bases: Vec<u64>,
    /// The parameters the cross-epoch argument ran at, rebuilt from the shapes
    /// rather than carried.
    pub config: ChainConfig,
    /// `(width, num_vars)` per table in sub-proof order: **the widths are the
    /// AIRs' and only the heights are the proof's.**
    ///
    /// ⚠ The proof states no width at all, and the two families' differ — 9 for
    /// a bookend, 4 for a page. `chain_config` takes the tallest STACK, whose
    /// height is a function of both, so a driver that assumed width 1 would
    /// derive a different query count from the one the proof was argued at.
    /// That is the epoch driver's own scar, and it is sharper here because a
    /// cross-epoch set is two families wide.
    pub shapes: Vec<(usize, usize)>,
    /// How the tables are committed: `num_epochs` singletons, then the pages.
    ///
    /// ★ Taken from the AIR SET's own split, never respelled from
    /// `global_groups(num_epochs, page_bases.len())` — that would be a second
    /// derivation of the thing [`WhirGlobalAirs`] exists to hold once, and the
    /// two are not even trivially equal: the set's page count is the
    /// CANONICALISED list's, not the wire list's.
    pub sizes: Vec<usize>,
    /// The AIR set, owned, built through
    /// [`crate::multilinear_continuation::global_airs_for`] — the same function
    /// the verifier builds through.
    ///
    /// ★ A FIELD, WHICH THE EPOCH DRIVER DOES NOT DO, because it can be: a
    /// `WhirGlobalAirs` owns its AIRs and carries no lifetime. `WhirEpochAirs`
    /// is held by the caller and passed to the builder separately, which is why
    /// "keep it alive for as long as the program build" had to be written down.
    /// Here the emitter cannot be handed a set built from arguments other than
    /// the ones this harvest verified against.
    airs: WhirGlobalAirs,
    /// The roots each epoch's bookend was committed under HERE, in epoch order
    /// — one window per epoch, as many roots as its group stacked into.
    ///
    /// This is the content of the published set: `GlobalLayout` publishes `z`,
    /// `alpha`, then these as lanes, and the root node compares them against
    /// the fold the interior carried up.
    pub bookend_roots: Vec<Vec<Commitment>>,
    /// What the cross-epoch wrap publishes, as a type rather than a count.
    pub published: GlobalLayout,
    /// The genesis stack this bundle's cross-epoch proof carries, or `None`
    /// when its genesis is entirely sparse — which most runs are.
    ///
    /// ★ TAKEN FROM THE VERY VERIFICATION THAT CONSUMED IT, never rebuilt. It
    /// carries the roots AND the `StackedLayout` and `Domain` the host
    /// committed under, because an emitter handed only the roots would have to
    /// derive those two a second time, and a second derivation of the object
    /// the host actually committed is the defect this whole struct's
    /// single-derivation rule exists to prevent.
    ///
    /// ⚠ NOT the 35 per-page roots `recursion::precomputed_commitments` builds.
    /// Those are UNIVARIATE Merkle roots over each page's LDE codeword, they
    /// are what the attestation's `program_id` folds, and **the multilinear
    /// path never compares one of them**. Two objects with confusable names
    /// over the same bytes: filling this from that list would build a program
    /// whose arena matched word for word and refused hundreds of thousands of
    /// rows later.
    ///
    /// ⛔ AND ITS ROOTS ARE THE FOURTH OWED PER-ELF PIN, for the DENSE pages
    /// only. The pages left to the sparse form owe a DIFFERENT thing — their
    /// nonzero genesis entries interned as program constants, bound by the
    /// program id — and the two obligations must never be written as one
    /// sentence, or whichever is actually unchecked looks covered by the other.
    /// See [`crate::multilinear_continuation::GlobalPrepared`].
    pub prepared: Option<crate::multilinear_continuation::GlobalPrepared>,
}

impl WhirRealGlobal {
    /// The AIR set this proof was verified against, for an emitter to build its
    /// layouts and statements from.
    ///
    /// ⚠ Borrowed, because the layouts built from it borrow in turn: a
    /// `layouts` field beside this one would be self-referential, which is the
    /// same reason a `statements` field is not on the epoch's driver output.
    pub fn airs(&self) -> &WhirGlobalAirs {
        &self.airs
    }

    /// How many tables the cross-epoch argument covers — the width of
    /// everything a guest walks.
    ///
    /// ⚠ From the AIR SET, not from `proof.table_num_vars.len()`. The two agree
    /// (the harvest refuses otherwise), and that is the point: reading the
    /// proof's own field back through a second name would be a count that
    /// cannot disagree with itself.
    pub fn num_tables(&self) -> usize {
        self.airs.refs().len()
    }
}

/// [`WhirRealGlobal`] for the cross-epoch half of an existing WHIR continuation
/// bundle, verified under the hash this PROCESS is set to.
///
/// The split mirrors [`crate::multilinear_continuation::verify_global`] against
/// `verify_global_bookends::<H>`, and
/// [`super::whir_real_epoch::real_epoch_from_whir_continuation`] against its own
/// `_under` form: one entry point that dispatches on the knob for production,
/// one that takes `H` so the hash agreement can be argued about — and tested —
/// at all.
pub fn real_global_from_whir_continuation(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
) -> Result<WhirRealGlobal, String> {
    crate::with_whir_hash!(|H| {
        real_global_from_whir_continuation_under::<H>(opts, elf_bytes, bundle)
    })
}

/// [`real_global_from_whir_continuation`], told which hash to verify under.
///
/// ★ THE PROOF IS VERIFIED BEFORE IT IS HARVESTED, for the epoch driver's
/// reason: an input built from a proof nobody checked pushes the failure into
/// the guest, where it costs a whole wrap prove to discover and reads as an
/// emitter bug.
///
/// ★ AND THIS IS WHERE THE HASH AGREEMENT LIVES, which is why the function is
/// generic rather than reading the knob. `whir_hash_knob::selected()` is a
/// cached process setting: it says what THIS PROCESS proves under, never what
/// the bundle in front of it was proven under. The agreement is the
/// verification — the transcript's sponge is part of the configuration, so a
/// bundle proven under another hash diverges from the first squeeze and fails
/// here. See
/// `crate::tests::multilinear_continuation_tests::a_cross_epoch_proof_proven_under_one_hash_is_refused_under_the_other`.
pub fn real_global_from_whir_continuation_under<H>(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
) -> Result<WhirRealGlobal, String>
where
    H: multilinear::whir_hash::WhirHash,
{
    let elf = Elf::load(elf_bytes).map_err(|e| format!("the inner ELF must load: {e}"))?;
    let num_epochs = bundle.epochs.len();
    if num_epochs == 0 {
        return Err("a bundle with no epochs has no cross-epoch proof to harvest".to_string());
    }

    // ★ The acceptance check, through the verifier's own bookend form — which
    // builds its AIR set through `global_airs_for`, the same function this
    // harvest calls below, and which hands back the roots the binding compares
    // instead of a bare `bool`.
    let verified = crate::multilinear_continuation::verify_global_bookends::<H>(
        &elf,
        elf_bytes,
        &bundle.global,
        num_epochs,
        &bundle.touched_page_bases,
        bundle.num_private_input_pages,
        opts,
    )
    .map_err(|e| format!("the cross-epoch proof could not be verified: {e:?}"))?;
    let Some(verified) = verified else {
        return Err(format!(
            "the cross-epoch proof of this bundle does not verify under {}. Either the \
             bundle is not the one this ELF and these options describe, or it was proven \
             under a different hash — which is checked cryptographically and not by a tag, \
             since a WHIR proof's bytes are hash-agnostic by design",
            <H as multilinear::whir_hash::WhirHash>::NAME,
        ));
    };
    // Both halves of the verdict, taken from the verification that produced
    // them: the roots the epoch binding compares, and the genesis stack the
    // emitter opens. Re-deriving either is what this return type exists to
    // prevent.
    let crate::multilinear_continuation::GlobalVerdict {
        bookend_roots,
        prepared,
    } = verified;

    // ★ THE VERIFIER'S OWN DERIVATION, not a second one that agrees.
    let airs = crate::multilinear_continuation::global_airs_for(
        &elf,
        opts,
        num_epochs,
        &bundle.touched_page_bases,
        bundle.num_private_input_pages,
    );
    let air_refs = airs.refs();
    if air_refs.len() != bundle.global.table_num_vars.len() {
        return Err(format!(
            "the cross-epoch layout has {} tables and the proof states {} heights",
            air_refs.len(),
            bundle.global.table_num_vars.len(),
        ));
    }
    let shapes: Vec<(usize, usize)> = air_refs
        .iter()
        .zip(&bundle.global.table_num_vars)
        .map(|(air, &num_vars)| (air.trace_layout().0, num_vars as usize))
        .collect();
    let config = crate::multilinear_prove::chain_config(&shapes);
    let sizes = airs.groups();

    // ★ The bookend roots are the VERIFICATION's, taken from the object that
    // accepted the proof rather than re-derived from `stacks` here. An earlier
    // draft re-derived them because only the `bool` form was reachable, and a
    // second spelling of the group split is exactly the drift this campaign
    // keeps finding — the same argument `global_airs_for` makes for the AIR
    // set, applied to the values the published set carries.
    if bookend_roots.len() != num_epochs {
        return Err(format!(
            "the cross-epoch proof chained {} bookends and the run has {num_epochs} epochs",
            bookend_roots.len(),
        ));
    }

    Ok(WhirRealGlobal {
        proof: bundle.global.clone(),
        elf_digest: crate::statement::elf_digest(elf_bytes),
        num_epochs,
        num_private_input_pages: bundle.num_private_input_pages,
        page_bases: bundle.touched_page_bases.clone(),
        config,
        shapes,
        sizes,
        airs,
        bookend_roots,
        published: GlobalLayout {
            num_epochs,
            lanes_per_root: lanes_per_root(),
        },
        // From the verdict above, which is the object the verification
        // consumed. `None` here means this run's genesis was entirely sparse,
        // not that the route is unbuilt.
        prepared,
    })
}
