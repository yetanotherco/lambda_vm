#![allow(dead_code)]
//! # ⛔ THE `dead_code` ALLOW, ITS CONDITION, AND WHAT IT COSTS WHILE IT STANDS
//!
//! This module is in the LIBRARY target, where every item is unreachable until
//! something PUBLICLY reachable uses it — and `make lint`'s first arm compiles
//! the lib target alone under `-D warnings`, so the whole class is a hard error
//! there while `cargo test` uses every item and reports nothing. Measured, not
//! assumed: without that line `cargo clippy -p lambda-vm-prover --lib` gives
//! three errors — the struct never constructed, `airs`/`num_tables` never used,
//! and the driver never called — because the only callers are this module's
//! tests.
//!
//! ★ THE CONDITION IS A PRODUCTION READER, NOT A PRODUCTION SIGNATURE, and that
//! distinction is [`super::whir_real_epoch`]'s scar: its allow was written to
//! come out "the moment `whir_epoch_program` takes a `WhirRealEpoch`", that
//! happened, and removing the allow put seven errors back — `dead_code` asks
//! what is REACHABLE, not what is named. So this comes out when the cross-epoch
//! program builder both TAKES a [`WhirRealGlobal`] and READS its fields; if it
//! is still here after that lands, something did not get wired.
//!
//! ⚠ What it costs meanwhile, stated so nobody has to guess: a genuinely unused
//! item added to this module is not reported while it stands.
//!
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
    pub(crate) proof: GlobalProof,
    /// `statement::elf_digest(elf_bytes)` — the program this run was of.
    pub(crate) elf_digest: [u8; 32],
    /// The run's epoch count, which is also the bookend count.
    pub(crate) num_epochs: usize,
    /// How many of the touched pages are private-input pages, which is what
    /// decides each page table's preprocessed route.
    pub(crate) num_private_input_pages: usize,
    /// The bundle's touched page list, as it travels — the canonical order the
    /// AIRs are built in is the config build's, not necessarily this.
    pub(crate) page_bases: Vec<u64>,
    /// The parameters the cross-epoch argument ran at, rebuilt from the shapes
    /// rather than carried.
    pub(crate) config: ChainConfig,
    /// `(width, num_vars)` per table in sub-proof order: **the widths are the
    /// AIRs' and only the heights are the proof's.**
    ///
    /// ⚠ The proof states no width at all, and the two families' differ — 9 for
    /// a bookend, 4 for a page. `chain_config` takes the tallest STACK, whose
    /// height is a function of both, so a driver that assumed width 1 would
    /// derive a different query count from the one the proof was argued at.
    /// That is the epoch driver's own scar, and it is sharper here because a
    /// cross-epoch set is two families wide.
    pub(crate) shapes: Vec<(usize, usize)>,
    /// How the tables are committed: `num_epochs` singletons, then the pages.
    ///
    /// ★ Taken from the AIR SET's own split, never respelled from
    /// `global_groups(num_epochs, page_bases.len())` — that would be a second
    /// derivation of the thing [`WhirGlobalAirs`] exists to hold once, and the
    /// two are not even trivially equal: the set's page count is the
    /// CANONICALISED list's, not the wire list's.
    pub(crate) sizes: Vec<usize>,
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
    pub(crate) bookend_roots: Vec<Vec<Commitment>>,
    /// What the cross-epoch wrap publishes, as a type rather than a count.
    pub(crate) published: GlobalLayout,
    /// ★ PER PAGE TABLE, IN THE AIR SET'S OWN ORDER: whether that page is a
    /// private-input page — the one bit that decides its preprocessed route.
    ///
    /// ⛔ IT IS THE CONFIG'S FLAG, NOT A COUNT OF COLUMNS. The emitter has to
    /// know which pages carry INIT, and the tempting source is how many
    /// preprocessed columns each AIR presents. That is a guard written on the
    /// answer: a page whose INIT column vanished upstream would present one
    /// column, be routed as private, and have its genesis checked by nothing —
    /// the reading could not move under the failure it exists to catch. So this
    /// is `PageConfig::is_private_input` of the very configs the AIR set was
    /// built from, and the column count becomes the emitter's CROSS-CHECK
    /// instead of its key.
    ///
    /// ⚠ A SECOND CALL OF `global_memory_configs`, not a second spelling of it.
    /// `global_airs_for` ran it internally and kept only the AIRs; this runs the
    /// same function on the same three arguments, so the two cannot disagree on
    /// anything but the arguments, which are derived once above. Exposing the
    /// configs from [`WhirGlobalAirs`] would remove even that, and is the change
    /// this comment exists to justify rather than to excuse.
    pub(crate) page_is_private: Vec<bool>,
    /// ⛔ RESERVED, AND EMPTY ON EVERY PATH THAT EXISTS TODAY: the roots of a
    /// MULTILINEAR commitment over the page family's INIT columns, for the
    /// prepared opening a cross-epoch program needs instead of folding ≈9.2 M
    /// genesis rows.
    ///
    /// ⚠ NOT the 35 per-page roots `recursion::precomputed_commitments` builds.
    /// Those are UNIVARIATE Merkle roots over each page's LDE codeword, they
    /// are what the attestation's `program_id` folds, and **the multilinear
    /// path never compares one of them** — `verify_global_bookends` checks
    /// INIT by folding the columns rebuilt from the ELF, with no opening and no
    /// root at all. Two objects with confusable names, one of which no verifier
    /// on this path reads: filling this field from that list would build a
    /// program whose arena matched word for word and refused hundreds of
    /// thousands of rows later. Whoever fills it takes the value from the very
    /// `Prepared` object the host verification consumed, and states the pin it
    /// owes where the root is interned.
    ///
    /// ⛔⛔ AND IT STAYS `None`, BECAUSE THE OBJECT CANNOT EXIST ON THIS PATH —
    /// measured by reading, not assumed. `multi_prove` lifts a prepared
    /// commitment's roots into the ROOTS BLOCK
    /// (`absorb_roots_and_challenge(transcript, committed.roots(), &prepared_roots)`),
    /// and both `prove_global` and `verify_global_bookends` pass `None`, so
    /// those roots are empty in every cross-epoch proof that exists. A program
    /// that absorbed one would absorb a root the honest proof never absorbed,
    /// derive a different `z`, and stop executing at the first table. Separately,
    /// `Prepared` names ONE table and settles its columns with `Claimed::Shared`
    /// at that table's single reduced point, while a cross-epoch INIT family
    /// spans one page table per touched page, each with its own point. So an
    /// INIT opening is not a machine-side addition: it is a change to the
    /// cross-epoch prover, its verifier, its proof bytes and `Prepared`'s shape.
    /// Until that is taken, [`crate::lfm::whir_global`] checks the genesis
    /// columns themselves and this field is honestly empty.
    pub(crate) prepared_roots: Option<Vec<Commitment>>,
}

impl WhirRealGlobal {
    /// The AIR set this proof was verified against, for an emitter to build its
    /// layouts and statements from.
    ///
    /// ⚠ Borrowed, because the layouts built from it borrow in turn: a
    /// `layouts` field beside this one would be self-referential, which is the
    /// same reason a `statements` field is not on the epoch's driver output.
    pub(crate) fn airs(&self) -> &WhirGlobalAirs {
        &self.airs
    }

    /// How many tables the cross-epoch argument covers — the width of
    /// everything a guest walks.
    ///
    /// ⚠ From the AIR SET, not from `proof.table_num_vars.len()`. The two agree
    /// (the harvest refuses otherwise), and that is the point: reading the
    /// proof's own field back through a second name would be a count that
    /// cannot disagree with itself.
    pub(crate) fn num_tables(&self) -> usize {
        self.airs.refs().len()
    }
}

/// [`WhirRealGlobal`] for the cross-epoch half of an existing WHIR continuation
/// bundle.
///
/// ★ THE PROOF IS VERIFIED BEFORE IT IS HARVESTED, for the epoch driver's
/// reason: an input built from a proof nobody checked pushes the failure into
/// the guest, where it costs a whole wrap prove to discover and reads as an
/// emitter bug.
///
/// ⚠ THE HASH AGREEMENT IS THE PROCESS KNOB'S HERE, NOT THE CALLER'S, and that
/// is a real difference from the epoch driver. `verify_epoch_bookend::<H>` is
/// generic, so `real_epoch_from_whir_continuation_under::<H>` can ask "was this
/// bundle proven under `H`" cryptographically. The cross-epoch verifier
/// dispatches on `whir_hash_knob::selected()` INSIDE itself, so this harvest
/// verifies under whatever hash the process is set to and cannot be told
/// otherwise. A bundle proven under the other hash still fails — every
/// challenge diverges from the first squeeze — but it fails against the
/// process's choice rather than the caller's, and no `_under::<H>` form can be
/// honest until the verifier's dispatch moves out to its callers the way the
/// epoch half's already has.
pub fn real_global_from_whir_continuation(
    opts: &crate::ProofOptions,
    elf_bytes: &[u8],
    bundle: &ContinuationProof,
) -> Result<WhirRealGlobal, String> {
    let elf = Elf::load(elf_bytes).map_err(|e| format!("the inner ELF must load: {e}"))?;
    let num_epochs = bundle.epochs.len();
    if num_epochs == 0 {
        return Err("a bundle with no epochs has no cross-epoch proof to harvest".to_string());
    }

    // ★ The acceptance check, through the verifier's own entry point — which
    // builds its AIR set through `global_airs_for`, the same function this
    // harvest calls below.
    let accepted = crate::multilinear_continuation::verify_global(
        &elf,
        elf_bytes,
        &bundle.global,
        num_epochs,
        &bundle.touched_page_bases,
        bundle.num_private_input_pages,
        opts,
    )
    .map_err(|e| format!("the cross-epoch proof could not be verified: {e:?}"))?;
    if !accepted {
        return Err(format!(
            "the cross-epoch proof of this bundle does not verify under {}. Either the \
             bundle is not the one this ELF and these options describe, or it was proven \
             under a different hash — which is checked cryptographically and not by a tag, \
             since a WHIR proof's bytes are hash-agnostic by design",
            crate::whir_hash_knob::selected().name(),
        ));
    }

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

    // The route decision, from the configs the AIR set was built from — see the
    // field's own doc for why it is not the column count.
    let page_is_private: Vec<bool> = crate::continuation::global_memory_configs(
        &bundle.touched_page_bases,
        &elf,
        bundle.num_private_input_pages,
    )
    .iter()
    .map(|config| config.is_private_input)
    .collect();
    if page_is_private.len() + num_epochs != air_refs.len() {
        return Err(format!(
            "the cross-epoch layout has {} tables and {num_epochs} bookends, which leaves              {} pages, and the ELF's page configs describe {}",
            air_refs.len(),
            air_refs.len().saturating_sub(num_epochs),
            page_is_private.len(),
        ));
    }

    // ⚠ A SECOND SPELLING OF TWO LINES THE VERIFIER ALREADY RAN, and it is here
    // only because `verify_global_bookends` — which computes exactly this and
    // returns it — is private to `multilinear_continuation` while
    // `verify_global` hands back a bare `bool`. The inputs are the ones derived
    // above, so the two cannot drift on anything but those two lines; making
    // the bookend form reachable would remove even that, and is the change this
    // comment exists to justify rather than to excuse.
    let (stacks, _domains) = crate::multilinear_prove::stacks(&shapes, &sizes, &config)
        .map_err(|e| format!("the cross-epoch stacks: {e:?}"))?;
    let polys: Vec<usize> = stacks[..num_epochs].iter().map(|l| l.num_polys()).collect();
    let bookend_roots: Vec<Vec<Commitment>> = bundle
        .global
        .l2g_roots(&polys)
        .ok_or_else(|| {
            format!(
                "the cross-epoch proof carries {} roots, too few for {num_epochs} bookends",
                bundle.global.proof.roots.len(),
            )
        })?
        .into_iter()
        .map(<[_]>::to_vec)
        .collect();

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
        page_is_private,
        published: GlobalLayout {
            num_epochs,
            lanes_per_root: lanes_per_root(),
        },
        // ⛔ There is no INIT opening yet, and `None` is the honest state of it.
        // See the field's own doc for the list it must NOT be filled from.
        prepared_roots: None,
    })
}
