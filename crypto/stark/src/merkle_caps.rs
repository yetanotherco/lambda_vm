//! Merkle caps of a univariate STARK proof (design/CAP.md §4, lever S1).
//!
//! Every tree of a proof is opened once per query: the trace trees (main,
//! precomputed, aux), the composition tree and each committed FRI layer. Under
//! a cap policy a tree of depth `D` gets a height-`c` cap
//! (`CapPolicy::height(num_queries, D)`); its `2^c` cap nodes ride at the end
//! of the authentication path of the tree's FIRST opening in proof order (the
//! "owner path"), and every path of that tree is cut to `D − c` siblings.
//!
//! [`StarkCaps`] is the one place the heights are computed from public shape
//! data (the policy, the query count, `log2(lde)` and the committed FRI layer
//! count). The prover embeds with it and the verifier checks with it, so the
//! split point of every path is a verifier constant, never read from a proof.
//!
//! [`TreeCheck`] is the verifier's per-tree check: built ONCE per tree (the
//! owner path's length and its cap-to-root check), then used for every query.
//! At `c = 0` it never touches the owner opening and is exactly the C1b
//! exact-length check, so the default format verifies the bytes it did.

use crypto::merkle_tree::cap::{CapPolicy, CappedRoot};
use crypto::merkle_tree::traits::IsMerkleTreeBackend;

use crate::config::Commitment;

/// The cap height of every tree of one table's proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StarkCaps {
    /// Depth of the trace, precomputed, aux and composition trees.
    pub trace_depth: usize,
    /// Cap height of those four trees (they share depth and opening count).
    pub trace: usize,
    /// Depth of committed FRI layer `i`.
    pub fri_depths: Vec<usize>,
    /// Cap height of committed FRI layer `i`.
    pub fri: Vec<usize>,
}

impl StarkCaps {
    /// Depth of the trace, precomputed, aux and composition trees: a leaf is a
    /// row PAIR, so `lde / 2` leaves and `log2(lde) − 1` levels (0 for a
    /// two-point LDE, where the leaf hash is the root).
    pub fn trace_tree_depth(lde_log: usize) -> usize {
        lde_log.saturating_sub(1)
    }

    /// Depth of committed FRI layer `i`: it holds `lde / 2^(i+1)` values in
    /// pair leaves, so `log2(lde) − i − 2` levels.
    pub fn fri_layer_depth(lde_log: usize, layer: usize) -> usize {
        lde_log.saturating_sub(layer + 2)
    }

    /// The heights for a proof with `num_queries` queries over an LDE of
    /// `2^lde_log` points and `num_committed` committed FRI layers of today's
    /// PAIR layout (layer `i` is `log2(lde) − i − 2` deep). Every tree is
    /// opened `num_queries` times. A proof under a fold schedule
    /// (`fri = dp`) has other layer depths: use [`Self::for_options`].
    pub fn new(
        policy: CapPolicy,
        num_queries: usize,
        lde_log: usize,
        num_committed: usize,
    ) -> Self {
        let trace_depth = Self::trace_tree_depth(lde_log);
        let fri_depths: Vec<usize> = (0..num_committed)
            .map(|i| Self::fri_layer_depth(lde_log, i))
            .collect();
        let fri = fri_depths
            .iter()
            .map(|&d| policy.height(num_queries, d))
            .collect();
        Self {
            trace_depth,
            trace: policy.height(num_queries, trace_depth),
            fri_depths,
            fri,
        }
    }

    /// The heights of a table proved under `options` over an LDE of
    /// `2^lde_log` points, with the table's RESOLVED leaf layout `one_row`
    /// (the same resolution `FriFoldLayout::for_options` takes): the trace
    /// trees are the layout's depth and the committed FRI layers are the ones
    /// the proof format's fold layout commits, so under a fold schedule
    /// (`fri = dp`) layer `j` is `layer_depth(j)` deep, not `log2(lde) − j − 2`.
    ///
    /// This is the one public entry point the in-guest verifier checks its own
    /// cap heights against (row pairs only there: `one_row = false`). `Err`
    /// for a format that cannot be laid out.
    pub fn for_options(
        options: &crate::proof::options::ProofOptions,
        lde_log: usize,
        one_row: bool,
    ) -> Result<Self, crate::fri::schedule::FriFormatError> {
        let blowup_log = (options.blowup_factor as u32).trailing_zeros();
        let layout = crate::fri::terminal::FriFoldLayout::for_options(
            lde_log as u32,
            blowup_log,
            options,
            one_row,
        )?;
        Ok(Self::from_layout(
            options.format.merkle_cap,
            options.fri_number_of_queries,
            lde_log,
            &layout,
        ))
    }

    /// The heights over an explicit FRI fold layout (the prover's and the
    /// verifier's route: each holds the layout it built from the options).
    /// The trace trees take the layout's leaf layout (row pairs: `log2(lde) −
    /// 1`; one row: `log2(lde)`) and committed layer `j` is
    /// `layout.layer_depth(lde_log, j)` deep: `log2(lde) − j − 2` under the
    /// all-ones row-pair schedule (so this is [`Self::new`] there), the group
    /// tree's depth under any other.
    pub(crate) fn from_layout(
        policy: CapPolicy,
        num_queries: usize,
        lde_log: usize,
        layout: &crate::fri::terminal::FriFoldLayout,
    ) -> Self {
        Self::with_depths(
            policy,
            num_queries,
            crate::leaf_layout::LeafLayout::from_one_row(layout.one_row).tree_depth(lde_log),
            layout.layer_depths(lde_log as u32),
        )
    }

    /// The heights for trace trees of depth `trace_depth` and committed FRI
    /// layers of depths `fri_depths`, every tree opened `num_queries` times.
    ///
    /// The general form of [`Self::new`], for any leaf layout and FRI
    /// schedule: the caller passes the depths its layout implies
    /// ([`crate::leaf_layout::LeafLayout::tree_depth`] and the FRI layout's
    /// per-layer depths). At row pairs and the all-ones schedule those are
    /// exactly [`Self::new`]'s.
    pub fn with_depths(
        policy: CapPolicy,
        num_queries: usize,
        trace_depth: usize,
        fri_depths: Vec<usize>,
    ) -> Self {
        let fri = fri_depths
            .iter()
            .map(|&d| policy.height(num_queries, d))
            .collect();
        Self {
            trace_depth,
            trace: policy.height(num_queries, trace_depth),
            fri_depths,
            fri,
        }
    }

    /// True when some tree has a cap (`c > 0`).
    pub fn any(&self) -> bool {
        self.trace > 0 || self.fri.iter().any(|&c| c > 0)
    }
}

/// The verifier's check for one tree: its authenticated cap, plus the owner
/// opening's own siblings when the tree is capped.
#[derive(Clone, Copy, Debug)]
pub struct TreeCheck<'a> {
    capped: CappedRoot<'a, Commitment>,
    /// `Some` iff `c > 0`: the owner path minus its cap. Query 0 of this tree
    /// is checked with these siblings.
    owner_siblings: Option<&'a [Commitment]>,
}

impl<'a> TreeCheck<'a> {
    /// Build the check of one tree of depth `depth` and cap height
    /// `cap_height` against `root`.
    ///
    /// At `c = 0` the owner opening is never read (`owner_path` is not
    /// called): the check is the exact-length full-path check, and the default
    /// format touches no index a count guard has not covered. At `c > 0`,
    /// `owner_path` must return the tree's first opening's path (`None` when
    /// the proof has none, which rejects); its length must be exactly
    /// `D − c + 2^c` and its cap must hash to `root`.
    pub fn build<B: IsMerkleTreeBackend<Node = Commitment>>(
        root: &'a Commitment,
        depth: usize,
        cap_height: usize,
        owner_path: impl FnOnce() -> Option<&'a [Commitment]>,
    ) -> Option<Self> {
        if cap_height == 0 {
            return Some(Self {
                capped: CappedRoot::uncapped(root, depth),
                owner_siblings: None,
            });
        }
        let (capped, siblings) =
            CappedRoot::from_owner::<B>(root, owner_path()?, depth, cap_height)?;
        Some(Self {
            capped,
            owner_siblings: Some(siblings),
        })
    }

    /// Check query `query`'s opening of this tree: `path` as the proof carries
    /// it, the transcript's leaf `index`, and the leaf hash of the opened
    /// values. Query 0 of a capped tree is the owner: its siblings are the
    /// owner path minus the cap (split once in [`build`](Self::build)); every
    /// other query's path must be exactly `D − c` long.
    pub fn verify<B: IsMerkleTreeBackend<Node = Commitment>>(
        &self,
        query: usize,
        path: &[Commitment],
        index: usize,
        leaf_hash: Commitment,
    ) -> bool {
        let siblings = match (query, self.owner_siblings) {
            (0, Some(owner)) => owner,
            _ => path,
        };
        self.capped.verify::<B>(siblings, index, leaf_hash)
    }

    pub fn cap_height(&self) -> usize {
        self.capped.cap_height()
    }
}

/// The checks of every tree of one table's proof.
#[derive(Clone, Debug)]
pub struct TableTreeChecks<'a> {
    pub main: TreeCheck<'a>,
    pub precomputed: Option<TreeCheck<'a>>,
    pub aux: Option<TreeCheck<'a>>,
    pub composition: TreeCheck<'a>,
    /// One per committed FRI layer, in layer order.
    pub fri: Vec<TreeCheck<'a>>,
}
