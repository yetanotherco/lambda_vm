//! Committing a codeword by fold blocks, so each query is one Merkle opening.
//!
//! The pre-image of folded index `j` is the stride-`N/2^k` coset
//! `{ j, j + N/2^k, …, j + (2^k - 1)·N/2^k }`.

use crypto::merkle_tree::{
    cap::{CapPolicy, CappedRoot, MAX_CAP_HEIGHT, embed_cap_arity, tree_levels},
    merkle::MerkleTree,
    proof::Proof,
    traits::IsMerkleTreeBackend,
    utils::{level_offsets4, level_sizes4},
};
use math::{
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsPrimeField, IsSubFieldOf},
    },
    traits::AsBytes,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::{
    Error,
    whir::Domain,
    whir_hash::{KeccakWhir, WhirHash},
};

/// 32-byte commitments, matching the rest of the prover.
///
/// ★ **The width is the same for every [`WhirHash`]** — a keccak digest is 32
/// bytes and an algebraic digest is four canonical Goldilocks felts, which is
/// also 32 bytes. That is what keeps a hash swap out of the proof format: every
/// type below and above this one keeps its layout, its rkyv derives and its
/// serialized length.
pub type Commitment = [u8; 32];
type Backend<F, H> = <H as WhirHash>::Backend<F>;
type Tree<F, H> = MerkleTree<Backend<F, H>>;

/// A committed codeword and the tree needed to open it.
pub struct CodewordCommitment<F: IsField + 'static, H: WhirHash = KeccakWhir>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    tree: Tree<F, H>,
    codeword: Codeword<F>,
    log_folding: usize,
    log_domain_size: usize,
    /// The top of the tree, when the commitment was retired and revived
    /// ([`Self::retire`], [`RetiredCommitment::revive`]): the paths are then
    /// read from it and from the leaves under one kept node, re-hashed from
    /// the codeword. `None` on every commitment that kept its tree.
    top: Option<TreeTop>,
}

/// Paths served from a kept tree top, and the leaves re-hashed on the host to
/// serve them — the whole cost of not keeping the bottom levels. A revived
/// commitment never builds a device tree (there is no code path for it), so
/// these two are all a phase-B opening spends on its first round's tree.
static TOP_PATH_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TOP_LEAVES_REHASHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Nanoseconds the kept-top paths spent gathering their blocks from the
/// codeword, and re-hashing them on the host (the subtrees and their check).
static TOP_GATHER_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TOP_REHASH_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `BLOCK_WHIR_REHASH_SERIAL=1`: the kept-top paths re-hash their blocks one
/// at a time, as before they were re-hashed in parallel. A measurement knob
/// (the same bytes either way); read once.
#[cfg(feature = "parallel")]
fn rehash_serial() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("BLOCK_WHIR_REHASH_SERIAL").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// `(seconds gathering, seconds re-hashing)` of the kept-top paths so far.
pub fn top_path_secs() -> (f64, f64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        TOP_GATHER_NANOS.load(Relaxed) as f64 / 1e9,
        TOP_REHASH_NANOS.load(Relaxed) as f64 / 1e9,
    )
}

/// The kept-top paths of codewords on the card, by route: re-hashed on the card
/// (the default), or on the host when the card's re-hash failed or
/// `LAMBDA_VM_TOP_PATHS_HOST=1` asked for it — the fallback, ≈ 16× today's host
/// cost at 8 dropped levels, which a block reports as a WARN line.
static TOP_DEVICE_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TOP_DEVICE_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TOP_HOST_FALLBACK_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static TOP_HOST_FALLBACK_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `LAMBDA_VM_TOP_PATHS_HOST=1`: a codeword on the card has its kept-top paths
/// re-hashed on the host, as when the card's re-hash fails (the fallback's
/// measurement and its gate). Read once.
fn top_paths_on_host() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("LAMBDA_VM_TOP_PATHS_HOST").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}

/// `(calls, seconds)` of the kept-top paths of codewords on the card, re-hashed
/// on the card and on the host (the fallback): `((card calls, card s), (host
/// calls, host s))`.
pub fn top_path_routes() -> ((u64, f64), (u64, f64)) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        (
            TOP_DEVICE_CALLS.load(Relaxed),
            TOP_DEVICE_NANOS.load(Relaxed) as f64 / 1e9,
        ),
        (
            TOP_HOST_FALLBACK_CALLS.load(Relaxed),
            TOP_HOST_FALLBACK_NANOS.load(Relaxed) as f64 / 1e9,
        ),
    )
}

/// `(open_many calls served from a kept top, leaves re-hashed for them)`.
pub fn top_path_counts() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        TOP_PATH_CALLS.load(Relaxed),
        TOP_LEAVES_REHASHED.load(Relaxed),
    )
}

/// How many bottom levels a retired commitment's tree drops
/// ([`CodewordCommitment::retire`]): `device` for a codeword the card holds,
/// whose dropped levels are re-hashed on the card when a path needs them, and
/// `host` for one on the host, re-hashed there. Each level dropped halves the
/// kept top and doubles the leaves a path re-hashes. Clamped per tree to its
/// depth less the tallest cap it may be asked for.
///
/// ★ A PROVER-SIDE MEMORY CHOICE, NOT A FORMAT ONE (see [`TreeTop`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TreeDrop {
    pub device: usize,
    pub host: usize,
}

impl TreeDrop {
    /// The same depth wherever the codeword is.
    pub const fn uniform(levels: usize) -> Self {
        Self {
            device: levels,
            host: levels,
        }
    }

    /// The levels to drop from a tree `depth` deep whose tallest cap is
    /// `tallest_cap`, its codeword on the card or not: never into the cap.
    pub fn levels(self, on_device: bool, depth: usize, tallest_cap: usize) -> usize {
        let wanted = if on_device { self.device } else { self.host };
        wanted.min(depth.saturating_sub(tallest_cap))
    }

    /// [`Self::levels`] for a tree of `arity` children per node over
    /// `2^depth` leaves, `tallest_cap` counting that arity's levels. At arity
    /// 4 the drop counts 4-ary levels, half the binary count, so a dropped
    /// block keeps the leaves it has at arity 2 (8 → 4: 256 leaves); and it
    /// stops where a kept node covers `4^dropped` whole leaves, which an odd
    /// tree's two-node top does not.
    pub fn levels_at(
        self,
        on_device: bool,
        depth: usize,
        tallest_cap: usize,
        arity: usize,
    ) -> usize {
        if arity != 4 {
            return self.levels(on_device, depth, tallest_cap);
        }
        let wanted = if on_device { self.device } else { self.host } / 2;
        wanted
            .min(depth / 2)
            .min(tree_levels(depth, 4).saturating_sub(tallest_cap))
    }
}

/// The cap height of one tree over `2^depth` leaves of `arity` children per
/// node, opened `openings` times under `policy`: `policy.height` over the
/// tree's levels (4-ary levels at arity 4), and at arity 4 no taller than a cap
/// of `2^MAX_CAP_HEIGHT` nodes. At arity 2 it is `policy.height(openings,
/// depth)`, today's. Prover and verifier both derive it from public numbers.
pub fn tree_cap_height(policy: CapPolicy, openings: usize, depth: usize, arity: usize) -> usize {
    let c = policy.height(openings, tree_levels(depth, arity));
    if arity == 4 {
        c.min(MAX_CAP_HEIGHT / 2)
    } else {
        c
    }
}

/// Nodes a kept top holds: the levels above the `dropped` bottom ones, root
/// first — `2^(depth − dropped + 1) − 1` at arity 2, the matching prefix of
/// the arity-4 layout (`MerkleTree`'s, top-down) at arity 4.
fn kept_top_nodes(num_leaves: usize, depth: usize, dropped: usize, arity: usize) -> usize {
    if arity == 4 {
        level_sizes4(num_leaves)[dropped..].iter().sum()
    } else {
        (1usize << (depth - dropped + 1)) - 1
    }
}

impl std::fmt::Display for TreeDrop {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "card {} · host {}", self.device, self.host)
    }
}

/// What a retired commitment keeps of its tree: the heap prefix down to the
/// level whose nodes each cover `2^dropped` leaves (`4^dropped` at arity 4,
/// whose prefix is the arity-4 layout's top levels).
///
/// ★ A PROVER-SIDE MEMORY CHOICE, NOT A FORMAT ONE. The paths it answers are
/// the paths of the whole tree, byte for byte: the levels it dropped are
/// rebuilt from the codeword's queried cosets, and the rebuilt node is checked
/// against the kept one before any path leaves.
#[derive(Clone, Debug)]
pub struct TreeTop {
    /// `2^(depth − dropped + 1) − 1` nodes, root first (the host layout), or
    /// the arity-4 layout's top levels.
    nodes: Vec<Commitment>,
    /// Levels dropped, in the tree's own levels (4-ary at arity 4).
    dropped: usize,
}

impl TreeTop {
    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.nodes.len() * core::mem::size_of::<Commitment>()
    }
}

/// A commitment whose codeword was let go after its root was taken: the root,
/// the top of its tree and its shape. [`Self::revive`] makes it openable again
/// once the caller has recomputed the same codeword.
#[derive(Clone, Debug)]
pub struct RetiredCommitment {
    root: Commitment,
    top: TreeTop,
    log_folding: usize,
    log_domain_size: usize,
}

impl RetiredCommitment {
    pub fn root(&self) -> Commitment {
        self.root
    }

    /// Host bytes this keeps of the tree.
    pub fn tree_bytes(&self) -> usize {
        self.top.bytes()
    }

    /// The bottom levels its tree dropped.
    pub fn dropped(&self) -> usize {
        self.top.dropped
    }

    pub fn log_domain_size(&self) -> usize {
        self.log_domain_size
    }

    /// The commitment again, over `codeword` — which must be the committed
    /// codeword. A wrong one is caught when a path is read: its leaves then do
    /// not hash to the kept node ([`Error::RecomputedCodewordMismatch`]), so no
    /// path of a codeword the root does not bind can leave.
    pub fn revive<F: IsField + 'static, H: WhirHash>(
        self,
        codeword: Codeword<F>,
    ) -> Result<CodewordCommitment<F, H>, Error>
    where
        FieldElement<F>: AsBytes + Sync + Send,
    {
        if codeword.len() != 1usize << self.log_domain_size {
            return Err(Error::CodewordTooShort {
                coefficients: codeword.len(),
                domain: 1usize << self.log_domain_size,
            });
        }
        Ok(CodewordCommitment {
            tree: Tree::<F, H>::from_root(self.root),
            codeword,
            log_folding: self.log_folding,
            log_domain_size: self.log_domain_size,
            top: Some(self.top),
        })
    }
}

/// Where a commitment's codeword lives.
///
/// A device commit leaves it there and the chain folds it there: it is the
/// biggest array the proof holds, and all the host needs of it is the handful
/// of values a query opens.
#[derive(Debug)]
pub enum Codeword<F: IsField> {
    Host(Vec<FieldElement<F>>),
    Device(crate::gpu::DeviceCodeword),
}

impl<F: IsField> Codeword<F> {
    pub fn len(&self) -> usize {
        match self {
            Self::Host(values) => values.len(),
            Self::Device(device) => device.elements(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values on the host, when they are there.
    pub fn host(&self) -> Option<&[FieldElement<F>]> {
        match self {
            Self::Host(values) => Some(values),
            Self::Device(_) => None,
        }
    }

    /// The device holding them, when one does.
    pub fn device(&self) -> Option<&crate::gpu::DeviceCodeword> {
        match self {
            Self::Host(_) => None,
            Self::Device(device) => Some(device),
        }
    }
}

/// One opened block, with its authentication path.
#[derive(
    Clone,
    Debug,
    serde::Serialize,
    serde::Deserialize,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[serde(bound = "")]
pub struct CosetOpening<F: IsField> {
    /// The `2^k` codeword values, in coset order.
    pub values: Vec<FieldElement<F>>,
    pub proof: Proof<Commitment>,
}

/// A node's sibling in the host heap layout (root at 0, children `2i + 1` and
/// `2i + 2`): a right child's is the node before it, a left child's the one
/// after.
fn sibling(pos: usize) -> usize {
    if pos.is_multiple_of(2) {
        pos - 1
    } else {
        pos + 1
    }
}

/// The codeword positions that fold onto `index`.
///
/// `log_folding` is `k`; the coset has `2^k` entries at stride `N / 2^k`.
pub fn coset_of(index: usize, log_domain_size: usize, log_folding: usize) -> Vec<usize> {
    let stride = 1usize << (log_domain_size - log_folding);
    (0..(1usize << log_folding))
        .map(|t| index + t * stride)
        .collect()
}

/// Hand-written: the Merkle tree behind it is not `Debug`, and its nodes are
/// not what a reader of this type wants to see anyway.
/// Where a codeword position lives in the committed blocks.
///
/// Blocks are strided, so position `p` sits in leaf `p mod num_leaves` at slot
/// `p / num_leaves`. The inverse of [`coset_of`].
pub fn leaf_and_slot(position: usize, num_leaves: usize) -> (usize, usize) {
    (position % num_leaves, position / num_leaves)
}

impl<F: IsField + 'static, H: WhirHash> std::fmt::Debug for CodewordCommitment<F, H>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodewordCommitment")
            .field("root", &self.root())
            .field("leaves", &self.num_leaves())
            .field("log_folding", &self.log_folding)
            .field("log_domain_size", &self.log_domain_size)
            .finish()
    }
}

impl<F: IsField + 'static, H: WhirHash> CodewordCommitment<F, H>
where
    FieldElement<F>: AsBytes + Sync + Send,
{
    /// Groups `codeword` into fold blocks and Merkle-commits them.
    pub fn new(codeword: &[FieldElement<F>], log_folding: usize) -> Result<Self, Error> {
        Self::from_codeword(codeword.to_vec(), log_folding)
    }

    /// The same, taking the codeword. The commitment keeps it — the prover
    /// folds from here rather than encoding a second time — so a caller that
    /// has no further use for its own copy hands it over instead of paying for
    /// a second one.
    pub fn from_codeword(
        codeword: Vec<FieldElement<F>>,
        log_folding: usize,
    ) -> Result<Self, Error> {
        if !codeword.len().is_power_of_two() {
            return Err(Error::NotPowerOfTwo(codeword.len()));
        }
        let log_domain_size = codeword.len().trailing_zeros() as usize;
        if log_folding > log_domain_size {
            return Err(Error::ColumnTallerThanStack {
                column_vars: log_folding,
                n_stack: log_domain_size,
            });
        }

        if let Some(nodes) = crate::gpu::commit_tree_ext3(&codeword, log_folding, H::DEVICE) {
            let tree = Tree::<F, H>::from_precomputed_nodes(nodes).ok_or(Error::EmptyPolynomial)?;
            return Ok(Self {
                tree,
                codeword: Codeword::Host(codeword),
                log_folding,
                log_domain_size,
                top: None,
            });
        }
        Self::from_codeword_on_host(codeword, log_folding)
    }

    /// The same with the leaves hashed here, whatever a device would have done
    /// — the reference the kernel is checked against.
    pub fn from_codeword_on_host(
        codeword: Vec<FieldElement<F>>,
        log_folding: usize,
    ) -> Result<Self, Error> {
        if !codeword.len().is_power_of_two() {
            return Err(Error::NotPowerOfTwo(codeword.len()));
        }
        let log_domain_size = codeword.len().trailing_zeros() as usize;
        if log_folding > log_domain_size {
            return Err(Error::ColumnTallerThanStack {
                column_vars: log_folding,
                n_stack: log_domain_size,
            });
        }
        let num_leaves = 1usize << (log_domain_size - log_folding);
        let block = 1usize << log_folding;
        // One reused buffer per worker: a block is the leaf the backend hashes,
        // and the coset is strided, so it has to be gathered somewhere.
        let hash_leaf = |buffer: &mut Vec<FieldElement<F>>, j: usize| {
            buffer.clear();
            buffer.extend((0..block).map(|t| codeword[j + t * num_leaves].clone()));
            Backend::<F, H>::hash_data(buffer)
        };
        #[cfg(feature = "parallel")]
        let hashed: Vec<_> = (0..num_leaves)
            .into_par_iter()
            .map_init(|| Vec::with_capacity(block), hash_leaf)
            .collect();
        #[cfg(not(feature = "parallel"))]
        let hashed: Vec<_> = {
            let mut buffer = Vec::with_capacity(block);
            (0..num_leaves).map(|j| hash_leaf(&mut buffer, j)).collect()
        };

        let tree = Tree::<F, H>::build_from_hashed_leaves(hashed).ok_or(Error::EmptyPolynomial)?;
        Ok(Self {
            tree,
            codeword: Codeword::Host(codeword),
            log_folding,
            log_domain_size,
            top: None,
        })
    }

    /// The same, with the codeword and the tree both already computed — a
    /// device commit that handed the codeword back. The nodes carry no proof
    /// of their own correctness, so the caller answers for the layout:
    /// `2*num_leaves - 1` nodes, root first, leaves last.
    pub fn from_precomputed(
        codeword: Vec<FieldElement<F>>,
        nodes: Vec<Commitment>,
        log_folding: usize,
    ) -> Result<Self, Error> {
        if !codeword.len().is_power_of_two() {
            return Err(Error::NotPowerOfTwo(codeword.len()));
        }
        let log_domain_size = codeword.len().trailing_zeros() as usize;
        if log_folding > log_domain_size {
            return Err(Error::ColumnTallerThanStack {
                column_vars: log_folding,
                n_stack: log_domain_size,
            });
        }
        let tree = Tree::<F, H>::from_precomputed_nodes(nodes).ok_or(Error::EmptyPolynomial)?;
        Ok(Self {
            tree,
            codeword: Codeword::Host(codeword),
            log_folding,
            log_domain_size,
            top: None,
        })
    }

    /// A commitment whose codeword stays on the device that built it.
    ///
    /// Only the root comes back. The tree is rebuilt there when the queries
    /// are known — see [`paths`](Self::paths) — so this holds a root-only
    /// tree and nothing else.
    pub fn from_device(
        codeword: crate::gpu::DeviceCodeword,
        root: Commitment,
        log_folding: usize,
    ) -> Result<Self, Error> {
        let elements = codeword.elements();
        if !elements.is_power_of_two() {
            return Err(Error::NotPowerOfTwo(elements));
        }
        let log_domain_size = elements.trailing_zeros() as usize;
        if log_folding > log_domain_size {
            return Err(Error::ColumnTallerThanStack {
                column_vars: log_folding,
                n_stack: log_domain_size,
            });
        }
        Ok(Self {
            tree: Tree::<F, H>::from_root(root),
            codeword: Codeword::Device(codeword),
            log_folding,
            log_domain_size,
            top: None,
        })
    }

    pub fn root(&self) -> Commitment {
        self.tree.root
    }

    pub fn num_leaves(&self) -> usize {
        1usize << (self.log_domain_size - self.log_folding)
    }

    /// `log2(num_leaves)`: the index bits, whatever the arity (siblings on a
    /// full path at arity 2).
    pub fn depth(&self) -> usize {
        self.log_domain_size - self.log_folding
    }

    /// Children per node of this commitment's tree.
    pub fn arity() -> usize {
        <Backend<F, H> as IsMerkleTreeBackend>::ARITY
    }

    /// The tree's levels: [`Self::depth`] at arity 2, `⌈depth / 2⌉` at arity
    /// 4. Cap heights and kept tops count these.
    pub fn levels(&self) -> usize {
        tree_levels(self.depth(), Self::arity())
    }

    /// Siblings on a full authentication path: one per level at arity 2,
    /// three per level at arity 4.
    pub fn path_len(&self) -> usize {
        (Self::arity() - 1) * self.levels()
    }

    pub fn log_folding(&self) -> usize {
        self.log_folding
    }

    pub fn log_domain_size(&self) -> usize {
        self.log_domain_size
    }

    /// The committed codeword, in domain order.
    ///
    /// The prover folds from here rather than encoding a second time — on a
    /// real trace that second NTT is the most expensive thing in the proof
    /// after the sumcheck, and it computes something already in memory.
    pub fn codeword(&self) -> &Codeword<F> {
        &self.codeword
    }

    /// Lets the codeword go and keeps the root and the tree's top: every level
    /// but the bottom `drop` levels for where the codeword is ([`TreeDrop`]) —
    /// fewer when the tree is shallower, and never into the Merkle cap `cap`
    /// may ask this tree for, which is read from the kept levels. What the
    /// block prover holds between committing a group and opening it. `cap` is
    /// the policy at this tree's arity (`ChainFormat::cap_policy_at`).
    pub fn retire(
        self,
        drop: TreeDrop,
        cap: crypto::merkle_tree::cap::CapPolicy,
    ) -> Result<RetiredCommitment, Error> {
        if let Some(top) = self.top {
            // Already a revived one: its top is what it keeps.
            return Ok(RetiredCommitment {
                root: self.tree.root,
                top,
                log_folding: self.log_folding,
                log_domain_size: self.log_domain_size,
            });
        }
        let depth = self.depth();
        let arity = Self::arity();
        // The tallest cap the policy gives a tree this deep, whatever its
        // opening count (the height is monotone in it).
        let tallest_cap = tree_cap_height(cap, usize::MAX, depth, arity);
        let on_device = matches!(self.codeword, Codeword::Device(_));
        let dropped = drop.levels_at(on_device, depth, tallest_cap, arity);
        let keep = kept_top_nodes(self.num_leaves(), depth, dropped, arity);
        let nodes =
            match &self.codeword {
                Codeword::Device(device) => device
                    .top_nodes(self.log_folding, keep, H::DEVICE)
                    .ok_or(Error::DeviceFailed {
                        stage: "retiring a commitment",
                    })?,
                Codeword::Host(_) => {
                    let all = self.tree.nodes();
                    if all.len() < keep {
                        return Err(Error::DeviceFailed {
                            stage: "retiring a commitment with no host tree",
                        });
                    }
                    all[..keep].to_vec()
                }
            };
        if nodes.len() != keep || nodes.first() != Some(&self.tree.root) {
            return Err(Error::RecomputedCodewordMismatch { block: 0 });
        }
        Ok(RetiredCommitment {
            root: self.tree.root,
            top: TreeTop { nodes, dropped },
            log_folding: self.log_folding,
            log_domain_size: self.log_domain_size,
        })
    }

    /// The paths of `indices` from the kept top: each queried block of
    /// `2^dropped` leaves is re-hashed — on the card when the codeword is
    /// there ([`Self::card_top_paths`]), else, or when the card's re-hash
    /// fails, on the host ([`Self::host_top_paths`]) — its subtree's root is
    /// checked against the kept node, and the path continues through the kept
    /// levels. Either route gives the same paths, byte for byte.
    fn top_paths(&self, top: &TreeTop, indices: &[usize]) -> Result<Vec<Proof<Commitment>>, Error> {
        use std::sync::atomic::Ordering::Relaxed;
        let num_leaves = self.num_leaves();
        if let Some(&bad) = indices.iter().find(|index| **index >= num_leaves) {
            return Err(Error::QueryOutOfRange {
                index: bad,
                bound: num_leaves,
            });
        }
        let on_device = matches!(self.codeword, Codeword::Device(_));
        if on_device && !top_paths_on_host() {
            let started = std::time::Instant::now();
            if let Some(paths) = self.card_top_paths(top, indices) {
                TOP_DEVICE_CALLS.fetch_add(1, Relaxed);
                TOP_DEVICE_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
                return paths;
            }
        }
        let started = std::time::Instant::now();
        let paths = self.host_top_paths(top, indices);
        if on_device {
            TOP_HOST_FALLBACK_CALLS.fetch_add(1, Relaxed);
            TOP_HOST_FALLBACK_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        }
        paths
    }

    /// [`Self::top_paths`] on the card: the queried blocks' subtrees hashed and
    /// built where the codeword lies (`DeviceCodeword::block_subtrees`), and
    /// only their nodes brought home. `None` when the card could not (the
    /// caller falls back to the host); a block whose rebuilt root is not the
    /// kept node is refused, as on the host.
    fn card_top_paths(
        &self,
        top: &TreeTop,
        indices: &[usize],
    ) -> Option<Result<Vec<Proof<Commitment>>, Error>> {
        let Codeword::Device(device) = &self.codeword else {
            return None;
        };
        // The card re-hashes binary subtrees only; an arity-4 tree's kept-top
        // paths are served by the host (`host_top_paths`).
        if Self::arity() != 2 {
            return None;
        }
        let dropped = top.dropped;
        let span = 1usize << dropped;
        let mut blocks: Vec<usize> = indices.iter().map(|index| index >> dropped).collect();
        blocks.sort_unstable();
        blocks.dedup();
        let nodes = device.block_subtrees(self.log_folding, &blocks, dropped, H::DEVICE)?;
        let leaves = (blocks.len() << dropped).next_power_of_two().max(2);
        if nodes.len() != 2 * leaves - 1 {
            return None;
        }
        TOP_PATH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        TOP_LEAVES_REHASHED.fetch_add(
            (blocks.len() << dropped) as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let frontier = (1usize << (self.depth() - dropped)) - 1;
        // The blocks are `2^dropped`-aligned among the gathered leaves, so
        // block `k`'s root sits `dropped` levels up, `k`-th on its level.
        let roots_at = (leaves >> dropped) - 1;
        for (k, &block) in blocks.iter().enumerate() {
            if top.nodes.get(frontier + block) != Some(&nodes[roots_at + k]) {
                return Some(Err(Error::RecomputedCodewordMismatch { block }));
            }
        }
        let paths = indices
            .iter()
            .map(|&index| {
                let block = index >> dropped;
                let k = blocks
                    .binary_search(&block)
                    .map_err(|_| Error::QueryOutOfRange {
                        index,
                        bound: self.num_leaves(),
                    })?;
                let mut merkle_path = Vec::with_capacity(self.depth());
                let mut pos = (leaves - 1) + (k << dropped) + (index & (span - 1));
                for _ in 0..dropped {
                    merkle_path.push(nodes[sibling(pos)]);
                    pos = (pos - 1) / 2;
                }
                let mut pos = frontier + block;
                while pos != 0 {
                    merkle_path.push(top.nodes[sibling(pos)]);
                    pos = (pos - 1) / 2;
                }
                Ok(Proof { merkle_path })
            })
            .collect();
        Some(paths)
    }

    /// [`Self::top_paths`] on the host: the leaves of each queried block are
    /// gathered from the codeword and hashed here.
    fn host_top_paths(
        &self,
        top: &TreeTop,
        indices: &[usize],
    ) -> Result<Vec<Proof<Commitment>>, Error> {
        let num_leaves = self.num_leaves();
        // A block is the `span` leaves under one kept frontier node: `2^d` at
        // arity 2, `4^d` at arity 4.
        let block_bits = if Self::arity() == 4 {
            2 * top.dropped
        } else {
            top.dropped
        };
        let span = 1usize << block_bits;
        let mut blocks: Vec<usize> = indices.iter().map(|index| index >> block_bits).collect();
        blocks.sort_unstable();
        blocks.dedup();
        let leaves: Vec<usize> = blocks
            .iter()
            .flat_map(|block| (block << block_bits)..((block + 1) << block_bits))
            .collect();
        let gathering = std::time::Instant::now();
        let values = self.gather(&leaves)?;
        TOP_GATHER_NANOS.fetch_add(
            gathering.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        let rehashing = std::time::Instant::now();
        TOP_PATH_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        TOP_LEAVES_REHASHED.fetch_add(leaves.len() as u64, std::sync::atomic::Ordering::Relaxed);
        let frontier = self.frontier(top);
        // Each queried block's subtree depends on its own leaves alone, so the
        // blocks are re-hashed in parallel; the subtrees stay in block order
        // and the first block (in that order) whose root is not the kept node
        // is the one refused, as a serial walk would.
        let rehash = |k: usize| -> Result<Tree<F, H>, Error> {
            let block = blocks[k];
            let hashed: Vec<Commitment> = values[k * span..(k + 1) * span]
                .iter()
                .map(Backend::<F, H>::hash_data)
                .collect();
            let subtree =
                Tree::<F, H>::build_from_hashed_leaves(hashed).ok_or(Error::EmptyPolynomial)?;
            if top.nodes.get(frontier + block) != Some(&subtree.root) {
                return Err(Error::RecomputedCodewordMismatch { block });
            }
            Ok(subtree)
        };
        #[cfg(feature = "parallel")]
        let rehashed: Vec<Result<Tree<F, H>, Error>> = if rehash_serial() {
            (0..blocks.len()).map(rehash).collect()
        } else {
            (0..blocks.len()).into_par_iter().map(rehash).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let rehashed: Vec<Result<Tree<F, H>, Error>> = (0..blocks.len()).map(rehash).collect();
        let subtrees = rehashed.into_iter().collect::<Result<Vec<_>, _>>()?;
        TOP_REHASH_NANOS.fetch_add(
            rehashing.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        indices
            .iter()
            .map(|&index| {
                let block = index >> block_bits;
                let at = blocks
                    .binary_search(&block)
                    .map_err(|_| Error::QueryOutOfRange {
                        index,
                        bound: num_leaves,
                    })?;
                let mut merkle_path = subtrees[at]
                    .get_proof_by_pos(index & (span - 1))
                    .ok_or(Error::QueryOutOfRange {
                        index,
                        bound: num_leaves,
                    })?
                    .merkle_path;
                self.push_kept_siblings(top, block, &mut merkle_path)?;
                Ok(Proof { merkle_path })
            })
            .collect()
    }

    /// Where the kept top's frontier level (the nodes over the dropped
    /// blocks) starts in its node vector.
    fn frontier(&self, top: &TreeTop) -> usize {
        if Self::arity() == 4 {
            level_offsets4(&level_sizes4(self.num_leaves()))[top.dropped]
        } else {
            (1usize << (self.depth() - top.dropped)) - 1
        }
    }

    /// The siblings of frontier node `block`'s path up the kept levels, read
    /// from the top: one per level at arity 2; at arity 4 the three other
    /// children of its group in child order, the padding digest where the
    /// group is short (an odd tree's top), as `MerkleTree`'s own paths.
    fn push_kept_siblings(
        &self,
        top: &TreeTop,
        block: usize,
        path: &mut Vec<Commitment>,
    ) -> Result<(), Error> {
        let missing = || Error::RecomputedCodewordMismatch { block };
        if Self::arity() != 4 {
            let mut pos = self.frontier(top) + block;
            while pos != 0 {
                path.push(*top.nodes.get(sibling(pos)).ok_or_else(missing)?);
                pos = (pos - 1) / 2;
            }
            return Ok(());
        }
        let sizes = level_sizes4(self.num_leaves());
        let offsets = level_offsets4(&sizes);
        let pad = Backend::<F, H>::padding_node();
        let mut pos = block;
        for level in top.dropped..sizes.len() - 1 {
            let first = pos / 4 * 4;
            for c in (first..first + 4).filter(|&c| c != pos) {
                let node = if c < sizes[level] {
                    top.nodes.get(offsets[level] + c).copied()
                } else {
                    pad
                };
                path.push(node.ok_or_else(missing)?);
            }
            pos /= 4;
        }
        Ok(())
    }

    /// The fold blocks of `indices`, from wherever the codeword is.
    fn gather(&self, indices: &[usize]) -> Result<Vec<Vec<FieldElement<F>>>, Error> {
        let num_leaves = self.num_leaves();
        match &self.codeword {
            Codeword::Host(values) => Ok(indices
                .iter()
                .map(|index| {
                    coset_of(*index, self.log_domain_size, self.log_folding)
                        .into_iter()
                        .map(|p| values[p].clone())
                        .collect()
                })
                .collect()),
            Codeword::Device(device) => device
                .cosets(indices, num_leaves, 1usize << self.log_folding)
                .ok_or(Error::QueryOutOfRange {
                    index: 0,
                    bound: num_leaves,
                }),
        }
    }

    /// Opens every block a round asks for.
    ///
    /// One call rather than one per query: on a device the blocks are gathered
    /// in a single pass over the codeword, and a round asks for a hundred of
    /// them.
    pub fn open_many(&self, indices: &[usize]) -> Result<Vec<CosetOpening<F>>, Error> {
        self.open_many_capped(indices, 0, false)
    }

    /// [`open_many`](Self::open_many) under a Merkle cap of height
    /// `cap_height` (the owner-path encoding, `crypto::merkle_tree::cap`).
    ///
    /// Every path is cut to `depth − cap_height` siblings. When `owner` is
    /// set, this call's first opening is the tree's first opening in proof
    /// order and carries the tree's cap (`2^cap_height` nodes) after its
    /// siblings. At `cap_height = 0` this is exactly `open_many`, whatever
    /// `owner` says.
    ///
    /// On a device the cap is read from the tree the paths are gathered from,
    /// inside the same rebuild, so it costs no extra tree build.
    pub fn open_many_capped(
        &self,
        indices: &[usize],
        cap_height: usize,
        owner: bool,
    ) -> Result<Vec<CosetOpening<F>>, Error> {
        let num_leaves = self.num_leaves();
        // ★ ONE CALL, ONE DEVICE TREE REBUILD. Counted rather than inferred:
        // `whir_round::prove` opens the current commitment AND its successor,
        // so a non-final round passes here twice, and `check_closure`'s arm F
        // asserts the total against the rounds that produced it.
        crate::whir_split::bump(&crate::whir_split::REBUILD_CALLS);
        // ⛔ THE SPLIT IS HERE AND NOT INSIDE `paths()`. On the device arm
        // `paths()` is a range check, ONE device call and a `map` into `Proof`,
        // so splitting inside it would weigh the rebuild against host
        // bookkeeping and read ~100% every time. What competes with the rebuild
        // is the COSET GATHER, which is the next statement, not a nested one.
        let __wq_tree = crate::whir_split::mark();
        let proofs = self.paths_capped(indices, cap_height, owner)?;
        crate::whir_split::add(&crate::whir_split::TREE_REBUILD, __wq_tree);

        let block = 1usize << self.log_folding;
        let __wq_gather = crate::whir_split::mark();
        let blocks: Vec<Vec<FieldElement<F>>> = match &self.codeword {
            Codeword::Host(values) => indices
                .iter()
                .map(|index| {
                    coset_of(*index, self.log_domain_size, self.log_folding)
                        .into_iter()
                        .map(|p| values[p].clone())
                        .collect()
                })
                .collect(),
            Codeword::Device(device) => {
                device
                    .cosets(indices, num_leaves, block)
                    .ok_or(Error::QueryOutOfRange {
                        index: 0,
                        bound: num_leaves,
                    })?
            }
        };

        crate::whir_split::add(&crate::whir_split::COSET_GATHER, __wq_gather);

        let __wq_assemble = crate::whir_split::mark();
        let openings: Vec<CosetOpening<F>> = blocks
            .into_iter()
            .zip(proofs)
            .map(|(values, proof)| CosetOpening { values, proof })
            .collect();
        crate::whir_split::add(&crate::whir_split::OPEN_ASSEMBLE, __wq_assemble);
        Ok(openings)
    }

    /// One authentication path per index, from wherever the tree is.
    ///
    /// A codeword the device kept has no tree here — only its root. The device
    /// keeps the tree past the commit, inside the codeword's promise and
    /// evictable (`math_cuda::whir`: whole by default, or its leaf layer under
    /// `LFM_WHIR_WHOLE_TREES=0`), and rebuilds what an eviction took. Bringing
    /// it home would cost ten times the rehash, because a pageable copy of half
    /// a gigabyte is the slowest thing in the commit. What a proof wants of a
    /// tree is a kilobyte per query.
    fn paths(&self, indices: &[usize]) -> Result<Vec<Proof<Commitment>>, Error> {
        if let Some(top) = &self.top {
            return self.top_paths(top, indices);
        }
        let num_leaves = self.num_leaves();
        let out_of_range = |index: usize| Error::QueryOutOfRange {
            index,
            bound: num_leaves,
        };
        match &self.codeword {
            Codeword::Device(device) => {
                if let Some(&bad) = indices.iter().find(|index| **index >= num_leaves) {
                    return Err(out_of_range(bad));
                }
                // Past the range check there is one way to fail, and it is the
                // device: the paths are read there — from the kept tree, or a
                // tree rebuilt — because that is where the codeword is.
                Ok(device
                    .paths(self.log_folding, indices, H::DEVICE)
                    .ok_or(Error::DeviceFailed {
                        stage: "opening paths",
                    })?
                    .into_iter()
                    .map(|merkle_path| Proof { merkle_path })
                    .collect())
            }
            Codeword::Host(_) => indices
                .iter()
                .map(|index| {
                    self.tree
                        .get_proof_by_pos(*index)
                        .ok_or_else(|| out_of_range(*index))
                })
                .collect(),
        }
    }

    /// [`paths`](Self::paths), cut to the cap and, for the owner, with the
    /// cap appended to the first path.
    fn paths_capped(
        &self,
        indices: &[usize],
        cap_height: usize,
        owner: bool,
    ) -> Result<Vec<Proof<Commitment>>, Error> {
        if cap_height == 0 {
            return self.paths(indices);
        }
        let depth = self.depth();
        let arity = Self::arity();
        let levels = self.levels();
        if cap_height > levels {
            return Err(Error::CapEmbedFailed {
                reason: "cap taller than the tree",
            });
        }
        let embed_failed = |_: crypto::merkle_tree::cap::CapError| Error::CapEmbedFailed {
            reason: "path or cap of the wrong length",
        };
        let (mut proofs, cap) = if owner && let Some(top) = &self.top {
            // The cap is kept whole when it sits in the kept levels.
            if cap_height > levels - top.dropped {
                return Err(Error::CapEmbedFailed {
                    reason: "cap below the kept top of a retired tree",
                });
            }
            let range = if arity == 4 {
                // The real nodes of the level `cap_height` below the root.
                let sizes = level_sizes4(self.num_leaves());
                let level = levels - cap_height;
                let start = level_offsets4(&sizes)[level];
                start..start + sizes[level]
            } else {
                let start = (1usize << cap_height) - 1;
                start..2 * start + 1
            };
            let cap = top
                .nodes
                .get(range)
                .ok_or(Error::CapEmbedFailed {
                    reason: "cap below the kept top of a retired tree",
                })?
                .to_vec();
            (self.top_paths(top, indices)?, Some(cap))
        } else if owner {
            match &self.codeword {
                Codeword::Device(device) => {
                    let num_leaves = self.num_leaves();
                    if let Some(&bad) = indices.iter().find(|index| **index >= num_leaves) {
                        return Err(Error::QueryOutOfRange {
                            index: bad,
                            bound: num_leaves,
                        });
                    }
                    // ONE rebuild: the cap comes from the tree the paths are
                    // gathered from.
                    let (paths, cap) = device
                        .paths_and_cap(self.log_folding, indices, cap_height, H::DEVICE)
                        .ok_or(Error::DeviceFailed {
                            stage: "opening paths and cap",
                        })?;
                    let proofs: Vec<Proof<Commitment>> = paths
                        .into_iter()
                        .map(|merkle_path| Proof { merkle_path })
                        .collect();
                    (proofs, Some(cap))
                }
                Codeword::Host(_) => {
                    let cap = self.tree.cap(cap_height).ok_or(Error::CapEmbedFailed {
                        reason: "the host tree has no cap at this height",
                    })?;
                    (self.paths(indices)?, Some(cap))
                }
            }
        } else {
            (self.paths(indices)?, None)
        };
        match cap {
            Some(cap) => {
                let mut refs: Vec<&mut Vec<Commitment>> =
                    proofs.iter_mut().map(|p| &mut p.merkle_path).collect();
                embed_cap_arity(&mut refs, depth, &cap, arity).map_err(embed_failed)?;
            }
            None if arity == 4 => {
                // Every path whole (three siblings a level), then cut to the
                // levels under the cap.
                let full = self.path_len();
                let keep = 3 * (levels - cap_height);
                for proof in &mut proofs {
                    if proof.merkle_path.len() != full {
                        return Err(Error::CapEmbedFailed {
                            reason: "path or cap of the wrong length",
                        });
                    }
                    proof.merkle_path.truncate(keep);
                }
            }
            None => {
                for proof in &mut proofs {
                    proof
                        .truncate_to_cap(depth, cap_height)
                        .map_err(embed_failed)?;
                }
            }
        }
        Ok(proofs)
    }

    /// Opens the block that folds onto `index`.
    pub fn open(&self, index: usize) -> Result<CosetOpening<F>, Error> {
        let num_leaves = self.num_leaves();
        let proof = self.paths(&[index])?.pop().ok_or(Error::QueryOutOfRange {
            index,
            bound: num_leaves,
        })?;
        let values = match &self.codeword {
            Codeword::Host(values) => coset_of(index, self.log_domain_size, self.log_folding)
                .into_iter()
                .map(|p| values[p].clone())
                .collect(),
            // A query opens `2^log_folding` values of an array that is not
            // here: that is a gather, not a codeword coming back.
            Codeword::Device(device) => device
                .cosets(&[index], num_leaves, 1usize << self.log_folding)
                .and_then(|blocks| blocks.into_iter().next())
                .ok_or(Error::QueryOutOfRange {
                    index,
                    bound: num_leaves,
                })?,
        };
        Ok(CosetOpening { values, proof })
    }
}

/// Checks an opening against a root, under `H`'s hash.
///
/// `H` is explicit at every call site rather than defaulted, because a free
/// function's type parameter cannot carry a default and — more to the point —
/// because "which hash authenticated this path" is the whole content of the
/// call. A verifier reading a proof under the wrong `H` gets `false` here, not
/// a different-but-plausible answer.
///
/// `depth` is the tree's depth (`log2` of its leaf count), a verifier
/// constant: the path must be exactly that long and `index < 2^depth`. A path
/// of any other length is refused before it is folded, so a leaf hash can
/// never be compared with an internal node.
pub fn verify_opening<F, H>(
    root: &Commitment,
    depth: usize,
    index: usize,
    opening: &CosetOpening<F>,
) -> bool
where
    F: IsField + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    verify_opening_capped::<F, H>(
        &CappedRoot::uncapped(root, depth),
        index,
        opening,
        &opening.proof.merkle_path,
    )
}

/// Checks an opening against one tree's authenticated cap.
///
/// `siblings` is the opening's path with any cap split off — the whole
/// `opening.proof.merkle_path` for every opening but a tree's owner, whose
/// siblings [`CappedRoot::from_owner`] returns. It must be exactly
/// `depth − c` long and fold `hash(values)` at `index` onto
/// `cap[index >> (depth − c)]`. At `c = 0` the cap is the root, and this is
/// [`verify_opening`].
pub fn verify_opening_capped<F, H>(
    check: &CappedRoot<'_, Commitment>,
    index: usize,
    opening: &CosetOpening<F>,
    siblings: &[Commitment],
) -> bool
where
    F: IsField + 'static,
    H: WhirHash,
    FieldElement<F>: AsBytes + Sync + Send,
{
    check.verify::<Backend<F, H>>(siblings, index, Backend::<F, H>::hash_data(&opening.values))
}

/// One level of a block's fold.
///
/// Within the block the pair for slot `t` is `(t, t + half)`, mirroring the
/// global layout one level up. Each pair sits at its own domain point: slot `t`
/// is at `j + t·(N/L)`, so the points are `g^j·η^t` with `η` a primitive
/// `L`-th root of unity. Using one `x` for the whole level is only correct when
/// the block holds a single pair.
fn fold_block_level<F, A, B>(
    values: &[FieldElement<A>],
    domain: &Domain<F>,
    position: usize,
    alpha: &FieldElement<B>,
) -> Vec<FieldElement<B>>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<A> + IsSubFieldOf<B>,
    A: IsField + IsSubFieldOf<B>,
    B: IsField,
{
    let two_inv = (FieldElement::<F>::one() + FieldElement::<F>::one())
        .inv()
        .expect("2 is invertible");
    let half = values.len() / 2;
    let eta = domain
        .generator()
        .pow((domain.size() / values.len()) as u64);

    let mut out = Vec::with_capacity(half);
    let mut x = domain.generator().pow(position as u64);
    for t in 0..half {
        let (a, b) = (&values[t], &values[t + half]);
        let even = &two_inv * (a + b);
        let x_inv = x.inv().expect("domain elements are nonzero");
        let odd = (&two_inv * x_inv) * (a - b);
        // The base element on the left: the only direction the tower gives.
        out.push(even + odd * alpha);
        x *= &eta;
    }
    out
}

/// Folds an opened block down to the single value it contributes.
///
/// The verifier's local mirror of [`fold_codeword_k`](crate::whir::fold_codeword_k):
/// it never sees the whole codeword, only this block, and must reach the same
/// value the prover would have.
///
/// The values go in over `C` and come out over `N`, matching the codeword fold:
/// the committed trace's blocks are base-field, and the first fold lifts them.
pub fn fold_coset<F, C, N>(
    values: &[FieldElement<C>],
    domain: &Domain<F>,
    index: usize,
    alphas: &[FieldElement<N>],
) -> Result<FieldElement<N>, Error>
where
    F: IsFFTField + IsPrimeField + IsSubFieldOf<C> + IsSubFieldOf<N>,
    C: IsField + IsSubFieldOf<N>,
    N: IsField,
{
    if values.len() != 1usize << alphas.len() {
        return Err(Error::CodewordTooShort {
            coefficients: values.len(),
            domain: 1usize << alphas.len(),
        });
    }
    let Some((first, rest)) = alphas.split_first() else {
        return Ok(values[0].clone().to_extension::<N>());
    };

    let mut current_domain = domain.clone();
    // The block's own position within each successively squared domain.
    let mut position = index;

    let mut current = fold_block_level::<F, C, N>(values, &current_domain, position, first);
    current_domain = current_domain.squared()?;
    position %= current_domain.size();

    for alpha in rest {
        current = fold_block_level::<F, N, N>(&current, &current_domain, position, alpha);
        current_domain = current_domain.squared()?;
        position %= current_domain.size();
    }

    Ok(current.into_iter().next().expect("one value remains"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField as F;

    use crate::{
        mle::Mle,
        whir::{encode, fold_codeword_k, monomial_coefficients},
        whir_hash::KeccakWhir,
    };

    type FE = FieldElement<F>;

    fn pseudo_codeword(num_vars: usize, log_blowup: usize, seed: u64) -> (Vec<FE>, Domain<F>) {
        let vals: Vec<FE> = (0..(1u64 << num_vars))
            .map(|i| FE::from((i.wrapping_mul(6364136223846793005).wrapping_add(seed)) >> 13))
            .collect();
        let f = Mle::new(vals).unwrap();
        let domain = Domain::<F>::new(num_vars + log_blowup).unwrap();
        let cw = encode(&monomial_coefficients(&f), &domain).unwrap();
        (cw, domain)
    }

    /// A retired tree drops its codeword's own depth of levels — the card's
    /// or the host's — and never into the tallest cap it may be asked for:
    /// a shallow tree keeps at least its cap.
    #[test]
    fn the_drop_follows_the_codeword_and_stops_at_the_cap() {
        let drop = TreeDrop { device: 8, host: 4 };
        assert_eq!(drop.levels(true, 23, 3), 8);
        assert_eq!(drop.levels(false, 23, 3), 4);
        assert_eq!(drop.levels(true, 9, 3), 6, "clamped to depth − cap");
        assert_eq!(drop.levels(false, 5, 3), 2);
        assert_eq!(drop.levels(true, 2, 3), 0, "a tree no deeper than its cap");
        assert_eq!(TreeDrop::uniform(5), TreeDrop { device: 5, host: 5 });
        assert_eq!(drop.to_string(), "card 8 · host 4");
    }

    /// The clamp on a real retire: a host tree 6 deep under the tallest auto
    /// cap (3) asked to drop 8 drops 3, and its paths are the whole tree's.
    #[test]
    fn a_shallow_tree_drops_down_to_its_cap_and_keeps_its_paths() {
        let (cw, _) = pseudo_codeword(6, 2, 11);
        let kept = CodewordCommitment::<F, KeccakWhir>::new(&cw, 2).unwrap();
        assert_eq!(kept.depth(), 6);
        let indices = [0usize, 5, 17, 63];
        let want: Vec<_> = indices.iter().map(|&i| kept.paths(&[i]).unwrap()).collect();
        let cap = crypto::merkle_tree::cap::CapPolicy::Auto;
        let retired = CodewordCommitment::<F, KeccakWhir>::new(&cw, 2)
            .unwrap()
            .retire(TreeDrop::uniform(8), cap)
            .unwrap();
        assert_eq!(retired.dropped(), 6 - cap.height(usize::MAX, 6));
        assert_eq!(retired.dropped(), 3);
        let revived = retired.revive::<F, KeccakWhir>(Codeword::Host(cw)).unwrap();
        let got: Vec<_> = indices
            .iter()
            .map(|&i| revived.paths(&[i]).unwrap())
            .collect();
        assert_eq!(format!("{got:?}"), format!("{want:?}"));
    }

    /// ★ The card's re-hash of a retired tree's queried blocks (box only, a
    /// codeword big enough for the card): at 4, 6 and 8 dropped levels, under
    /// both hashes, the card's paths are the host's and the kept tree's, byte
    /// for byte; a kept node flipped is refused on the card as on the host.
    #[cfg(feature = "cuda")]
    #[test]
    fn the_card_rehashes_a_retired_trees_blocks_to_the_host_paths() {
        fn check<H: WhirHash>() {
            use crate::stacked_eval::StackedCommitment;
            use crate::stacking::StackedLayout;
            let config = crate::whir_chain::ChainConfig {
                log_blowup: 2,
                log_folding: 2,
                num_queries: 3,
                grind: crate::whir_chain::GrindBits::default(),
                format: crate::whir_chain::ChainFormat::DEFAULT,
            };
            let num_vars = 14;
            let columns: Vec<Mle<F>> = (0..3u64)
                .map(|c| {
                    Mle::new(
                        (0..1u64 << num_vars)
                            .map(|i| FE::from(i.wrapping_mul(0x9e37_79b9).wrapping_add(c) >> 7))
                            .collect(),
                    )
                    .unwrap()
                })
                .collect();
            let layout = StackedLayout::build(&[num_vars; 3], num_vars).unwrap();
            let borrowed = crate::stacking::borrow(&columns);
            let mut kept =
                StackedCommitment::<F, H>::commit(layout.clone(), &borrowed, None, &config)
                    .unwrap();
            let indices: Vec<usize> = (0..40usize)
                .map(|q| (q * 2_654_435_761) % (1 << 14))
                .collect();
            let want: Vec<Vec<Proof<Commitment>>> = kept
                .commitments_mut()
                .iter()
                .map(|c| c.paths(&indices).unwrap())
                .collect();
            for d in [4usize, 6, 8] {
                let retired =
                    StackedCommitment::<F, H>::commit(layout.clone(), &borrowed, None, &config)
                        .unwrap()
                        .retire(TreeDrop::uniform(d), &config)
                        .unwrap();
                let mut revived = retired.revive::<H, _>(&borrowed, None, &config).unwrap();
                for (c, want) in revived.commitments_mut().iter_mut().zip(&want) {
                    assert!(
                        matches!(c.codeword, Codeword::Device(_)),
                        "the codeword is on the card"
                    );
                    let top = c.top.clone().expect("a kept top");
                    assert_eq!(top.dropped, d);
                    let card = c
                        .card_top_paths(&top, &indices)
                        .expect("the card re-hashes")
                        .unwrap();
                    let host = c.host_top_paths(&top, &indices).unwrap();
                    assert_eq!(
                        format!("{card:?}"),
                        format!("{host:?}"),
                        "d {d}: card vs host"
                    );
                    assert_eq!(
                        format!("{card:?}"),
                        format!("{want:?}"),
                        "d {d}: card vs kept"
                    );
                    // A kept node under a queried block flipped: refused on both.
                    let frontier = (1usize << (c.depth() - d)) - 1;
                    let mut bad = top.clone();
                    bad.nodes[frontier + (indices[0] >> d)][0] ^= 1;
                    assert!(matches!(
                        c.card_top_paths(&bad, &indices),
                        Some(Err(Error::RecomputedCodewordMismatch { .. }))
                    ));
                    assert!(matches!(
                        c.host_top_paths(&bad, &indices),
                        Err(Error::RecomputedCodewordMismatch { .. })
                    ));
                }
            }
        }
        check::<KeccakWhir>();
        check::<crate::whir_hash::RpxWhir>();
    }

    #[test]
    fn a_coset_is_a_stride_of_the_domain() {
        // 3 folds over a 32-point domain: stride 4, four entries.
        assert_eq!(coset_of(0, 5, 2), vec![0, 8, 16, 24]);
        assert_eq!(coset_of(3, 5, 2), vec![3, 11, 19, 27]);
        assert_eq!(coset_of(1, 4, 1), vec![1, 9]);
    }

    #[test]
    fn leaves_cover_the_codeword_exactly_once() {
        let (cw, _) = pseudo_codeword(3, 2, 1);
        let commitment = CodewordCommitment::<F, KeccakWhir>::new(&cw, 2).unwrap();
        assert_eq!(commitment.num_leaves(), cw.len() / 4);

        let mut seen = vec![0usize; cw.len()];
        for j in 0..commitment.num_leaves() {
            for p in coset_of(j, commitment.log_domain_size(), 2) {
                seen[p] += 1;
            }
        }
        assert!(seen.iter().all(|c| *c == 1), "coverage = {seen:?}");
    }

    #[test]
    fn an_opening_verifies_against_the_root() {
        let (cw, _) = pseudo_codeword(3, 2, 7);
        let commitment = CodewordCommitment::<F, KeccakWhir>::new(&cw, 1).unwrap();
        let root = commitment.root();

        for j in 0..commitment.num_leaves() {
            let opening = commitment.open(j).unwrap();
            assert_eq!(opening.values.len(), 2);
            assert!(
                verify_opening::<F, KeccakWhir>(&root, commitment.depth(), j, &opening),
                "leaf {j}"
            );
        }
    }

    #[test]
    fn a_tampered_opening_is_rejected() {
        let (cw, _) = pseudo_codeword(3, 2, 9);
        let commitment = CodewordCommitment::<F, KeccakWhir>::new(&cw, 1).unwrap();
        let root = commitment.root();

        let mut opening = commitment.open(2).unwrap();
        opening.values[0] += FE::one();
        assert!(!verify_opening::<F, KeccakWhir>(
            &root,
            commitment.depth(),
            2,
            &opening
        ));
    }

    #[test]
    fn an_opening_does_not_verify_at_another_index() {
        let (cw, _) = pseudo_codeword(3, 2, 11);
        let commitment = CodewordCommitment::<F, KeccakWhir>::new(&cw, 1).unwrap();
        let root = commitment.root();
        let opening = commitment.open(2).unwrap();
        assert!(!verify_opening::<F, KeccakWhir>(
            &root,
            commitment.depth(),
            3,
            &opening
        ));
    }

    #[test]
    fn a_query_beyond_the_leaves_is_an_error() {
        let (cw, _) = pseudo_codeword(2, 1, 3);
        let commitment = CodewordCommitment::<F, KeccakWhir>::new(&cw, 1).unwrap();
        let out = commitment.num_leaves();
        assert!(matches!(
            commitment.open(out).unwrap_err(),
            Error::QueryOutOfRange { .. }
        ));
    }

    /// What makes the query phase sound: the verifier folds only the block it
    /// was given and lands on the same value the prover computed from the whole
    /// codeword.
    #[test]
    fn folding_a_coset_matches_folding_the_whole_codeword() {
        for k in 1..=3usize {
            let (cw, domain) = pseudo_codeword(4, 2, 40 + k as u64);
            let alphas: Vec<FE> = (0..k).map(|i| FE::from(13 + i as u64)).collect();

            let (folded, _) = fold_codeword_k(&cw, &domain, &alphas).unwrap();
            let commitment = CodewordCommitment::<F, KeccakWhir>::new(&cw, k).unwrap();

            for (j, expected) in folded.iter().enumerate() {
                let opening = commitment.open(j).unwrap();
                let local = fold_coset(&opening.values, &domain, j, &alphas).unwrap();
                assert_eq!(local, *expected, "k={k}, leaf={j}");
            }
        }
    }

    #[test]
    fn folding_a_block_of_the_wrong_size_is_an_error() {
        let (_, domain) = pseudo_codeword(3, 2, 1);
        let values = vec![FE::one(); 3];
        assert!(matches!(
            fold_coset(&values, &domain, 0, &[FE::one()]).unwrap_err(),
            Error::CodewordTooShort { .. }
        ));
    }

    #[test]
    fn folding_by_zero_returns_the_single_value() {
        let (cw, domain) = pseudo_codeword(3, 2, 5);
        let commitment = CodewordCommitment::<F, KeccakWhir>::new(&cw, 0).unwrap();
        assert_eq!(commitment.num_leaves(), cw.len());
        let opening = commitment.open(6).unwrap();
        assert_eq!(fold_coset(&opening.values, &domain, 6, &[]).unwrap(), cw[6]);
    }

    #[test]
    fn a_codeword_that_is_not_a_power_of_two_is_rejected() {
        let values = vec![FE::one(); 6];
        assert!(matches!(
            CodewordCommitment::<F, KeccakWhir>::new(&values, 1).unwrap_err(),
            Error::NotPowerOfTwo(6)
        ));
    }

    #[test]
    fn leaf_and_slot_inverts_the_coset_layout() {
        let (log_domain, k) = (5usize, 2usize);
        let num_leaves = 1usize << (log_domain - k);
        for j in 0..num_leaves {
            for (slot, position) in coset_of(j, log_domain, k).into_iter().enumerate() {
                assert_eq!(leaf_and_slot(position, num_leaves), (j, slot));
            }
        }
    }

    #[test]
    fn a_position_resolves_to_the_value_it_holds() {
        let (cw, _) = pseudo_codeword(3, 2, 21);
        let commitment = CodewordCommitment::<F, KeccakWhir>::new(&cw, 2).unwrap();
        let num_leaves = commitment.num_leaves();

        for (position, value) in cw.iter().enumerate() {
            let (leaf, slot) = leaf_and_slot(position, num_leaves);
            let opening = commitment.open(leaf).unwrap();
            assert_eq!(opening.values[slot], *value, "position {position}");
        }
    }
}
