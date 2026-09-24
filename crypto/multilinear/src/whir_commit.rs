//! Committing a codeword by fold blocks, so each query is one Merkle opening.
//!
//! The pre-image of folded index `j` is the stride-`N/2^k` coset
//! `{ j, j + N/2^k, …, j + (2^k - 1)·N/2^k }`.

use crypto::merkle_tree::{
    cap::{CappedRoot, embed_cap},
    merkle::MerkleTree,
    proof::Proof,
    traits::IsMerkleTreeBackend,
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
        })
    }

    pub fn root(&self) -> Commitment {
        self.tree.root
    }

    pub fn num_leaves(&self) -> usize {
        1usize << (self.log_domain_size - self.log_folding)
    }

    /// Siblings on a full authentication path: `log2(num_leaves)`.
    pub fn depth(&self) -> usize {
        self.log_domain_size - self.log_folding
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
    /// A codeword the device kept has no tree here — only its root. Rebuilding
    /// it there costs a keccak pass; keeping it would cost half a gigabyte of
    /// device memory per commitment for the whole proof, and bringing it home
    /// costs ten times the rehash, because a pageable copy of half a gigabyte
    /// is the slowest thing in the commit. What a proof wants of a tree is a
    /// kilobyte per query.
    fn paths(&self, indices: &[usize]) -> Result<Vec<Proof<Commitment>>, Error> {
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
                // device: the tree has to be rebuilt there because that is
                // where the codeword is.
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
        if cap_height > depth {
            return Err(Error::CapEmbedFailed {
                reason: "cap taller than the tree",
            });
        }
        let embed_failed = |_: crypto::merkle_tree::cap::CapError| Error::CapEmbedFailed {
            reason: "path or cap of the wrong length",
        };
        let (mut proofs, cap) = if owner {
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
                embed_cap(&mut refs, depth, &cap).map_err(embed_failed)?;
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
/// never be compared with an internal node (design/CAP.md §9.4).
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
