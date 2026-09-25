//! Device-vs-host parity for S2's one-row trees and openings.
//!
//! Compiled for `cuda` builds with tests or `test-utils`; every entry needs a
//! GPU, so the callers are `#[ignore]`d box tests. The stark crate instantiates
//! them under Keccak and Blake3 (`tests::zf_s2_device_tests`), the prover crate
//! under the production RPX pin (`tests::zf_rpx_device_tests`).
//!
//! Each entry builds a tree on the device at `rows_per_leaf` 1 (and, as the
//! control, 2) and pins against the host commit over the SAME evaluations:
//! - the root, and the device tree's leaf count (`lde / rows_per_leaf`);
//! - the authentication path of scattered leaves and both ends, gathered off
//!   the resident tree (`gather_proofs_dev`, the production opening path),
//!   against the host tree's;
//! - where the entry keeps an LDE handle, the device row gather at the rows a
//!   query opens (`LeafLayout::query_rows`), against the host rows;
//! - that the one-row root differs from the row-pair root (a device path that
//!   ignored the layout would equal it).
//!
//! The LDE itself is parity-pinned by the existing fused-commit tests, so the
//! host reference consumes the evaluations the device returned: this isolates
//! the leaf layout and the tree.
//!
//! Every entry returns `Err` when the device declines (threshold, budget), so a
//! host fallback is a failure, never a pass.

use std::format;
use std::string::String;
use std::sync::Arc;
use std::vec;
use std::vec::Vec;

use crypto::merkle_tree::merkle::MerkleTree;
use crypto::merkle_tree::traits::IsMerkleTreeBackend;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;

use crate::config::{Commitment, StarkHash};
use crate::fri::vectors::splitmix64;
use crate::leaf_layout::LeafLayout;
use crate::prover::{GenericProver, IsStarkProver};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;
type Ext = FieldElement<E>;
type P<H> = GenericProver<F, E, (), H>;

const LAYOUTS: [LeafLayout; 2] = [LeafLayout::Row, LeafLayout::RowPair];

fn base_values(count: usize, seed: &mut u64) -> Vec<Felt> {
    (0..count).map(|_| Felt::from(splitmix64(seed))).collect()
}

fn ext_values(count: usize, seed: &mut u64) -> Vec<Ext> {
    (0..count)
        .map(|_| {
            Ext::new([
                Felt::from(splitmix64(seed)),
                Felt::from(splitmix64(seed)),
                Felt::from(splitmix64(seed)),
            ])
        })
        .collect()
}

/// Leaves to open: both ends, their neighbours and a spread of random ones.
fn open_positions(num_leaves: usize, seed: &mut u64) -> Vec<usize> {
    let mut p = vec![0, 1, num_leaves / 2, num_leaves - 2, num_leaves - 1];
    p.extend((0..16).map(|_| (splitmix64(seed) % num_leaves as u64) as usize));
    p
}

/// The resident device tree against the host tree over the same leaves: the
/// root, the leaf count, and every opened path gathered on device.
fn check_tree<B>(
    what: &str,
    dev: &math_cuda::lde::GpuMerkleTree,
    host: &MerkleTree<B>,
    host_root: &Commitment,
    num_leaves: usize,
    seed: &mut u64,
) -> Result<(), String>
where
    B: IsMerkleTreeBackend<Node = Commitment>,
{
    if dev.root != *host_root {
        return Err(format!("{what}: device root differs from the host root"));
    }
    if dev.leaves_len != num_leaves {
        return Err(format!(
            "{what}: device tree has {} leaves, the layout needs {num_leaves}",
            dev.leaves_len
        ));
    }
    let stream = math_cuda::device::backend()
        .map_err(|e| format!("{what}: no cuda backend: {e:?}"))?
        .next_stream();
    let positions = open_positions(num_leaves, seed);
    let proofs = crate::gpu_lde::gather_proofs_dev(dev, &positions, &stream)
        .ok_or_else(|| format!("{what}: the device path gather failed"))?;
    for (pos, proof) in positions.iter().zip(&proofs) {
        let want = host
            .get_proof_by_pos(*pos)
            .ok_or_else(|| format!("{what}: host tree has no leaf {pos}"))?;
        if proof.merkle_path != want.merkle_path {
            return Err(format!("{what}: the path of leaf {pos} differs"));
        }
    }
    Ok(())
}

/// Leaf count of a tree over `lde` rows under `layout`.
fn leaves_of(lde: usize, layout: LeafLayout) -> usize {
    lde / layout.rows_per_leaf()
}

/// The fused main commit (`try_expand_leaf_and_tree_row_major_keep`, the R1
/// main arm and the LFM artifact commit) over a random `n × m` base trace at
/// `blowup`, at one row and row pairs: tree parity, plus the device row gather
/// at the one-row query rows (the R4 main opening values).
pub fn main_tree_parity<H: StarkHash>(
    n: usize,
    m: usize,
    blowup: usize,
    seed: u64,
) -> Result<String, String> {
    let mut rng = seed;
    let data = base_values(n * m, &mut rng);
    let weights = base_values(n, &mut rng);
    let lde_len = n * blowup;
    let mut roots = Vec::new();
    for layout in LAYOUTS {
        let what = format!("main {n}x{m} blowup {blowup} {layout:?}");
        let (tree, handle, lde) =
            crate::gpu_lde::try_expand_leaf_and_tree_row_major_keep::<F, F, H::Batched<F>>(
                "s2_device_parity",
                "S2 main parity",
                &data,
                None,
                n,
                m,
                blowup,
                &weights,
                true,
                layout.rows_per_leaf(),
            )
            .ok_or_else(|| format!("{what}: the device commit declined"))?;
        let (host, host_root) =
            P::<H>::commit_rows_bit_reversed_with(&lde, m, layout.rows_per_leaf())
                .ok_or_else(|| format!("{what}: host commit failed"))?;
        if tree.root != host_root {
            return Err(format!("{what}: returned root-only tree differs"));
        }
        let dev = handle
            .tree
            .as_ref()
            .ok_or_else(|| format!("{what}: no resident tree"))?;
        check_tree(
            &what,
            dev,
            &host,
            &host_root,
            leaves_of(lde_len, layout),
            &mut rng,
        )?;
        // The R4 opening values: the rows a query opens, off the resident LDE.
        let queries = open_positions(leaves_of(lde_len, layout), &mut rng);
        let rows: Vec<u32> = queries
            .iter()
            .flat_map(|&q| {
                let (row, sym) = layout.query_rows(q, lde_len);
                core::iter::once(row as u32).chain(sym.map(|r| r as u32))
            })
            .collect();
        let stream = math_cuda::device::backend()
            .map_err(|e| format!("{what}: {e:?}"))?
            .next_stream();
        let got = math_cuda::barycentric::gather_rows_base_on_device(&handle, &rows, &stream)
            .map_err(|e| format!("{what}: device row gather failed: {e:?}"))?;
        for (i, &r) in rows.iter().enumerate() {
            let want: Vec<u64> = lde[r as usize * m..(r as usize + 1) * m]
                .iter()
                .map(|x| x.canonical())
                .collect();
            let have: Vec<u64> = got[i * m..(i + 1) * m]
                .iter()
                .map(|&x| Felt::from(x).canonical())
                .collect();
            if have != want {
                return Err(format!("{what}: device row gather differs at LDE row {r}"));
            }
        }
        roots.push(host_root);
    }
    if roots[0] == roots[1] {
        return Err(format!(
            "main {n}x{m}: the one-row root equals the row-pair root"
        ));
    }
    Ok(format!(
        "main {n}x{m} blowup {blowup}: one-row and row-pair trees equal the host, \
         one-row tree {lde_len} leaves"
    ))
}

/// The preprocessed split commit (`try_expand_split_trees_row_major_keep`):
/// the precomputed tree (full host tree) and the multiplicity tree (resident)
/// at one row and row pairs, against the host subset commits.
pub fn split_tree_parity<H: StarkHash>(
    n: usize,
    m: usize,
    split: usize,
    blowup: usize,
    seed: u64,
) -> Result<String, String> {
    let mut rng = seed;
    let data = base_values(n * m, &mut rng);
    let weights = base_values(n, &mut rng);
    let lde_len = n * blowup;
    let mut roots = Vec::new();
    for layout in LAYOUTS {
        let rpl = layout.rows_per_leaf();
        let what = format!("split {n}x{m} at {split} blowup {blowup} {layout:?}");
        let (pre, mult, handle, lde) =
            crate::gpu_lde::try_expand_split_trees_row_major_keep::<F, F, H::Batched<F>>(
                "s2_device_parity",
                &data,
                None,
                n,
                m,
                blowup,
                &weights,
                split,
                true,
                true,
                rpl,
            )
            .ok_or_else(|| format!("{what}: the device commit declined"))?;
        let pre = pre.ok_or_else(|| format!("{what}: no precomputed tree"))?;
        let (host_pre, host_pre_root) =
            P::<H>::commit_rows_bit_reversed_subset_with(&lde, m, 0, split, rpl)
                .ok_or_else(|| format!("{what}: host precomputed commit failed"))?;
        let (host_mult, host_mult_root) =
            P::<H>::commit_rows_bit_reversed_subset_with(&lde, m, split, m, rpl)
                .ok_or_else(|| format!("{what}: host multiplicity commit failed"))?;
        if pre.root != host_pre_root {
            return Err(format!("{what}: precomputed root differs"));
        }
        let num_leaves = leaves_of(lde_len, layout);
        for pos in open_positions(num_leaves, &mut rng) {
            if pre.get_proof_by_pos(pos).map(|p| p.merkle_path)
                != host_pre.get_proof_by_pos(pos).map(|p| p.merkle_path)
            {
                return Err(format!("{what}: precomputed path of leaf {pos} differs"));
            }
        }
        if mult.root != host_mult_root {
            return Err(format!("{what}: multiplicity root differs"));
        }
        let dev = handle
            .tree
            .as_ref()
            .ok_or_else(|| format!("{what}: no resident tree"))?;
        check_tree(
            &what,
            dev,
            &host_mult,
            &host_mult_root,
            num_leaves,
            &mut rng,
        )?;
        roots.push(host_pre_root);
    }
    if roots[0] == roots[1] {
        return Err(format!(
            "split {n}x{m}: the one-row root equals the row-pair root"
        ));
    }
    Ok(format!(
        "split {n}x{m} at {split} blowup {blowup}: both subset trees equal the host at both layouts"
    ))
}

/// The aux commits: the fused ext3 commit from a host trace
/// (`try_expand_leaf_and_tree_ext3_row_major_keep`) and from a resident aux
/// trace (`..._keep_dev`, the LogUp aux path), at one row and row pairs; the
/// two must agree with each other and with the host commit, and the device
/// ext3 row gather must return the rows a query opens.
pub fn aux_tree_parity<H: StarkHash>(
    n: usize,
    m: usize,
    blowup: usize,
    seed: u64,
) -> Result<String, String> {
    let mut rng = seed;
    let data = ext_values(n * m, &mut rng);
    let weights = base_values(n, &mut rng);
    let lde_len = n * blowup;
    let raw: Vec<u64> = data
        .iter()
        .flat_map(|x| x.value().iter().map(|c| c.canonical()).collect::<Vec<_>>())
        .collect();
    let mut roots = Vec::new();
    for layout in LAYOUTS {
        let rpl = layout.rows_per_leaf();
        let what = format!("aux {n}x{m} blowup {blowup} {layout:?}");
        let (tree, handle, lde) = crate::gpu_lde::try_expand_leaf_and_tree_ext3_row_major_keep::<
            F,
            E,
            H::Batched<E>,
        >(
            "s2_device_parity", &data, n, m, blowup, &weights, true, rpl
        )
        .ok_or_else(|| format!("{what}: the device commit declined"))?;
        let (host, host_root) = P::<H>::commit_rows_bit_reversed_with(&lde, m, rpl)
            .ok_or_else(|| format!("{what}: host commit failed"))?;
        if tree.root != host_root {
            return Err(format!("{what}: returned root-only tree differs"));
        }
        let dev = handle
            .tree
            .as_ref()
            .ok_or_else(|| format!("{what}: no resident tree"))?;
        check_tree(
            &what,
            dev,
            &host,
            &host_root,
            leaves_of(lde_len, layout),
            &mut rng,
        )?;

        // The resident arm over the same trace (uploaded as the LogUp build
        // would leave it: row-major ext3).
        let be = math_cuda::device::backend().map_err(|e| format!("{what}: {e:?}"))?;
        let stream = be.next_stream();
        let buf = stream
            .clone_htod(&raw)
            .map_err(|e| format!("{what}: upload failed: {e:?}"))?;
        stream.synchronize().map_err(|e| format!("{what}: {e:?}"))?;
        let ra = math_cuda::logup::ResidentAux {
            buf: Arc::new(buf),
            num_aux_cols: m,
            num_rows: n,
            table_contribution: [0; 3],
        };
        let (rtree, rhandle, _) =
            crate::gpu_lde::try_expand_leaf_and_tree_ext3_row_major_keep_dev::<F, E, H::Batched<E>>(
                "s2_device_parity",
                &ra,
                blowup,
                &weights,
                true,
                rpl,
            )
            .ok_or_else(|| format!("{what}: the resident aux commit declined"))?;
        if rtree.root != host_root {
            return Err(format!(
                "{what}: the resident aux root differs from the host root"
            ));
        }
        let rdev = rhandle
            .tree
            .as_ref()
            .ok_or_else(|| format!("{what}: no resident aux tree"))?;
        check_tree(
            &format!("{what} (resident)"),
            rdev,
            &host,
            &host_root,
            leaves_of(lde_len, layout),
            &mut rng,
        )?;

        let queries = open_positions(leaves_of(lde_len, layout), &mut rng);
        let rows: Vec<u32> = queries
            .iter()
            .flat_map(|&q| {
                let (row, sym) = layout.query_rows(q, lde_len);
                core::iter::once(row as u32).chain(sym.map(|r| r as u32))
            })
            .collect();
        let got = math_cuda::barycentric::gather_rows_ext3_on_device(&handle, &rows, &stream)
            .map_err(|e| format!("{what}: device ext3 row gather failed: {e:?}"))?;
        let got = crate::constraint_ir::gpu_interp::ext3_u64_to_field::<E>(&got)
            .ok_or_else(|| format!("{what}: gather is not ext3"))?;
        for (i, &r) in rows.iter().enumerate() {
            if got[i * m..(i + 1) * m] != lde[r as usize * m..(r as usize + 1) * m] {
                return Err(format!(
                    "{what}: device ext3 row gather differs at LDE row {r}"
                ));
            }
        }
        roots.push(host_root);
    }
    if roots[0] == roots[1] {
        return Err(format!(
            "aux {n}x{m}: the one-row root equals the row-pair root"
        ));
    }
    Ok(format!(
        "aux {n}x{m} blowup {blowup}: host-input and resident trees equal the host at both layouts"
    ))
}

/// The composition trees: from host part evaluations
/// (`try_build_comp_poly_tree_gpu`) and from resident part slabs
/// (`try_build_comp_poly_tree_gpu_from_dev`), at one row and row pairs,
/// against `commit_bit_reversed_with` over the parts; and the device gather of
/// the parts at a query's rows.
pub fn composition_tree_parity<H: StarkHash>(
    lde_len: usize,
    parts: usize,
    seed: u64,
) -> Result<String, String> {
    let mut rng = seed;
    let evals: Vec<Vec<Ext>> = (0..parts).map(|_| ext_values(lde_len, &mut rng)).collect();
    // The resident layout: part `c` component `k` is the slab `(c·3 + k)`.
    let mut slabs = vec![0u64; 3 * parts * lde_len];
    for (c, part) in evals.iter().enumerate() {
        for (r, x) in part.iter().enumerate() {
            for (k, comp) in x.value().iter().enumerate() {
                slabs[(c * 3 + k) * lde_len + r] = comp.canonical();
            }
        }
    }
    let be = math_cuda::device::backend().map_err(|e| format!("composition: {e:?}"))?;
    let stream = be.next_stream();
    let buf = stream
        .clone_htod(&slabs)
        .map_err(|e| format!("composition: upload failed: {e:?}"))?;
    stream
        .synchronize()
        .map_err(|e| format!("composition: {e:?}"))?;
    let handle = math_cuda::lde::GpuLdeExt3 {
        buf: Arc::new(buf),
        m: parts,
        lde_size: lde_len,
        tree: None,
        ready: None,
    };
    let mut roots = Vec::new();
    for layout in LAYOUTS {
        let rpl = layout.rows_per_leaf();
        let what = format!("composition lde {lde_len} parts {parts} {layout:?}");
        let (host, host_root) =
            crate::commitment::commit_bit_reversed_with::<E, H::Batched<E>>(&evals, rpl)
                .ok_or_else(|| format!("{what}: host commit failed"))?;
        let (tree, dev) =
            crate::gpu_lde::try_build_comp_poly_tree_gpu::<E, H::Batched<E>>(&evals, rpl)
                .ok_or_else(|| format!("{what}: the device tree (host parts) declined"))?;
        if tree.root != host_root {
            return Err(format!("{what}: returned root-only tree differs"));
        }
        check_tree(
            &what,
            &dev,
            &host,
            &host_root,
            leaves_of(lde_len, layout),
            &mut rng,
        )?;
        let (rtree, rdev) =
            crate::gpu_lde::try_build_comp_poly_tree_gpu_from_dev::<E, H::Batched<E>>(&handle, rpl)
                .ok_or_else(|| format!("{what}: the device tree (resident parts) declined"))?;
        if rtree.root != host_root {
            return Err(format!("{what}: the resident-parts root differs"));
        }
        check_tree(
            &format!("{what} (resident parts)"),
            &rdev,
            &host,
            &host_root,
            leaves_of(lde_len, layout),
            &mut rng,
        )?;
        // The R4 composition opening values off the resident parts.
        let queries = open_positions(leaves_of(lde_len, layout), &mut rng);
        let rows: Vec<u32> = queries
            .iter()
            .flat_map(|&q| {
                let (row, sym) = layout.query_rows(q, lde_len);
                core::iter::once(row as u32).chain(sym.map(|r| r as u32))
            })
            .collect();
        let got = math_cuda::barycentric::gather_rows_ext3_on_device(&handle, &rows, &stream)
            .map_err(|e| format!("{what}: device parts gather failed: {e:?}"))?;
        let got = crate::constraint_ir::gpu_interp::ext3_u64_to_field::<E>(&got)
            .ok_or_else(|| format!("{what}: gather is not ext3"))?;
        for (i, &r) in rows.iter().enumerate() {
            let want: Vec<Ext> = evals.iter().map(|p| p[r as usize]).collect();
            if got[i * parts..(i + 1) * parts] != want[..] {
                return Err(format!(
                    "{what}: device parts gather differs at LDE row {r}"
                ));
            }
        }
        roots.push(host_root);
    }
    if roots[0] == roots[1] {
        return Err(format!(
            "composition lde {lde_len}: the one-row root equals the row-pair root"
        ));
    }
    Ok(format!(
        "composition lde {lde_len} parts {parts}: host-parts and resident-parts trees equal the host at both layouts"
    ))
}

/// The (e) leaf-digest KAT (`fri::vectors::one_row_leaf_digests_json`): the
/// same 16-row base (5 columns) and ext3 (2 columns) matrices, hashed by the
/// device row-major leaf kernels at one row and row pairs, against the CPU
/// leaves the checked-in `e_leaf_digests_{hash}.json` was generated from.
pub fn leaf_digest_parity<H: StarkHash>() -> Result<String, String> {
    const ROWS: usize = 16;
    let mut st = crate::fri::vectors::KAT_SEED + 100;
    let base: Vec<Vec<Felt>> = (0..5)
        .map(|_| (0..ROWS).map(|_| Felt::from(splitmix64(&mut st))).collect())
        .collect();
    let mut st = crate::fri::vectors::KAT_SEED + 200;
    let ext: Vec<Vec<Ext>> = (0..2)
        .map(|_| {
            (0..ROWS)
                .map(|_| crate::fri::vectors::next_ext(&mut st))
                .collect()
        })
        .collect();
    // Row-major u64 views (an ext3 element = three consecutive u64).
    let base_rm: Vec<u64> = (0..ROWS)
        .flat_map(|r| base.iter().map(move |c| c[r].canonical()))
        .collect();
    let ext_rm: Vec<u64> = (0..ROWS)
        .flat_map(|r| {
            ext.iter().flat_map(move |c| {
                c[r].value()
                    .iter()
                    .map(|x| x.canonical())
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let hash = crate::gpu_lde::device_hash_of::<H::Batched<F>>();
    for layout in LAYOUTS {
        let rpl = layout.rows_per_leaf();
        let want_b = crate::commitment::leaves_bit_reversed_grouped::<F, H::Batched<F>>(&base, rpl);
        let want_e = crate::commitment::leaves_bit_reversed_grouped::<E, H::Batched<E>>(&ext, rpl);
        let got_b = math_cuda::lde::row_major_leaves(hash, &base_rm, 5, 0, 5, ROWS, rpl)
            .map_err(|e| format!("leaf KAT base {layout:?}: {e:?}"))?;
        let got_e = math_cuda::lde::row_major_leaves(hash, &ext_rm, 6, 0, 6, ROWS, rpl)
            .map_err(|e| format!("leaf KAT ext3 {layout:?}: {e:?}"))?;
        let flat = |v: &[Commitment]| v.iter().flatten().copied().collect::<Vec<u8>>();
        if got_b != flat(&want_b) {
            return Err(format!(
                "leaf KAT base {layout:?}: device leaves differ from the CPU"
            ));
        }
        if got_e != flat(&want_e) {
            return Err(format!(
                "leaf KAT ext3 {layout:?}: device leaves differ from the CPU"
            ));
        }
    }
    Ok(String::from(
        "the (e) leaf-digest KAT: device leaves equal the CPU at both layouts",
    ))
}

/// Every tree entry at the shapes the box runs: narrow and wide, blowup 2 and
/// 4, the LDE floor (2^14) and a production-sized 2^20 LDE. Prints one
/// `S2DEV` line per case and a summary line; `Err` lists every failure.
pub fn run_tree_parity<H: StarkHash>(name: &str) -> Result<usize, Vec<String>> {
    let mut results: Vec<Result<String, String>> = vec![leaf_digest_parity::<H>()];
    for (i, &(n, m, blowup)) in [
        (1usize << 12, 1usize, 4usize),
        (1 << 13, 20, 2),
        (1 << 12, 134, 4),
        (1 << 18, 7, 4),
    ]
    .iter()
    .enumerate()
    {
        results.push(main_tree_parity::<H>(n, m, blowup, 0x5230_0000 + i as u64));
    }
    for (i, &(n, m, split, blowup)) in [(1usize << 12, 5usize, 2usize, 4usize), (1 << 18, 9, 4, 4)]
        .iter()
        .enumerate()
    {
        results.push(split_tree_parity::<H>(
            n,
            m,
            split,
            blowup,
            0x5231_0000 + i as u64,
        ));
    }
    for (i, &(n, m, blowup)) in [
        (1usize << 12, 1usize, 4usize),
        (1 << 13, 13, 2),
        (1 << 18, 5, 4),
    ]
    .iter()
    .enumerate()
    {
        results.push(aux_tree_parity::<H>(n, m, blowup, 0x5232_0000 + i as u64));
    }
    for (i, &(lde, parts)) in [(1usize << 14, 1usize), (1 << 14, 2), (1 << 20, 2)]
        .iter()
        .enumerate()
    {
        results.push(composition_tree_parity::<H>(
            lde,
            parts,
            0x5233_0000 + i as u64,
        ));
    }
    let total = results.len();
    let mut failures = Vec::new();
    for r in results {
        match r {
            Ok(msg) => std::println!("S2DEV {name} {msg}"),
            Err(e) => failures.push(e),
        }
    }
    if failures.is_empty() {
        std::println!("S2DEV {name}: {total} tree cases equal");
        Ok(total)
    } else {
        Err(failures)
    }
}
