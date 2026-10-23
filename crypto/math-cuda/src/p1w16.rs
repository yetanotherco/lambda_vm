//! ⛔ D-HASH stage 1 (diagnostic; on NO proving path): the Poseidon1
//! width-16 measurement kernels (`kernels/p1w16.cu`), their parity entry
//! points, and the microbenchmark against the RPX kernels.
//!
//! The module is loaded on first use from its own cubin, never by
//! [`crate::device::Backend::init`], so a proving process never loads it.
//! Digests are four canonical `u64`s per node (a measurement format).

use std::sync::OnceLock;
use std::time::Instant;

use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use crate::Result;
use crate::device::backend;

const P1W16_CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/p1w16.cubin"));

/// Threads per block, as RPX's kernels.
const BLOCK_DIM: u32 = 128;

/// Felts per permutation state, rate, digest.
pub const WIDTH: usize = 16;
pub const RATE: usize = 12;
pub const DIGEST: usize = 4;

/// The kernel variants the cubin instantiates, in `p1w16.cu`'s order: the two
/// multiply variants and the Fourier-domain partial rounds of the circulant
/// instance, then `c1`, the Grain Cauchy MDS alternative.
pub const VARIANTS: [&str; 4] = ["v0", "v1", "v2", "c1"];

/// Is variant `v` the Cauchy alternative (a different permutation, checked
/// against `permute_cauchy` on the host)?
pub fn is_cauchy(v: usize) -> bool {
    VARIANTS[v].starts_with('c')
}

/// One variant's kernels.
pub struct Kernels {
    pub permute_probe: CudaFunction,
    pub leaves_base_coset: CudaFunction,
    pub leaves_ext3_coset: CudaFunction,
    pub merkle_level4: CudaFunction,
    pub grind_search: CudaFunction,
}

/// ZisK's instance (stage P1 of the Poseidon1 base STARK): its leaf hash over
/// the coset and LDE-row geometries (Fourier-domain permutation), and the
/// width-8 permutation and grind at [`W8_VARIANTS`].
pub struct Zisk {
    pub leaves_base_coset: CudaFunction,
    pub leaves_ext3_coset: CudaFunction,
    pub leaves_rows: CudaFunction,
    pub leaves_row_pair: CudaFunction,
    pub w8_permute_probe: Vec<CudaFunction>,
    pub w8_grind_search: Vec<CudaFunction>,
}

/// The width-8 variants: textbook rounds, then the Fourier-domain partial rounds.
pub const W8_VARIANTS: [&str; 2] = ["v1", "v2"];

/// The loaded module: every variant, ZisK's instance and the bench fill.
pub struct Module {
    _module: std::sync::Arc<CudaModule>,
    pub variants: Vec<Kernels>,
    pub zisk: Zisk,
    pub fill: CudaFunction,
}

/// Load the module once.
pub fn module() -> Result<&'static Module> {
    static MODULE: OnceLock<Module> = OnceLock::new();
    if let Some(m) = MODULE.get() {
        return Ok(m);
    }
    let be = backend()?;
    let module = be.ctx.load_module(Ptx::from_binary(P1W16_CUBIN.to_vec()))?;
    let kernels = |v: &str| -> Result<Kernels> {
        Ok(Kernels {
            permute_probe: module.load_function(&format!("p1w16_permute_probe_{v}"))?,
            leaves_base_coset: module.load_function(&format!("p1w16_leaves_base_coset_{v}"))?,
            leaves_ext3_coset: module.load_function(&format!("p1w16_leaves_ext3_coset_{v}"))?,
            merkle_level4: module.load_function(&format!("p1w16_merkle_level4_{v}"))?,
            grind_search: module.load_function(&format!("p1w16_grind_search_{v}"))?,
        })
    };
    let zisk = Zisk {
        leaves_base_coset: module.load_function("p1w16_zleaves_base_coset_v2")?,
        leaves_ext3_coset: module.load_function("p1w16_zleaves_ext3_coset_v2")?,
        leaves_rows: module.load_function("p1w16_zleaves_rows_v2")?,
        leaves_row_pair: module.load_function("p1w16_zleaves_row_pair_v2")?,
        w8_permute_probe: W8_VARIANTS
            .iter()
            .map(|v| module.load_function(&format!("p1w8_permute_probe_{v}")))
            .collect::<Result<_>>()?,
        w8_grind_search: W8_VARIANTS
            .iter()
            .map(|v| module.load_function(&format!("p1w8_grind_search_{v}")))
            .collect::<Result<_>>()?,
    };
    let m = Module {
        variants: VARIANTS.iter().map(|v| kernels(v)).collect::<Result<_>>()?,
        zisk,
        fill: module.load_function("p1w16_fill")?,
        _module: module,
    };
    Ok(MODULE.get_or_init(|| m))
}

fn cfg(threads: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: ((threads as u32).div_ceil(BLOCK_DIM), 1, 1),
        block_dim: (BLOCK_DIM, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// `n` permutations of `states` (`n·16` felts) on device.
pub fn permute_many(variant: usize, states: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(states.len() % WIDTH, 0);
    let n = (states.len() / WIDTH) as u64;
    let k = &module()?.variants[variant];
    let stream = backend()?.next_stream();
    let input = stream.clone_htod(states)?;
    let mut out = stream.alloc_zeros::<u64>(states.len())?;
    unsafe {
        stream
            .launch_builder(&k.permute_probe)
            .arg(&input)
            .arg(&n)
            .arg(&mut out)
            .launch(cfg(n))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// Coset leaves over a base (`ext3 = false`) or ext3 codeword: `num_leaves`
/// digests of `block` elements each, element `t` of leaf `j` at `j + t·num_leaves`.
pub fn leaves_coset(variant: usize, codeword: &[u64], block: u64, ext3: bool) -> Result<Vec<u64>> {
    let per = if ext3 { 3 } else { 1 };
    let num_leaves = codeword.len() as u64 / (block * per);
    let k = &module()?.variants[variant];
    let stream = backend()?.next_stream();
    let input = stream.clone_htod(codeword)?;
    let mut out = stream.alloc_zeros::<u64>(num_leaves as usize * DIGEST)?;
    let f = if ext3 {
        &k.leaves_ext3_coset
    } else {
        &k.leaves_base_coset
    };
    unsafe {
        stream
            .launch_builder(f)
            .arg(&input)
            .arg(&num_leaves)
            .arg(&block)
            .arg(&mut out)
            .launch(cfg(num_leaves))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// One 4-ary level: `children.len() / 16` parents.
pub fn merkle_level4(variant: usize, children: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(children.len() % WIDTH, 0);
    let n = (children.len() / WIDTH) as u64;
    let k = &module()?.variants[variant];
    let stream = backend()?.next_stream();
    let input = stream.clone_htod(children)?;
    let mut out = stream.alloc_zeros::<u64>(n as usize * DIGEST)?;
    unsafe {
        stream
            .launch_builder(&k.merkle_level4)
            .arg(&input)
            .arg(&mut out)
            .arg(&n)
            .launch(cfg(n))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// The smallest nonce in `[0, count)` whose grind head is `< limit`, if any.
pub fn grind(
    variant: usize,
    inner: &[u64; 4],
    limit: u64,
    count: u64,
    grid: u32,
) -> Result<Option<u64>> {
    let k = &module()?.variants[variant];
    let stream = backend()?.next_stream();
    let inner_dev = stream.clone_htod(inner.as_slice())?;
    let mut result = stream.clone_htod(&[u64::MAX])?;
    let base = 0u64;
    unsafe {
        stream
            .launch_builder(&k.grind_search)
            .arg(&inner_dev)
            .arg(&limit)
            .arg(&base)
            .arg(&count)
            .arg(&mut result)
            .launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let host = stream.clone_dtoh(&result)?;
    stream.synchronize()?;
    Ok((host[0] != u64::MAX).then_some(host[0]))
}

/// Median of `reps` timings of `f` (each ends in a stream synchronize), in ns.
fn median_ns(reps: u32, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    f()?; // warm-up (module load, clocks)
    let mut t = Vec::with_capacity(reps as usize);
    for _ in 0..reps {
        let start = Instant::now();
        f()?;
        t.push(start.elapsed().as_nanos() as f64);
    }
    t.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    Ok(t[t.len() / 2])
}

/// ★ The stage-1 microbenchmark: one `2^log_len`-element base codeword hashed
/// as 64-element WHIR fold cosets, then its tree, then a grind scan, under RPX
/// (the production kernels and walk) and under both Poseidon1 variants. Every
/// line is `BENCH <what> <hash> <ns> ns/perm (<perms> perms, <ms> ms)`.
pub fn microbench(log_len: u32, reps: u32, grind_count: u64) -> Result<Vec<String>> {
    let be = backend()?;
    let m = module()?;
    let stream = be.next_stream();
    let len = 1u64 << log_len;
    let block = 64u64;
    let num_leaves = len / block;
    let mut lines = Vec::new();
    let mut line = |what: &str, hash: &str, ns: f64, perms: u64| {
        let s = format!(
            "BENCH {what:<5} {hash:<9} {:>7.3} ns/perm ({perms} perms, {:.2} ms)",
            ns / perms as f64,
            ns / 1e6
        );
        println!("{s}");
        lines.push(s);
    };

    let mut codeword = stream.alloc_zeros::<u64>(len as usize)?;
    unsafe {
        stream
            .launch_builder(&m.fill)
            .arg(&mut codeword)
            .arg(&len)
            .launch(cfg(len))?;
    }
    stream.synchronize()?;

    // RPX: the production coset leaves into a tree buffer, then the production walk.
    let total_nodes = 2 * num_leaves as usize - 1;
    let mut nodes = stream.alloc_zeros::<u8>(total_nodes * 32)?;
    let leaves_off = (num_leaves as usize - 1) * 32;
    let ns = median_ns(reps, || {
        let mut leaves = nodes.slice_mut(leaves_off..leaves_off + num_leaves as usize * 32);
        unsafe {
            stream
                .launch_builder(&be.rpx_leaves_base_coset)
                .arg(&codeword)
                .arg(&num_leaves)
                .arg(&block)
                .arg(&mut leaves)
                .launch(crate::merkle::keccak_launch_cfg(num_leaves))?;
        }
        stream.synchronize()
    })?;
    line("leaf", "rpx", ns, num_leaves * block.div_ceil(8));
    let ns = median_ns(reps, || {
        crate::rpx::build_inner_tree_levels(&stream, be, &mut nodes, num_leaves as usize)?;
        stream.synchronize()
    })?;
    line("node", "rpx", ns, num_leaves - 1);
    drop(nodes);

    // Poseidon1 W16, both variants: the coset leaves, then 4-ary levels down to
    // fewer than four nodes.
    for (v, tag) in VARIANTS.iter().enumerate() {
        let k = &m.variants[v];
        let name = format!("p1w16-{tag}");
        let mut leaves = stream.alloc_zeros::<u64>(num_leaves as usize * DIGEST)?;
        let ns = median_ns(reps, || {
            unsafe {
                stream
                    .launch_builder(&k.leaves_base_coset)
                    .arg(&codeword)
                    .arg(&num_leaves)
                    .arg(&block)
                    .arg(&mut leaves)
                    .launch(cfg(num_leaves))?;
            }
            stream.synchronize()
        })?;
        line("leaf", &name, ns, num_leaves * block.div_ceil(RATE as u64));

        let mut sizes = Vec::new();
        let mut n = num_leaves;
        while n >= 4 {
            n /= 4;
            sizes.push(n);
        }
        let mut levels: Vec<_> = sizes
            .iter()
            .map(|&s| stream.alloc_zeros::<u64>(s as usize * DIGEST))
            .collect::<Result<_>>()?;
        let ns = median_ns(reps, || {
            for (i, &n_parents) in sizes.iter().enumerate() {
                let (done, rest) = levels.split_at_mut(i);
                let parents = &mut rest[0];
                let children = if i == 0 { &leaves } else { &done[i - 1] };
                unsafe {
                    stream
                        .launch_builder(&k.merkle_level4)
                        .arg(children)
                        .arg(parents)
                        .arg(&n_parents)
                        .launch(cfg(n_parents))?;
                }
            }
            stream.synchronize()
        })?;
        line("node4", &name, ns, sizes.iter().sum());
        drop(levels);
        drop(leaves);
    }

    // Grind: a full scan (limit 0 never hits) of `grind_count` nonces, the
    // simple grid-stride kernel of each hash at two grids.
    let inner = [1u64, 2, 3, 4];
    let inner_dev = stream.clone_htod(inner.as_slice())?;
    let limit = 0u64;
    let base = 0u64;
    for grid in [
        crate::grinding::GRID_DEFAULT,
        4 * crate::grinding::GRID_DEFAULT,
    ] {
        let launch = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut result = stream.clone_htod(&[u64::MAX])?;
        let ns = median_ns(reps, || {
            unsafe {
                stream
                    .launch_builder(&be.rpx_grind_search)
                    .arg(&inner_dev)
                    .arg(&limit)
                    .arg(&base)
                    .arg(&grind_count)
                    .arg(&mut result)
                    .launch(launch)?;
            }
            stream.synchronize()
        })?;
        line("grind", &format!("rpx-g{grid}"), ns, grind_count);
        for (v, tag) in VARIANTS.iter().enumerate() {
            let k = &m.variants[v];
            let ns = median_ns(reps, || {
                unsafe {
                    stream
                        .launch_builder(&k.grind_search)
                        .arg(&inner_dev)
                        .arg(&limit)
                        .arg(&base)
                        .arg(&grind_count)
                        .arg(&mut result)
                        .launch(launch)?;
                }
                stream.synchronize()
            })?;
            line("grind", &format!("p1-{tag}-g{grid}"), ns, grind_count);
        }
    }
    Ok(lines)
}

/// ZisK's leaf hash over coset leaves of a base (`ext3 = false`) or ext3
/// codeword, the geometry of [`leaves_coset`].
pub fn zisk_leaves_coset(codeword: &[u64], block: u64, ext3: bool) -> Result<Vec<u64>> {
    let per = if ext3 { 3 } else { 1 };
    let num_leaves = codeword.len() as u64 / (block * per);
    let z = &module()?.zisk;
    let stream = backend()?.next_stream();
    let input = stream.clone_htod(codeword)?;
    let mut out = stream.alloc_zeros::<u64>(num_leaves as usize * DIGEST)?;
    let f = if ext3 {
        &z.leaves_ext3_coset
    } else {
        &z.leaves_base_coset
    };
    unsafe {
        stream
            .launch_builder(f)
            .arg(&input)
            .arg(&num_leaves)
            .arg(&block)
            .arg(&mut out)
            .launch(cfg(num_leaves))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// ZisK's leaf hash over a column-major LDE matrix (column `c` of row `r` at
/// `columns[c·col_stride + r]`, rows read bit-reversed): one leaf per row, or
/// per row pair when `pair`. `num_rows` is a power of two, at least 2.
pub fn zisk_leaves_rows(
    columns: &[u64],
    col_stride: u64,
    num_cols: u64,
    num_rows: u64,
    pair: bool,
) -> Result<Vec<u64>> {
    assert!(num_rows.is_power_of_two() && num_rows >= 2);
    let z = &module()?.zisk;
    let stream = backend()?.next_stream();
    let input = stream.clone_htod(columns)?;
    let num_leaves = if pair { num_rows / 2 } else { num_rows };
    let mut out = stream.alloc_zeros::<u64>(num_leaves as usize * DIGEST)?;
    let log_num_rows = num_rows.trailing_zeros() as u64;
    let f = if pair {
        &z.leaves_row_pair
    } else {
        &z.leaves_rows
    };
    unsafe {
        stream
            .launch_builder(f)
            .arg(&input)
            .arg(&col_stride)
            .arg(&num_cols)
            .arg(&num_rows)
            .arg(&log_num_rows)
            .arg(&mut out)
            .launch(cfg(num_leaves))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// `n` width-8 permutations of `states` (`n·8` felts) at [`W8_VARIANTS`]`[variant]`.
pub fn w8_permute_many(variant: usize, states: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(states.len() % 8, 0);
    let n = (states.len() / 8) as u64;
    let f = &module()?.zisk.w8_permute_probe[variant];
    let stream = backend()?.next_stream();
    let input = stream.clone_htod(states)?;
    let mut out = stream.alloc_zeros::<u64>(states.len())?;
    unsafe {
        stream
            .launch_builder(f)
            .arg(&input)
            .arg(&n)
            .arg(&mut out)
            .launch(cfg(n))?;
    }
    let host = stream.clone_dtoh(&out)?;
    stream.synchronize()?;
    Ok(host)
}

/// ZisK's grind: the smallest nonce in `[0, count)` whose width-8 permutation
/// of `[c0, c1, c2, nonce, 0, 0, 0, 0]` has lane 0 below `limit`, if any.
pub fn w8_grind(
    variant: usize,
    challenge: &[u64; 3],
    limit: u64,
    count: u64,
    grid: u32,
) -> Result<Option<u64>> {
    let f = &module()?.zisk.w8_grind_search[variant];
    let stream = backend()?.next_stream();
    let challenge_dev = stream.clone_htod(challenge.as_slice())?;
    let mut result = stream.clone_htod(&[u64::MAX])?;
    let base = 0u64;
    unsafe {
        stream
            .launch_builder(f)
            .arg(&challenge_dev)
            .arg(&limit)
            .arg(&base)
            .arg(&count)
            .arg(&mut result)
            .launch(LaunchConfig {
                grid_dim: (grid, 1, 1),
                block_dim: (BLOCK_DIM, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }
    let host = stream.clone_dtoh(&result)?;
    stream.synchronize()?;
    Ok((host[0] != u64::MAX).then_some(host[0]))
}

/// The LDE-row widths the P1 bench hashes: row pairs (the trace-commit layout)
/// and single rows (an ext3 table's slabs).
const ROW_PAIR_WIDTHS: [u64; 3] = [16, 48, 96];
const ROW_WIDTHS: [u64; 2] = [24, 96];

/// ★ Stage P1's microbenchmark: ZisK's instance against the production RPX
/// kernels on one `2^log_len`-felt buffer.
///
/// - `leaf`: 64-felt cosets (the I-HASH geometry): RPX, the I-HASH leaf (v2,
///   the continuity control for FAST 282) and ZisK's leaf.
/// - `node` / `node4`, and `tree` (the whole tree's time per leaf): RPX's
///   binary production walk against 4-ary levels.
/// - `rows*`: the STARK's column-major LDE rows at [`ROW_PAIR_WIDTHS`] and
///   [`ROW_WIDTHS`]: RPX's production leaf kernels against ZisK's leaf, per
///   permutation and per leaf.
/// - `grind`: a full scan of `grind_count` nonces: RPX, the I-HASH W16 grind
///   (v2) and ZisK's W8 grind at both variants, at two grids.
///
/// Every line is `BENCH <what> <hash> <ns> ns/perm [<ns> ns/leaf] (<perms> perms, <ms> ms)`.
pub fn microbench_zisk(log_len: u32, reps: u32, grind_count: u64) -> Result<Vec<String>> {
    let be = backend()?;
    let m = module()?;
    let stream = be.next_stream();
    let len = 1u64 << log_len;
    let mut lines = Vec::new();
    let mut line = |what: &str, hash: &str, ns: f64, perms: u64, leaves: Option<u64>| {
        let per_leaf = leaves
            .map(|l| format!(" {:>8.3} ns/leaf", ns / l as f64))
            .unwrap_or_default();
        let s = format!(
            "BENCH {what:<14} {hash:<10} {:>7.3} ns/perm{per_leaf} ({perms} perms, {:.2} ms)",
            ns / perms as f64,
            ns / 1e6
        );
        println!("{s}");
        lines.push(s);
    };

    let mut buf = stream.alloc_zeros::<u64>(len as usize)?;
    unsafe {
        stream
            .launch_builder(&m.fill)
            .arg(&mut buf)
            .arg(&len)
            .launch(cfg(len))?;
    }
    stream.synchronize()?;

    // Cosets: RPX's production leaves into a tree buffer, then its walk.
    let block = 64u64;
    let num_leaves = len / block;
    let total_nodes = 2 * num_leaves as usize - 1;
    let mut nodes = stream.alloc_zeros::<u8>(total_nodes * 32)?;
    let leaves_off = (num_leaves as usize - 1) * 32;
    let ns = median_ns(reps, || {
        let mut leaves = nodes.slice_mut(leaves_off..leaves_off + num_leaves as usize * 32);
        unsafe {
            stream
                .launch_builder(&be.rpx_leaves_base_coset)
                .arg(&buf)
                .arg(&num_leaves)
                .arg(&block)
                .arg(&mut leaves)
                .launch(crate::merkle::keccak_launch_cfg(num_leaves))?;
        }
        stream.synchronize()
    })?;
    line(
        "leaf",
        "rpx",
        ns,
        num_leaves * block.div_ceil(8),
        Some(num_leaves),
    );
    let ns = median_ns(reps, || {
        crate::rpx::build_inner_tree_levels(&stream, be, &mut nodes, num_leaves as usize)?;
        stream.synchronize()
    })?;
    line("node", "rpx", ns, num_leaves - 1, None);
    line("tree", "rpx", ns, num_leaves - 1, Some(num_leaves));
    drop(nodes);

    let p1_perms = num_leaves * block.div_ceil(RATE as u64);
    let mut leaves = stream.alloc_zeros::<u64>(num_leaves as usize * DIGEST)?;
    for (tag, f) in [
        ("p1-v2", &m.variants[2].leaves_base_coset),
        ("p1-zisk", &m.zisk.leaves_base_coset),
    ] {
        let ns = median_ns(reps, || {
            unsafe {
                stream
                    .launch_builder(f)
                    .arg(&buf)
                    .arg(&num_leaves)
                    .arg(&block)
                    .arg(&mut leaves)
                    .launch(cfg(num_leaves))?;
            }
            stream.synchronize()
        })?;
        line("leaf", tag, ns, p1_perms, Some(num_leaves));
    }
    let mut sizes = Vec::new();
    let mut n = num_leaves;
    while n >= 4 {
        n /= 4;
        sizes.push(n);
    }
    let mut levels: Vec<_> = sizes
        .iter()
        .map(|&s| stream.alloc_zeros::<u64>(s as usize * DIGEST))
        .collect::<Result<_>>()?;
    let level4 = &m.variants[2].merkle_level4;
    let ns = median_ns(reps, || {
        for (i, &n_parents) in sizes.iter().enumerate() {
            let (done, rest) = levels.split_at_mut(i);
            let parents = &mut rest[0];
            let children = if i == 0 { &leaves } else { &done[i - 1] };
            unsafe {
                stream
                    .launch_builder(level4)
                    .arg(children)
                    .arg(parents)
                    .arg(&n_parents)
                    .launch(cfg(n_parents))?;
            }
        }
        stream.synchronize()
    })?;
    let node_perms: u64 = sizes.iter().sum();
    line("node4", "p1-v2", ns, node_perms, None);
    line("tree", "p1-v2", ns, node_perms, Some(num_leaves));
    drop(levels);
    drop(leaves);

    // LDE rows: the buffer read as `num_cols` columns of `2^k` rows.
    for (pair, widths) in [(true, &ROW_PAIR_WIDTHS[..]), (false, &ROW_WIDTHS[..])] {
        for &num_cols in widths {
            let num_rows = 1u64 << (63 - (len / num_cols).leading_zeros());
            let num_leaves = if pair { num_rows / 2 } else { num_rows };
            let felts = if pair { 2 * num_cols } else { num_cols };
            let what = format!("rows{}-w{num_cols}", if pair { "-pair" } else { "" });
            let mut out = stream.alloc_zeros::<u8>(num_leaves as usize * 32)?;
            let ns = median_ns(reps, || {
                let mut view = out.slice_mut(..);
                if pair {
                    crate::rpx::launch_leaves_base_row_pair(
                        &stream, &buf, num_rows, num_cols, num_rows, &mut view,
                    )?;
                } else {
                    crate::rpx::launch_leaves_base(
                        &stream, &buf, num_rows, num_cols, num_rows, &mut view,
                    )?;
                }
                stream.synchronize()
            })?;
            line(
                &what,
                "rpx",
                ns,
                num_leaves * felts.div_ceil(8),
                Some(num_leaves),
            );
            drop(out);
            let mut out = stream.alloc_zeros::<u64>(num_leaves as usize * DIGEST)?;
            let f = if pair {
                &m.zisk.leaves_row_pair
            } else {
                &m.zisk.leaves_rows
            };
            let log_num_rows = num_rows.trailing_zeros() as u64;
            let ns = median_ns(reps, || {
                unsafe {
                    stream
                        .launch_builder(f)
                        .arg(&buf)
                        .arg(&num_rows)
                        .arg(&num_cols)
                        .arg(&num_rows)
                        .arg(&log_num_rows)
                        .arg(&mut out)
                        .launch(cfg(num_leaves))?;
                }
                stream.synchronize()
            })?;
            line(
                &what,
                "p1-zisk",
                ns,
                num_leaves * felts.div_ceil(RATE as u64),
                Some(num_leaves),
            );
        }
    }

    // Grind: full scans (limit 0 never hits), the simple grid-stride kernels.
    let inner = [1u64, 2, 3, 4];
    let inner_dev = stream.clone_htod(inner.as_slice())?;
    let limit = 0u64;
    let base = 0u64;
    for grid in [
        crate::grinding::GRID_DEFAULT,
        4 * crate::grinding::GRID_DEFAULT,
    ] {
        let launch = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut result = stream.clone_htod(&[u64::MAX])?;
        let mut arms: Vec<(String, &CudaFunction)> = vec![
            (format!("rpx-g{grid}"), &be.rpx_grind_search),
            (format!("p1w16-v2-g{grid}"), &m.variants[2].grind_search),
        ];
        for (v, tag) in W8_VARIANTS.iter().enumerate() {
            arms.push((format!("p1w8-{tag}-g{grid}"), &m.zisk.w8_grind_search[v]));
        }
        for (tag, f) in arms {
            let ns = median_ns(reps, || {
                unsafe {
                    stream
                        .launch_builder(f)
                        .arg(&inner_dev)
                        .arg(&limit)
                        .arg(&base)
                        .arg(&grind_count)
                        .arg(&mut result)
                        .launch(launch)?;
                }
                stream.synchronize()
            })?;
            line("grind", &tag, ns, grind_count, None);
        }
    }
    Ok(lines)
}

/// A device buffer of `len` distinct canonical felts (the bench fill), for
/// benches outside this crate that need a large random device matrix without
/// a host upload.
pub fn random_device_matrix(len: usize) -> Result<cudarc::driver::CudaSlice<u64>> {
    let m = module()?;
    let stream = backend()?.next_stream();
    let mut out = stream.alloc_zeros::<u64>(len)?;
    let n = len as u64;
    unsafe {
        stream
            .launch_builder(&m.fill)
            .arg(&mut out)
            .arg(&n)
            .launch(cfg(n))?;
    }
    stream.synchronize()?;
    Ok(out)
}
