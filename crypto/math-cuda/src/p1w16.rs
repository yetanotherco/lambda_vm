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

/// The two multiply variants the cubin instantiates (`p1w16.cu` VARIANTS).
pub const VARIANTS: [u32; 2] = [0, 1];

/// One variant's kernels.
pub struct Kernels {
    pub permute_probe: CudaFunction,
    pub leaves_base_coset: CudaFunction,
    pub leaves_ext3_coset: CudaFunction,
    pub merkle_level4: CudaFunction,
    pub grind_search: CudaFunction,
}

/// The loaded module: both variants and the bench fill.
pub struct Module {
    _module: std::sync::Arc<CudaModule>,
    pub variants: [Kernels; 2],
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
    let kernels = |v: u32| -> Result<Kernels> {
        Ok(Kernels {
            permute_probe: module.load_function(&format!("p1w16_permute_probe_v{v}"))?,
            leaves_base_coset: module.load_function(&format!("p1w16_leaves_base_coset_v{v}"))?,
            leaves_ext3_coset: module.load_function(&format!("p1w16_leaves_ext3_coset_v{v}"))?,
            merkle_level4: module.load_function(&format!("p1w16_merkle_level4_v{v}"))?,
            grind_search: module.load_function(&format!("p1w16_grind_search_v{v}"))?,
        })
    };
    let m = Module {
        variants: [kernels(0)?, kernels(1)?],
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
pub fn permute_many(variant: u32, states: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(states.len() % WIDTH, 0);
    let n = (states.len() / WIDTH) as u64;
    let k = &module()?.variants[variant as usize];
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
pub fn leaves_coset(variant: u32, codeword: &[u64], block: u64, ext3: bool) -> Result<Vec<u64>> {
    let per = if ext3 { 3 } else { 1 };
    let num_leaves = codeword.len() as u64 / (block * per);
    let k = &module()?.variants[variant as usize];
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
pub fn merkle_level4(variant: u32, children: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(children.len() % WIDTH, 0);
    let n = (children.len() / WIDTH) as u64;
    let k = &module()?.variants[variant as usize];
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
    variant: u32,
    inner: &[u64; 4],
    limit: u64,
    count: u64,
    grid: u32,
) -> Result<Option<u64>> {
    let k = &module()?.variants[variant as usize];
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
    for v in VARIANTS {
        let k = &m.variants[v as usize];
        let name = format!("p1w16-v{v}");
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
        for v in VARIANTS {
            let k = &m.variants[v as usize];
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
            line("grind", &format!("p1-v{v}-g{grid}"), ns, grind_count);
        }
    }
    Ok(lines)
}
