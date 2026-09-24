//! Device-vs-host parity for the FRI commit and query phases under a proof
//! format's fold layout — the S3 group encoding and today's pair encoding.
//!
//! Compiled for `cuda` builds with tests or `test-utils`; every entry needs a
//! GPU and a lowered `LAMBDA_VM_GPU_LDE_THRESHOLD` (the device commit admits
//! only LDEs at or above it), so the callers are `#[ignore]`d box tests. The
//! stark crate instantiates them under Keccak and Blake3, the prover crate
//! under the production RPX pin (`tests::zf_rpx_device_tests`).
//!
//! What one [`fri_parity`] call pins, device against the host CPU loop
//! (`commit_phase_cpu_with_layout` / the host query walks) over one random
//! codeword and one transcript:
//! - the terminal coefficients, every committed layer's root, and (when the
//!   device drained them) every layer's evaluations;
//! - the transcript after the commit phase (one more sampled element);
//! - per query (random pair indices plus both ends of the range) every
//!   opened value — the whole group under the group encoding, the sibling
//!   under the pair one — and every authentication path.

use std::format;
use std::string::String;
use std::vec::Vec;

use crypto::fiat_shamir::default_transcript::DefaultTranscript;
use crypto::fiat_shamir::is_transcript::IsTranscript;
use crypto::merkle_tree::cap::CapPolicy;
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField;

use crate::config::StarkHash;
use crate::fri::fri_commitment::FriLayer;
use crate::fri::fri_decommit::FriDecommitment;
use crate::fri::fri_functions::compute_coset_twiddles_inv;
use crate::fri::schedule::{FRI_SCHEDULE_DMAX, fri_chain_start, fri_schedule};
use crate::fri::terminal::FriFoldLayout;
use crate::fri::vectors::splitmix64;
use crate::proof::options::{FriMode, FriScheduleOverride, ProofFormat, ProofOptions};

type F = GoldilocksField;
type E = Degree3GoldilocksExtensionField;
type Felt = FieldElement<F>;
type Ext = FieldElement<E>;

/// Uneven and extreme shapes the DP does not pick at production sizes but the
/// format admits (override): `d = 6` (DMAX, a 64-value group, a multi-block
/// leaf under every hash), and unequal neighbours (a fold-count off-by-one
/// between the commit and the pending folds shows only there).
pub const EXTRA_SHAPES: &[&[u8]] = &[
    &[6],
    &[1, 6],
    &[6, 1],
    &[3, 1, 3],
    &[1, 3],
    &[2, 5, 1],
    &[1, 1, 1],
];

/// Every distinct fold schedule the DP produces for an LDE of `2^B`, `B ≤ 23`
/// (the S3 chain from `B − 1`), at terminal logs 4 (the vectors), 9 (base
/// legs) and 10 (LFM proofs), 3 and 110 queries, cap off and auto; then
/// [`EXTRA_SHAPES`]. Empty schedules (nothing committed) are skipped: the
/// device commit declines them before sampling.
pub fn dp_shapes() -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    for t in [4u32, 9, 10] {
        for q in [3u64, 110] {
            for cap in [CapPolicy::Off, CapPolicy::Auto] {
                for b in 2..=23u32 {
                    let b0 = fri_chain_start(b, false);
                    let s = fri_schedule(b0, t.min(b), q, cap, FRI_SCHEDULE_DMAX);
                    if !s.is_empty() && !out.contains(&s) {
                        out.push(s);
                    }
                }
            }
        }
    }
    for s in EXTRA_SHAPES {
        if !out.iter().any(|x| x.as_slice() == *s) {
            out.push(s.to_vec());
        }
    }
    out
}

/// Options for a group-encoded (dp) proof at blowup `2^blowup_log`, terminal
/// `k`, `queries` queries and cap policy `cap` (which the DP's objective
/// reads), with an optional explicit schedule.
pub fn dp_options(
    blowup_log: u32,
    k: u8,
    queries: usize,
    cap: CapPolicy,
    schedule: Option<&[u8]>,
) -> ProofOptions {
    ProofOptions {
        blowup_factor: 1u8 << blowup_log,
        fri_number_of_queries: queries,
        coset_offset: 3,
        grinding_factor: 0,
        fri_final_poly_log_degree: k,
        format: ProofFormat {
            merkle_cap: cap,
            fri_mode: FriMode::Dp,
            fri_schedule_override: schedule
                .map(|s| FriScheduleOverride::new(s).expect("override fits")),
            ..ProofFormat::DEFAULT
        },
    }
}

/// Today's options (pair encoding) at the same shape parameters.
pub fn pair_options(blowup_log: u32, k: u8) -> ProofOptions {
    ProofOptions {
        blowup_factor: 1u8 << blowup_log,
        fri_number_of_queries: 3,
        coset_offset: 3,
        grinding_factor: 0,
        fri_final_poly_log_degree: k,
        format: ProofFormat::DEFAULT,
    }
}

/// The smallest `(lde_log, options)` whose layout at blowup 2, `k = 1` (so a
/// terminal of 4) has exactly `schedule` as its committed folds.
pub fn smallest_case(schedule: &[u8]) -> (u32, ProofOptions) {
    let sum: u32 = schedule.iter().map(|&d| u32::from(d)).sum();
    // b0 = lde_log − 1 = terminal_log + Σd, terminal_log = 1 + 1.
    (sum + 3, dp_options(1, 1, 3, CapPolicy::Off, Some(schedule)))
}

fn raw(v: &[Ext]) -> Vec<[u64; 3]> {
    v.iter()
        .map(|e| {
            let c = e.value();
            [c[0].canonical(), c[1].canonical(), c[2].canonical()]
        })
        .collect()
}

/// Device-vs-host parity of the FRI commit and query phases under
/// `options`' fold layout for a random codeword of `2^lde_log` values.
/// `resident` keeps the device layers' evals resident only (the device-only
/// envelope's shape), so the device query phase gathers them on device.
///
/// `Err` names the first mismatch, or the device declining (threshold,
/// budget, a wiring gate) — never a silent pass.
pub fn fri_parity<H: StarkHash>(
    lde_log: u32,
    options: &ProofOptions,
    resident: bool,
    seed: u64,
) -> Result<String, String> {
    let blowup_log = options.blowup_factor.trailing_zeros();
    let k = u32::from(options.fri_final_poly_log_degree);
    let layout = FriFoldLayout::for_options(lde_log, blowup_log, options, false)
        .map_err(|e| format!("layout: {e}"))?;
    let n = 1usize << lde_log;
    let mut rng = seed;
    let evals: Vec<Ext> = (0..n)
        .map(|_| {
            Ext::new([
                Felt::from(splitmix64(&mut rng)),
                Felt::from(splitmix64(&mut rng)),
                Felt::from(splitmix64(&mut rng)),
            ])
        })
        .collect();
    let offset = Felt::from(options.coset_offset);
    let tw = compute_coset_twiddles_inv::<F>(&offset, n);
    let t0 = DefaultTranscript::<E>::new(&seed.to_le_bytes());

    let mut t_cpu = t0.clone();
    let (cpu_coeffs, cpu_layers) = crate::fri::commit_phase_cpu_with_layout::<F, E, _, H>(
        evals.clone(),
        &mut t_cpu,
        &offset,
        n,
        blowup_log,
        k,
        &layout,
        &tw,
    );

    let mut t_gpu = t0.clone();
    let device = if resident {
        crate::gpu_lde::try_fri_commit_gpu_resident::<F, E, _, H::Pair<E>>(
            &evals, &mut t_gpu, &offset, n, blowup_log, k, &layout, &tw,
        )
    } else {
        crate::gpu_lde::try_fri_commit_gpu::<F, E, _, H::Pair<E>>(
            &evals, &mut t_gpu, &offset, n, blowup_log, k, &layout, &tw,
        )
    };
    let (gpu_coeffs, gpu_layers) = device.ok_or_else(|| {
        format!(
            "the device FRI commit declined at LDE 2^{lde_log} (schedule {:?}): \
             run with a GPU and LAMBDA_VM_GPU_LDE_THRESHOLD <= {n}",
            layout.schedule
        )
    })?;

    let what = format!(
        "LDE 2^{lde_log}, schedule {:?}, legacy {}, resident {resident}",
        layout.schedule,
        layout.is_legacy()
    );
    if gpu_coeffs != cpu_coeffs {
        return Err(format!("{what}: terminal coefficients differ"));
    }
    if gpu_layers.len() != cpu_layers.len() || cpu_layers.len() != layout.num_committed {
        return Err(format!(
            "{what}: {} device layers, {} host layers",
            gpu_layers.len(),
            cpu_layers.len()
        ));
    }
    for (j, (g, c)) in gpu_layers.iter().zip(&cpu_layers).enumerate() {
        if g.merkle_tree.root != c.merkle_tree.root {
            return Err(format!("{what}: layer {j} root differs"));
        }
        if g.gpu_tree.as_ref().map(|t| t.root) != Some(c.merkle_tree.root) {
            return Err(format!("{what}: layer {j} resident tree root differs"));
        }
        if resident {
            if !g.evaluation.is_empty() || g.gpu_evals.is_none() {
                return Err(format!("{what}: layer {j} is not device-only"));
            }
        } else if raw(&g.evaluation) != raw(&c.evaluation) {
            return Err(format!("{what}: layer {j} evaluations differ"));
        }
    }
    let (after_cpu, after_gpu): (Ext, Ext) =
        (t_cpu.sample_field_element(), t_gpu.sample_field_element());
    if after_cpu != after_gpu {
        return Err(format!("{what}: the transcripts diverged"));
    }

    // Queries: random pair indices and both ends of the range.
    let half = n / 2;
    let mut iotas: Vec<usize> = (0..40)
        .map(|_| (splitmix64(&mut rng) % half as u64) as usize)
        .collect();
    iotas.push(0);
    iotas.push(half - 1);
    let (cpu_q, gpu_q) = queries::<H>(&cpu_layers, &gpu_layers, &iotas, &layout);
    let gpu_q = gpu_q.ok_or_else(|| format!("{what}: the device query phase declined"))?;
    for (q, (a, b)) in cpu_q.iter().zip(&gpu_q).enumerate() {
        if raw(&a.layers_evaluations_sym) != raw(&b.layers_evaluations_sym) {
            return Err(format!(
                "{what}: query {q} (iota {}) values differ",
                iotas[q]
            ));
        }
        if a.layers_auth_paths.len() != b.layers_auth_paths.len()
            || a.layers_auth_paths
                .iter()
                .zip(&b.layers_auth_paths)
                .any(|(x, y)| x.merkle_path != y.merkle_path)
        {
            return Err(format!(
                "{what}: query {q} (iota {}) paths differ",
                iotas[q]
            ));
        }
        if a.layers_evaluations_sym.len() != layout.opened_values_per_query() {
            return Err(format!("{what}: query {q} opens the wrong value count"));
        }
    }
    Ok(format!(
        "{what}: {} layers, {} queries equal",
        layout.num_committed,
        iotas.len()
    ))
}

#[allow(clippy::type_complexity)]
fn queries<H: StarkHash>(
    cpu: &[FriLayer<E, H::Pair<E>>],
    gpu: &[FriLayer<E, H::Pair<E>>],
    iotas: &[usize],
    layout: &FriFoldLayout,
) -> (Vec<FriDecommitment<E>>, Option<Vec<FriDecommitment<E>>>) {
    if layout.is_legacy() {
        // Host layers carry no device tree, so `query_phase` walks them on
        // the host; the device layers take the device gather.
        (
            crate::fri::query_phase::<E, H>(cpu, iotas),
            crate::gpu_lde::try_fri_query_phase_gpu::<E, H::Pair<E>>(gpu, iotas),
        )
    } else {
        (
            crate::fri::query_phase_groups_host::<E, H>(cpu, iotas, layout),
            crate::gpu_lde::try_fri_query_phase_gpu_groups::<E, H::Pair<E>>(gpu, iotas, layout),
        )
    }
}

/// One parity case: an LDE log and the options whose layout it runs under.
pub type Case = (u32, ProofOptions);

/// The shape sweep: every [`dp_shapes`] entry at its [`smallest_case`].
pub fn sweep_cases() -> Vec<Case> {
    dp_shapes().iter().map(|s| smallest_case(s)).collect()
}

/// Production sizes at the DP's own schedules: base legs (blowup 4, k = 7,
/// T = 9) at B = 14, 19, 21, 23 and an LFM-shaped proof (k = 8, T = 10) at
/// B = 22, 110 queries, cap auto (the objective the DP prices); and today's
/// pair encoding at B = 21.
pub fn production_cases() -> Vec<Case> {
    let mut cases: Vec<Case> = [14u32, 19, 21, 23]
        .iter()
        .map(|&b| (b, dp_options(2, 7, 110, CapPolicy::Auto, None)))
        .collect();
    cases.push((22, dp_options(2, 8, 110, CapPolicy::Auto, None)));
    cases.push((21, pair_options(2, 7)));
    cases
}

/// Device-only layers (no host copy of any layer's evals, so the query phase
/// gathers the groups off the resident evals): uneven shapes, the DMAX group,
/// a DP schedule at B = 16, and today's encoding.
pub fn resident_cases() -> Vec<Case> {
    let mut cases: Vec<Case> = [&[3u8, 1, 3][..], &[6, 1], &[1, 6]]
        .iter()
        .map(|s| smallest_case(s))
        .collect();
    cases.push((16, dp_options(2, 7, 110, CapPolicy::Auto, None)));
    cases.push((14, pair_options(2, 7)));
    cases
}

/// Today's encoding through the layout-taking drive: one committed layer and
/// up, blowup 2 and 4.
pub fn legacy_cases() -> Vec<Case> {
    let mut cases: Vec<Case> = [4u32, 5, 8, 12]
        .iter()
        .map(|&b| (b, pair_options(1, 1)))
        .collect();
    cases.extend([10u32, 16].iter().map(|&b| (b, pair_options(2, 3))));
    cases
}

/// Run [`fri_parity`] over `cases` (seeds `seed_base + i`), printing one
/// `FRIDEV` line per case; `Err` lists every failing case.
pub fn run_cases<H: StarkHash>(
    name: &str,
    cases: &[Case],
    resident: bool,
    seed_base: u64,
) -> Result<usize, Vec<String>> {
    let mut failures = Vec::new();
    for (i, (b, opts)) in cases.iter().enumerate() {
        match fri_parity::<H>(*b, opts, resident, seed_base + i as u64) {
            Ok(msg) => std::println!("FRIDEV {name} {msg}"),
            Err(e) => failures.push(e),
        }
    }
    if failures.is_empty() {
        std::println!("FRIDEV {name}: {} cases equal", cases.len());
        Ok(cases.len())
    } else {
        Err(failures)
    }
}
