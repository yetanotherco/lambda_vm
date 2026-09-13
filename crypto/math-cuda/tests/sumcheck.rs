//! The sumcheck round and fold kernels against the host they replace.
//!
//! Runs on the merge-queue GPU box via `make test-math-cuda` — `SumcheckSession`
//! needs a real device, like the other tests here.
//!
//! The reference is `multilinear`'s own program evaluation and multilinear
//! fold, so the two cannot drift: what is compared is a sum over the whole cube
//! at each interpolation node, and the tables the fold leaves behind.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext3;
use math::field::goldilocks::GoldilocksField as Gl;
use multilinear::gpu::{ext3_from_raw, ext3_raw, lower};
use multilinear::mle::Mle;
use multilinear::program::{Builder, Program};
use multilinear::sumcheck::round_evaluations_for_program;

type FE = FieldElement<Ext3>;

/// A factor with no structure a kernel could accidentally satisfy.
fn factor(num_vars: usize, seed: u64) -> Mle<Ext3> {
    let evals: Vec<FE> = (0..(1u64 << num_vars))
        .map(|i| {
            let mix = |k: u64| i.wrapping_mul(6364136223846793005 + k).wrapping_add(seed) >> 9;
            FE::new([
                FieldElement::<Gl>::from(mix(1)),
                FieldElement::<Gl>::from(mix(7)),
                FieldElement::<Gl>::from(mix(13)),
            ])
        })
        .collect();
    Mle::new(evals).expect("power of two")
}

/// Every op, a constant, and a chain long enough to recycle slots — the shape
/// a compiled batch has, in miniature.
fn program(width: usize) -> Program<Ext3> {
    let mut b = Builder::<Ext3>::new();
    let mut acc = b.var(0);
    for slot in 1..width {
        let v = b.var(slot);
        let doubled = b.add(v, v);
        let scaled = b.mul(doubled, acc);
        let shifted = b.sub(scaled, v);
        acc = b.neg(shifted);
    }
    let seven = b.fixed(FE::from(7u64));
    let root = b.mul(acc, seven);
    b.finish(root).expect("a root")
}

fn parity(num_vars: usize, width: usize, degree: usize) {
    let factors: Vec<Mle<Ext3>> = (0..width).map(|k| factor(num_vars, k as u64 + 1)).collect();
    let program = program(width);
    let lowered = lower(&program).expect("ext3 lowers");
    let raw: Vec<&[u64]> = factors
        .iter()
        .map(|f| unsafe {
            core::slice::from_raw_parts(f.evals().as_ptr() as *const u64, f.len() * 3)
        })
        .collect();

    let mut session = math_cuda::sumcheck::SumcheckSession::new(
        &raw,
        &lowered.nodes,
        &lowered.consts,
        lowered.num_slots,
        lowered.root_slot,
    )
    .expect("a session (needs a GPU)");

    let mut t = Vec::new();
    for node in 1..=degree {
        t.extend_from_slice(&ext3_raw(&FE::from(node as u64)).expect("ext3"));
    }

    // Two rounds, so the fold's output is what the second one reads.
    let mut folded = factors;
    for round in 0..2 {
        let sums = session.round(&t).expect("a round");
        let device: Vec<FE> = sums.chunks_exact(3).map(ext3_from_raw::<Ext3>).collect();
        let host =
            round_evaluations_for_program(&folded, &program, degree).expect("the host round");
        assert_eq!(
            device, host,
            "round {round} sums differ at 2^{num_vars}, width {width}"
        );

        let r = FE::new([
            FieldElement::<Gl>::from(11 + round as u64),
            FieldElement::<Gl>::from(5),
            FieldElement::<Gl>::from(2),
        ]);
        session.fold(&ext3_raw(&r).expect("ext3")).expect("a fold");
        for f in &mut folded {
            f.fix_first_variable_in_place(&r).expect("a variable");
        }

        let downloaded = session.download().expect("the folded factors");
        for (k, (device, host)) in downloaded.iter().zip(&folded).enumerate() {
            let device: Vec<FE> = device.chunks_exact(3).map(ext3_from_raw::<Ext3>).collect();
            assert_eq!(
                device,
                host.evals(),
                "factor {k} differs after round {round}"
            );
        }
    }
}

#[test]
fn device_rounds_match_the_host_sumcheck() {
    // Wider than one block, narrower than one, and a cube the grid strides over.
    parity(10, 6, 3);
    parity(14, 4, 5);
    parity(9, 12, 2);
}

/// The factors a device builds from the columns must be the ones the host
/// builds from the same columns and uploads.
///
/// The two disagree silently otherwise: a factor read at the wrong offset is a
/// perfectly well-formed proof of a different statement.
fn factor_parity(num_vars: usize, columns: usize, offsets: &[usize], publics: usize) {
    use math_cuda::sumcheck::DeviceFactors;

    let rows = 1usize << num_vars;
    let base: Vec<Mle<Gl>> = (0..columns)
        .map(|k| {
            let evals: Vec<FieldElement<Gl>> = (0..rows as u64)
                .map(|i| FieldElement::<Gl>::from(i.wrapping_mul(2654435761 + k as u64) >> 3))
                .collect();
            Mle::new(evals).expect("power of two")
        })
        .collect();
    let public: Vec<Mle<Ext3>> = (0..publics)
        .map(|k| factor(num_vars, 900 + k as u64))
        .collect();

    // One factor per (column, offset), then the public tables — the order
    // `kinds` gives them in.
    let mut host: Vec<Mle<Ext3>> = Vec::new();
    let mut plan: Vec<u64> = Vec::new();
    for (k, column) in base.iter().enumerate() {
        for offset in offsets {
            let shift = offset % rows;
            let (wrapped, rest) = column.evals().split_at(shift);
            host.push(
                Mle::new(
                    rest.iter()
                        .chain(wrapped)
                        .map(|v| (*v).to_extension::<Ext3>())
                        .collect(),
                )
                .expect("power of two"),
            );
            plan.push((k * rows) as u64);
            plan.push(shift as u64);
            plan.push((host.len() - 1) as u64);
        }
    }
    let mut public_slots = Vec::new();
    for table in &public {
        public_slots.push(host.len());
        host.push(table.clone());
    }

    let raw_columns: Vec<Vec<u64>> = base
        .iter()
        .map(|column| column.evals().iter().map(|v| *v.value()).collect())
        .collect();
    let raw_public: Vec<Vec<u64>> = public
        .iter()
        .map(|table| {
            table
                .evals()
                .iter()
                .flat_map(|v| ext3_raw(v).expect("ext3"))
                .collect()
        })
        .collect();

    let columns_ref: Vec<&[u64]> = raw_columns.iter().map(Vec::as_slice).collect();
    let public_ref: Vec<(usize, &[u64])> = public_slots
        .iter()
        .zip(&raw_public)
        .map(|(slot, table)| (*slot, table.as_slice()))
        .collect();

    let built = DeviceFactors::from_columns(&columns_ref, &plan, &public_ref, rows, host.len())
        .expect("factors on device (needs a GPU)");

    // Read them back the only way a `DeviceFactors` can be read: a sumcheck
    // that binds nothing yet, whose session owns the same buffer.
    let program = program(host.len());
    let lowered = lower(&program).expect("lowers");
    let mut session = built
        .session(
            &[],
            &lowered.nodes,
            &lowered.consts,
            lowered.num_slots,
            lowered.root_slot,
        )
        .expect("a session");
    let mut t = Vec::new();
    for node in 1..=2u64 {
        t.extend_from_slice(&ext3_raw(&FE::from(node)).expect("ext3"));
    }
    let device = session.round(&t).expect("a round");
    let expected = round_evaluations_for_program(&host, &program, 2).expect("the host round");
    let device: Vec<FE> = device.chunks_exact(3).map(ext3_from_raw::<Ext3>).collect();
    assert_eq!(
        device, expected,
        "factors built on device differ at 2^{num_vars}, {columns} columns"
    );
}

#[test]
fn factors_built_on_device_match_the_host() {
    factor_parity(10, 4, &[0, 1], 2);
    factor_parity(12, 3, &[0, 1, 7], 0);
    factor_parity(9, 2, &[0], 3);
}

/// The weight built share by share and the weight built all at once must be
/// the same buffer.
///
/// They are the same `eq` doubling either way; what differs is that one walks
/// a level of every share per launch and the other a level of one share. A
/// share reading the wrong coordinate — the index runs backwards — would be a
/// perfectly well-formed proof of a different claim.
fn weight_parity(len: usize, shares: &[(usize, usize)]) {
    use math_cuda::device::backend;

    let built: Vec<(usize, Vec<u64>, [u64; 3])> = shares
        .iter()
        .enumerate()
        .map(|(k, (offset, vars))| {
            let point: Vec<u64> = (0..*vars)
                .flat_map(|i| {
                    ext3_raw(&FE::new([
                        FieldElement::<Gl>::from((7 * k + i + 1) as u64),
                        FieldElement::<Gl>::from((3 * i + 2) as u64),
                        FieldElement::<Gl>::from((k + 5) as u64),
                    ]))
                    .expect("ext3")
                })
                .collect();
            let scale = ext3_raw(&FE::from((k + 1) as u64)).expect("ext3");
            (*offset, point, [scale[0], scale[1], scale[2]])
        })
        .collect();

    let be = backend().expect("a device");
    let stream = be.next_stream();

    let mut one_at_a_time = stream.alloc_zeros::<u64>(len * 3).expect("alloc");
    for (offset, point, scale) in &built {
        math_cuda::sumcheck::eq_expand_into(&stream, &mut one_at_a_time, *offset, point, scale)
            .expect("a share");
    }
    let mut all_at_once = stream.alloc_zeros::<u64>(len * 3).expect("alloc");
    math_cuda::sumcheck::eq_expand_shares_ext3(&stream, &mut all_at_once, &built)
        .expect("the shares");

    let expected = stream.clone_dtoh(&one_at_a_time).expect("read back");
    let got = stream.clone_dtoh(&all_at_once).expect("read back");
    stream.synchronize().expect("sync");
    assert_eq!(expected, got, "the batched weight differs at len {len}");
}

#[test]
fn a_weight_built_all_at_once_matches_one_share_at_a_time() {
    // Shares of different heights, in and out of descending order, with gaps.
    weight_parity(1 << 10, &[(0, 9), (512, 8), (768, 6), (832, 4)]);
    weight_parity(1 << 12, &[(0, 6), (64, 11), (2112, 5), (2144, 10)]);
    weight_parity(1 << 8, &[(0, 8)]);
}

/// Columns evaluated together against the same columns evaluated one at a
/// time, which is what the batched fold replaces.
#[test]
fn columns_folded_together_match_one_at_a_time() {
    let Ok(_) = math_cuda::device::backend() else {
        eprintln!("no device; skipping");
        return;
    };
    for (vars, width) in [(16usize, 5usize), (17, 12), (16, 1)] {
        let rows = 1usize << vars;
        let columns: Vec<Vec<u64>> = (0..width)
            .map(|k| {
                (0..rows as u64)
                    .map(|i| i.wrapping_mul(6364136223846793005 + k as u64) >> 11)
                    .collect()
            })
            .collect();
        let point: Vec<u64> = (0..vars)
            .flat_map(|i| {
                ext3_raw(&FE::new([
                    FieldElement::<Gl>::from((i + 3) as u64),
                    FieldElement::<Gl>::from((2 * i + 1) as u64),
                    FieldElement::<Gl>::from((i * i + 7) as u64),
                ]))
                .expect("ext3")
            })
            .collect();

        let borrowed: Vec<&[u64]> = columns.iter().map(|c| c.as_slice()).collect();
        let together =
            math_cuda::sumcheck::evaluate_many_base(&borrowed, &point).expect("the batch");
        assert_eq!(together.len(), width);
        for (k, column) in borrowed.iter().enumerate() {
            let alone = math_cuda::sumcheck::evaluate_mle_base(column, &point).expect("one");
            assert_eq!(
                together[k], alone,
                "column {k} of {width} at {vars} vars differs"
            );
        }
    }
}
