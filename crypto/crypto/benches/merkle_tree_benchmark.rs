use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use crypto::merkle_tree::{
    backends::field_element_vector::{
        FieldElementVectorBackend, QuaternaryFieldElementVectorBackend,
    },
    merkle::MerkleTree,
};
use math::field::{element::FieldElement, fields::fft_friendly::u64_goldilocks::GoldilocksField};
use sha3::Keccak256;

type F = GoldilocksField;
type FE = FieldElement<F>;

type BinaryBackend = FieldElementVectorBackend<F, Keccak256, 32>;
type QuaternaryBackend = QuaternaryFieldElementVectorBackend<F, Keccak256, 32>;

fn make_leaves(n: usize) -> Vec<Vec<FE>> {
    (0..n)
        .map(|i| vec![FE::from(i as u64), FE::from((i * 3 + 7) as u64)])
        .collect()
}

fn bench_tree_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree_build");

    for exp in [12, 14, 16, 18] {
        let n = 1 << exp;
        let leaves = make_leaves(n);

        group.bench_with_input(BenchmarkId::new("binary", format!("2^{exp}")), &leaves, |b, leaves| {
            b.iter(|| {
                black_box(MerkleTree::<BinaryBackend>::build(leaves).unwrap());
            })
        });

        group.bench_with_input(BenchmarkId::new("quaternary", format!("2^{exp}")), &leaves, |b, leaves| {
            b.iter(|| {
                black_box(MerkleTree::<QuaternaryBackend>::build(leaves).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_tree_build_from_hashed(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_tree_build_from_hashed");

    for exp in [12, 14, 16, 18] {
        let n = 1 << exp;
        let leaves = make_leaves(n);
        let hashed_binary: Vec<_> = leaves.iter().map(|l| BinaryBackend::hash_data(l)).collect();
        let hashed_quat: Vec<_> = leaves.iter().map(|l| QuaternaryBackend::hash_data(l)).collect();

        group.bench_with_input(BenchmarkId::new("binary", format!("2^{exp}")), &hashed_binary, |b, hashed| {
            b.iter(|| {
                black_box(MerkleTree::<BinaryBackend>::build_from_hashed_leaves(hashed.clone()).unwrap());
            })
        });

        group.bench_with_input(BenchmarkId::new("quaternary", format!("2^{exp}")), &hashed_quat, |b, hashed| {
            b.iter(|| {
                black_box(MerkleTree::<QuaternaryBackend>::build_from_hashed_leaves(hashed.clone()).unwrap());
            })
        });
    }

    group.finish();
}

fn bench_single_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_single_proof");

    let n = 1 << 16;
    let leaves = make_leaves(n);

    let binary_tree = MerkleTree::<BinaryBackend>::build(&leaves).unwrap();
    let quat_tree = MerkleTree::<QuaternaryBackend>::build(&leaves).unwrap();

    group.bench_function("binary_get_proof", |b| {
        b.iter(|| black_box(binary_tree.get_proof_by_pos(black_box(1000))))
    });

    group.bench_function("quaternary_get_proof", |b| {
        b.iter(|| black_box(quat_tree.get_proof_by_pos(black_box(1000))))
    });

    let binary_proof = binary_tree.get_proof_by_pos(1000).unwrap();
    let quat_proof = quat_tree.get_proof_by_pos(1000).unwrap();

    group.bench_function("binary_verify", |b| {
        b.iter(|| {
            black_box(binary_proof.verify::<BinaryBackend>(
                &binary_tree.root,
                1000,
                &leaves[1000],
            ))
        })
    });

    group.bench_function("quaternary_verify", |b| {
        b.iter(|| {
            black_box(quat_proof.verify::<QuaternaryBackend>(
                &quat_tree.root,
                1000,
                &leaves[1000],
            ))
        })
    });

    group.finish();
}

fn bench_batch_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_batch_proof");

    let n = 1 << 16;
    let leaves = make_leaves(n);
    let pos_list: Vec<usize> = (0..64).map(|i| i * (n / 64)).collect();

    let binary_tree = MerkleTree::<BinaryBackend>::build(&leaves).unwrap();
    let quat_tree = MerkleTree::<QuaternaryBackend>::build(&leaves).unwrap();

    group.bench_function("binary_get_batch_proof", |b| {
        b.iter(|| black_box(binary_tree.get_batch_proof(black_box(&pos_list))))
    });

    group.bench_function("quaternary_get_batch_proof", |b| {
        b.iter(|| black_box(quat_tree.get_batch_proof(black_box(&pos_list))))
    });

    let binary_batch = binary_tree.get_batch_proof(&pos_list).unwrap();
    let quat_batch = quat_tree.get_batch_proof(&pos_list).unwrap();
    let values: Vec<_> = pos_list.iter().map(|&i| leaves[i].clone()).collect();

    group.bench_function("binary_verify_batch", |b| {
        b.iter(|| {
            black_box(binary_batch.verify::<BinaryBackend>(
                &binary_tree.root,
                &pos_list,
                &values,
                n,
            ))
        })
    });

    group.bench_function("quaternary_verify_batch", |b| {
        b.iter(|| {
            black_box(quat_batch.verify::<QuaternaryBackend>(
                &quat_tree.root,
                &pos_list,
                &values,
                n,
            ))
        })
    });

    group.finish();
}

use crypto::merkle_tree::traits::IsMerkleTreeBackend;

criterion_group!(
    benches,
    bench_tree_build,
    bench_tree_build_from_hashed,
    bench_single_proof,
    bench_batch_proof,
);
criterion_main!(benches);
