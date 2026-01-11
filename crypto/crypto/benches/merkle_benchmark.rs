//! Merkle tree benchmarks for Poseidon2 backend
//!
//! Benchmarks the Poseidon2 Merkle tree backend which is field-native
//! and GPU-acceleratable.
//!
//! Run with: cargo bench -p crypto -- merkle

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use crypto::hash::poseidon2::Fp;
use crypto::merkle_tree::{
    backends::types::{BatchPoseidon2Backend, Poseidon2Backend},
    merkle::MerkleTree,
};

/// Simple PRNG for reproducible benchmarks (avoids gen keyword issue in Rust 2024)
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
}

/// Generate random Goldilocks field elements
fn random_goldilocks_elements(count: usize, seed: u64) -> Vec<Fp> {
    let mut rng = SimpleRng::new(seed);
    (0..count).map(|_| Fp::from(rng.next_u64())).collect()
}

/// Generate random field element vectors (for batch backends)
fn random_goldilocks_vectors(rows: usize, cols: usize, seed: u64) -> Vec<Vec<Fp>> {
    let mut rng = SimpleRng::new(seed);
    (0..rows)
        .map(|_| (0..cols).map(|_| Fp::from(rng.next_u64())).collect())
        .collect()
}

/// Benchmark single-element Merkle tree construction
fn bench_merkle_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_poseidon2_build");

    for log_size in [10, 12, 14, 16] {
        let size = 1 << log_size;
        group.throughput(Throughput::Elements(size as u64));

        let elements = random_goldilocks_elements(size, 12345);

        group.bench_with_input(
            BenchmarkId::new("single_element", format!("2^{}", log_size)),
            &elements,
            |b, elements| {
                b.iter(|| black_box(MerkleTree::<Poseidon2Backend>::build(elements).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark batch (vector) Merkle tree construction
fn bench_merkle_batch_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_poseidon2_batch_build");

    for log_size in [10, 12, 14] {
        let rows = 1 << log_size;
        let cols = 4; // Typical for trace columns
        group.throughput(Throughput::Elements(rows as u64));

        let vectors = random_goldilocks_vectors(rows, cols, 54321);

        group.bench_with_input(
            BenchmarkId::new("batch_4cols", format!("2^{}", log_size)),
            &vectors,
            |b, vectors| {
                b.iter(|| black_box(MerkleTree::<BatchPoseidon2Backend>::build(vectors).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark proof generation
fn bench_merkle_proof(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_poseidon2_proof");

    let size = 1 << 14; // 16K leaves
    let elements = random_goldilocks_elements(size, 11111);
    let tree = MerkleTree::<Poseidon2Backend>::build(&elements).unwrap();

    // Benchmark proof generation (100 proofs at random positions)
    let indices: Vec<usize> = (0..100).map(|i| i * (size / 100)).collect();

    group.throughput(Throughput::Elements(100));

    group.bench_function("100_proofs_16k_tree", |b| {
        b.iter(|| {
            for &idx in &indices {
                black_box(tree.get_proof_by_pos(idx).unwrap());
            }
        })
    });

    group.finish();
}

/// Benchmark proof verification
fn bench_merkle_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("merkle_poseidon2_verify");

    let size = 1 << 14;
    let elements = random_goldilocks_elements(size, 22222);
    let tree = MerkleTree::<Poseidon2Backend>::build(&elements).unwrap();

    // Pre-generate proofs
    let indices: Vec<usize> = (0..100).map(|i| i * (size / 100)).collect();
    let proofs: Vec<_> = indices
        .iter()
        .map(|&i| tree.get_proof_by_pos(i).unwrap())
        .collect();

    group.throughput(Throughput::Elements(100));

    group.bench_function("100_verifies_16k_tree", |b| {
        b.iter(|| {
            for (i, proof) in proofs.iter().enumerate() {
                let idx = indices[i];
                black_box(proof.verify::<Poseidon2Backend>(&tree.root, idx, &elements[idx]));
            }
        })
    });

    group.finish();
}

/// Benchmark hash operations (leaves + internal nodes)
fn bench_hash_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("poseidon2_hash");

    // Measure leaf hashing throughput
    let elements: Vec<Fp> = (0..10000).map(|i| Fp::from(i as u64)).collect();

    group.throughput(Throughput::Elements(10000));

    group.bench_function("hash_single_10k", |b| {
        b.iter(|| {
            for e in &elements {
                black_box(crypto::hash::poseidon2::Poseidon2::hash_single(e));
            }
        })
    });

    // Measure compression throughput (internal nodes)
    let pairs: Vec<(Fp, Fp)> = (0..10000)
        .map(|i| (Fp::from(i as u64), Fp::from((i + 10000) as u64)))
        .collect();

    group.bench_function("compress_10k", |b| {
        b.iter(|| {
            for (l, r) in &pairs {
                black_box(crypto::hash::poseidon2::Poseidon2::compress(l, r));
            }
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_merkle_build,
    bench_merkle_batch_build,
    bench_merkle_proof,
    bench_merkle_verify,
    bench_hash_throughput,
);
criterion_main!(benches);
