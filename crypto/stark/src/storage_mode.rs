/// Where intermediate prover state (traces, Merkle tree nodes) lives during proving.
///
/// `Ram` keeps everything on the heap (fastest). `Disk` backs those allocations
/// with memory-mapped files so large programs can prove on memory-constrained
/// machines, at the cost of extra wall time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StorageMode {
    #[default]
    Ram,
    Disk,
}
