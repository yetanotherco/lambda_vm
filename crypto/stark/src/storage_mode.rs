/// Storage backend for intermediate prover state: `Ram` (heap) or `Disk` (mmap).
/// Disk trades wall time for peak RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StorageMode {
    #[default]
    Ram,
    Disk,
}
