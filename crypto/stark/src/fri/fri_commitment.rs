use crypto::merkle_tree::{merkle::MerkleTree, traits::IsMerkleTreeBackend};
use math::{
    field::{element::FieldElement, traits::IsField},
    traits::AsBytes,
};

#[cfg_attr(not(feature = "disk-spill"), derive(Clone))]
pub struct FriLayer<F, B>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    B: IsMerkleTreeBackend,
{
    pub evaluation: Vec<FieldElement<F>>,
    pub merkle_tree: MerkleTree<B>,
    pub coset_offset: FieldElement<F>,
    pub domain_size: usize,
    #[cfg(feature = "disk-spill")]
    eval_mmap: Option<EvalMmapBacking>,
}

/// File-backed mmap storage for FRI layer evaluations.
/// After `spill_evaluation_to_disk()`, the in-memory evaluation vector is freed
/// and element access goes through this mmap instead.
#[cfg(feature = "disk-spill")]
#[derive(Clone)]
struct EvalMmapBacking {
    mmap: std::sync::Arc<memmap2::Mmap>,
    /// Owns the file descriptor backing the mmap. Dropping it would close
    /// the descriptor and invalidate the mmap.
    _file: std::sync::Arc<std::fs::File>,
    elem_size: usize,
}

impl<F, B> FriLayer<F, B>
where
    F: IsField,
    FieldElement<F>: AsBytes,
    B: IsMerkleTreeBackend,
{
    pub fn new(
        evaluation: &[FieldElement<F>],
        merkle_tree: MerkleTree<B>,
        coset_offset: FieldElement<F>,
        domain_size: usize,
    ) -> Self {
        Self {
            evaluation: evaluation.to_vec(),
            merkle_tree,
            coset_offset,
            domain_size,
            #[cfg(feature = "disk-spill")]
            eval_mmap: None,
        }
    }

    #[inline]
    pub fn get_evaluation(&self, index: usize) -> &FieldElement<F> {
        #[cfg(feature = "disk-spill")]
        if let Some(ref backing) = self.eval_mmap {
            let offset = index * backing.elem_size;
            // SAFETY: spill_evaluation_to_disk writes self.evaluation as contiguous
            // bytes to this mmap. FieldElement<F> is #[repr(transparent)] over its
            // base type, so the byte layout matches the original elements.
            return unsafe { &*(backing.mmap.as_ptr().add(offset) as *const FieldElement<F>) };
        }
        &self.evaluation[index]
    }

    #[cfg(feature = "disk-spill")]
    pub fn spill_evaluation_to_disk(&mut self) -> std::io::Result<()> {
        use std::io::Write;

        if self.evaluation.is_empty() || self.eval_mmap.is_some() {
            return Ok(());
        }

        let elem_size = std::mem::size_of::<FieldElement<F>>();
        let total_bytes = self.evaluation.len() * elem_size;

        let file = tempfile::tempfile()?;
        file.set_len(total_bytes as u64)?;
        {
            let mut writer =
                std::io::BufWriter::with_capacity(crypto::SPILL_BUF_CAPACITY, &file);
            // SAFETY: FieldElement<F> is #[repr(transparent)], so the Vec
            // can be viewed as a contiguous byte slice.
            let bytes = unsafe {
                std::slice::from_raw_parts(self.evaluation.as_ptr() as *const u8, total_bytes)
            };
            writer.write_all(bytes)?;
            writer.flush()?;
        }
        // SAFETY: tempfile() creates an anonymous file with no filesystem path,
        // so no other process can open or modify it.
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        self.evaluation = Vec::new();
        self.eval_mmap = Some(EvalMmapBacking {
            mmap: std::sync::Arc::new(mmap),
            _file: std::sync::Arc::new(file),
            elem_size,
        });
        Ok(())
    }
}
