use math::spill_safe::SpillSafe;

/// Mmap a fresh temp file, copy `slice` into the mapping, downgrade to
/// read-only, and return it.
///
/// Alignment: the mmap base is page-aligned (>= 4096), this function
/// asserts `align_of::<T>() <= 4096`, and Rust guarantees `size_of::<T>()`
/// is a multiple of `align_of::<T>()`, so every element offset is aligned.
pub fn spill_slice_to_mmap<T: SpillSafe>(slice: &[T]) -> std::io::Result<memmap2::Mmap> {
    const {
        assert!(
            std::mem::align_of::<T>() <= 4096,
            "T alignment must fit within mmap page alignment"
        )
    }

    let elem_size = size_of::<T>();
    let total_bytes = (slice.len() as u64)
        .checked_mul(elem_size as u64)
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "spill_slice_to_mmap: byte count overflows u64",
            )
        })?;

    let file = tempfile::tempfile()?;
    reserve_file_blocks(&file, total_bytes)?;

    // SAFETY: tempfile() creates an anonymous file with no filesystem path,
    // so no other process can open or modify it.
    let mut mmap_mut = unsafe { memmap2::MmapOptions::new().map_mut(&file)? };
    // SAFETY: SpillSafe's safety contract requires no padding on T, so
    // `slice`'s bytes are initialized and reading them as &[u8] is sound.
    let bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const u8, size_of_val(slice)) };
    mmap_mut.copy_from_slice(bytes);
    mmap_mut.make_read_only()
}

/// Reserve disk blocks up front so this call fails on a full disk.
/// Without reservation, the kernel sends SIGBUS during the later mmap write.
///
/// Linux only, using `posix_fallocate`. On other platforms we only call
/// `set_len` and skip reservation, so the kernel can still send SIGBUS if
/// the disk fills mid-write.
///
/// `/tmp` is often tmpfs (RAM-backed) on systemd-default distros; set
/// `TMPDIR` to a disk-backed path so spill files actually live on disk.
fn reserve_file_blocks(file: &std::fs::File, total_bytes: u64) -> std::io::Result<()> {
    file.set_len(total_bytes)?;
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let len = i64::try_from(total_bytes).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "spill file too large for posix_fallocate",
            )
        })?;
        let ret = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, len) };
        if ret != 0 {
            return Err(std::io::Error::from_raw_os_error(ret));
        }
    }
    Ok(())
}
