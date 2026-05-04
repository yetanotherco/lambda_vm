/// Resize `file` to `total_bytes` and reserve disk blocks via `posix_fallocate`
/// on Linux, so later mmap writes fault with `ENOSPC` from this call instead
/// of `SIGBUS` after the temp filesystem fills up.
///
/// Linux returns `EOPNOTSUPP` (or sometimes `EINVAL`) on filesystems that
/// can't pre-allocate (NFS, some overlay/FUSE mounts). We surface those as
/// errors so callers fail fast rather than risk a SIGBUS during the write.
///
/// On non-Linux targets `set_len` extends the inode but does not reserve
/// blocks: if the temp filesystem fills during the subsequent mmap write,
/// the process is killed by `SIGBUS` with no Rust-level error path.
pub fn reserve_file_blocks(file: &std::fs::File, total_bytes: u64) -> std::io::Result<()> {
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
