/// Resize `file` to `total_bytes` and reserve disk blocks where supported, so
/// later mmap writes fault with `ENOSPC` from this call instead of `SIGBUS`
/// after the temp filesystem fills up.
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
        // EOPNOTSUPP / EINVAL on overlay or network filesystems: file is sized
        // by `set_len`, just no early-ENOSPC guarantee.
        if ret != 0 && ret != libc::EOPNOTSUPP && ret != libc::EINVAL {
            return Err(std::io::Error::from_raw_os_error(ret));
        }
    }
    Ok(())
}
