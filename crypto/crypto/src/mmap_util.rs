/// Reserve disk blocks up front so this call fails on a full disk.
/// Without reservation, the kernel sends SIGBUS during the later mmap write.
///
/// Linux only, using `posix_fallocate`. On other platforms we only call
/// `set_len` and skip reservation, so the kernel can still send SIGBUS if
/// the disk fills mid-write.
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
