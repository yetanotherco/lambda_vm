/// Resize `file` to `total_bytes` and reserve disk blocks where supported, so
/// later mmap writes fault with `ENOSPC` from this call instead of `SIGBUS`
/// after the temp filesystem fills up.
pub fn reserve_file_blocks(file: &std::fs::File, total_bytes: u64) -> std::io::Result<()> {
    file.set_len(total_bytes)?;
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let ret = unsafe { libc::posix_fallocate(file.as_raw_fd(), 0, total_bytes as i64) };
        if ret != 0 {
            return Err(std::io::Error::from_raw_os_error(ret));
        }
    }
    Ok(())
}
