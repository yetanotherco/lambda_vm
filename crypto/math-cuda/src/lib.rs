//! GPU backend for the lambda-vm STARK prover.
//!
//! Primary entry points: [`lde::coset_lde_base`] for the LDE pipeline and
//! [`logup::logup_aux_resident`] for the device-resident LogUp aux build.
//! Everything else (`ntt`, element-wise arith) is either internal to those
//! pipelines or used by the parity test suite.

pub mod barycentric;
pub mod constraint_interp;
pub mod deep;
pub mod device;
pub mod fri;
pub mod inverse;
pub mod lde;
pub mod logup;
pub mod merkle;
pub mod ntt;
pub mod stagebytes;

// Re-exported for downstream crates so they can refer to CUDA primitive
// types without depending on cudarc directly.
pub use cudarc::driver::{CudaSlice, CudaStream};

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::{Backend, backend};

pub type Result<T> = std::result::Result<T, cudarc::driver::DriverError>;

/// Toolchain sanity: plain wrapping u64 vector add. Not a field op.
pub fn vector_add_u64(a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    launch_binary_u64(a, b, |be| &be.vector_add_u64)
}

/// Goldilocks field add on device, element-wise. Inputs may be non-canonical.
pub fn gl_add_u64(a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    launch_binary_u64(a, b, |be| &be.gl_add)
}

pub fn gl_sub_u64(a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    launch_binary_u64(a, b, |be| &be.gl_sub)
}

pub fn gl_mul_u64(a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    launch_binary_u64(a, b, |be| &be.gl_mul)
}

pub fn gl_neg_u64(a: &[u64]) -> Result<Vec<u64>> {
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let stream = be.next_stream();

    let a_dev = stream.clone_htod(a)?;
    let mut c_dev = stream.alloc_zeros::<u64>(n)?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.gl_neg)
            .arg(&a_dev)
            .arg(&mut c_dev)
            .arg(&n_u64)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&c_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Element-wise ext3 multiply. `a` and `b` are 3n u64s (interleaved
/// [a0,a1,a2,b0,b1,b2,...]). Test helper for the `ext3.cuh` header.
pub fn ext3_mul_u64(a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 3, 0);
    let n = a.len() / 3;
    if n == 0 {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let stream = be.next_stream();
    let a_dev = stream.clone_htod(a)?;
    let b_dev = stream.clone_htod(b)?;
    let mut c_dev = stream.alloc_zeros::<u64>(3 * n)?;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.ext3_mul)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&n_u64)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&c_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Element-wise ext3 subtract. Test helper for `ext3::sub` in `ext3.cuh`.
pub fn ext3_sub_u64(a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 3, 0);
    let n = a.len() / 3;
    if n == 0 {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let stream = be.next_stream();
    let a_dev = stream.clone_htod(a)?;
    let b_dev = stream.clone_htod(b)?;
    let mut c_dev = stream.alloc_zeros::<u64>(3 * n)?;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.ext3_sub)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&n_u64)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&c_dev)?;
    stream.synchronize()?;
    Ok(out)
}

/// Element-wise ext3 add.
pub fn ext3_add_u64(a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len() % 3, 0);
    let n = a.len() / 3;
    if n == 0 {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let stream = be.next_stream();
    let a_dev = stream.clone_htod(a)?;
    let b_dev = stream.clone_htod(b)?;
    let mut c_dev = stream.alloc_zeros::<u64>(3 * n)?;
    let cfg = LaunchConfig::for_num_elems(n as u32);
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(&be.ext3_add)
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&n_u64)
            .launch(cfg)?;
    }
    let out = stream.clone_dtoh(&c_dev)?;
    stream.synchronize()?;
    Ok(out)
}

fn launch_binary_u64<F>(a: &[u64], b: &[u64], pick: F) -> Result<Vec<u64>>
where
    F: for<'a> Fn(&'a Backend) -> &'a cudarc::driver::CudaFunction,
{
    assert_eq!(a.len(), b.len(), "length mismatch");
    let n = a.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let be = backend()?;
    let stream = be.next_stream();

    let a_dev = stream.clone_htod(a)?;
    let b_dev = stream.clone_htod(b)?;
    let mut c_dev = stream.alloc_zeros::<u64>(n)?;

    let cfg = LaunchConfig::for_num_elems(n as u32);
    let n_u64 = n as u64;
    unsafe {
        stream
            .launch_builder(pick(be))
            .arg(&a_dev)
            .arg(&b_dev)
            .arg(&mut c_dev)
            .arg(&n_u64)
            .launch(cfg)?;
    }

    let out = stream.clone_dtoh(&c_dev)?;
    stream.synchronize()?;
    Ok(out)
}
