//! Tests for FFT backend implementations

use crate::fft::backend::{goldilocks_backend, CpuFft};
use crate::fft::traits::Fft;
use crate::field::element::FieldElement;
use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use alloc::vec::Vec;

#[cfg(feature = "metal")]
use crate::fft::backend::MetalFftBackend;
#[cfg(feature = "metal")]
use crate::gpu::metal::MetalError;

type FE = FieldElement<GoldilocksField>;

// ============================================================================
// goldilocks_backend() tests
// ============================================================================

#[test]
fn test_goldilocks_backend_fft_roundtrip() {
    let backend = goldilocks_backend();
    let original: Vec<FE> = (1..=8).map(|i| FE::from(i as u64)).collect();

    let mut data = original.clone();
    backend.fft(&mut data).expect("FFT failed");
    backend.ifft(&mut data).expect("IFFT failed");

    assert_eq!(data, original, "FFT/IFFT roundtrip failed");
}

#[test]
fn test_goldilocks_backend_batch_fft() {
    let backend = goldilocks_backend();
    let poly_len = 4;
    let num_polys = 2;
    let original: Vec<FE> = (1..=(poly_len * num_polys) as u64).map(FE::from).collect();

    let mut data = original.clone();
    backend
        .batch_fft(&mut data, poly_len)
        .expect("Batch FFT failed");
    backend
        .batch_ifft(&mut data, poly_len)
        .expect("Batch IFFT failed");

    assert_eq!(data, original, "Batch FFT/IFFT roundtrip failed");
}

// ============================================================================
// CpuFft tests
// ============================================================================

#[test]
fn test_cpu_fft_trait_roundtrip() {
    let backend = CpuFft::new();
    let original: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();

    let mut data = original.clone();
    backend.fft(&mut data).expect("FFT failed");

    // Verify FFT actually transformed the data
    assert_ne!(data, original, "FFT should transform data");

    backend.ifft(&mut data).expect("IFFT failed");
    assert_eq!(data, original, "Roundtrip should restore original");
}

#[test]
fn test_cpu_fft_trait_batch() {
    let backend = CpuFft::new();
    let poly_len = 8;
    let original: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();

    let mut data = original.clone();
    backend
        .batch_fft(&mut data, poly_len)
        .expect("Batch FFT failed");
    backend
        .batch_ifft(&mut data, poly_len)
        .expect("Batch IFFT failed");

    assert_eq!(data, original, "Batch roundtrip should restore original");
}

// ============================================================================
// MetalFftBackend tests
// ============================================================================

#[cfg(feature = "metal")]
#[test]
fn test_metal_backend_creation() {
    match MetalFftBackend::try_new() {
        Ok(_) => println!("Metal backend created successfully"),
        Err(MetalError::NoDevice) => {
            println!("No Metal device (expected on non-macOS)");
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_backend_fft_roundtrip() {
    let backend = match MetalFftBackend::try_new() {
        Ok(b) => b,
        Err(MetalError::NoDevice) => {
            println!("Skipping: no Metal device");
            return;
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let original: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
    let mut data = original.clone();

    backend.fft(&mut data).expect("FFT failed");
    assert_ne!(data, original, "FFT should transform data");

    backend.ifft(&mut data).expect("IFFT failed");
    assert_eq!(data, original, "Roundtrip should restore original");
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_backend_batch_fft_roundtrip() {
    let backend = match MetalFftBackend::try_new() {
        Ok(b) => b,
        Err(MetalError::NoDevice) => {
            println!("Skipping: no Metal device");
            return;
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    };

    let poly_len = 8;
    let original: Vec<FE> = (1..=16).map(|i| FE::from(i as u64)).collect();
    let mut data = original.clone();

    backend
        .batch_fft(&mut data, poly_len)
        .expect("Batch FFT failed");
    backend
        .batch_ifft(&mut data, poly_len)
        .expect("Batch IFFT failed");

    assert_eq!(data, original, "Batch roundtrip should restore original");
}

#[cfg(feature = "metal")]
#[test]
fn test_zero_copy_conversion() {
    use crate::fft::backend::metal::as_u64_slice_mut;

    // Verify the zero-copy assumption holds
    let fe = FE::from(12345u64);
    assert_eq!(core::mem::size_of_val(&fe), core::mem::size_of::<u64>());
    assert_eq!(*fe.value(), 12345u64);

    // Test the conversion
    let mut data = alloc::vec![FE::from(1u64), FE::from(2u64), FE::from(3u64)];
    let raw = as_u64_slice_mut(&mut data);
    assert_eq!(raw, &[1u64, 2u64, 3u64]);

    // Modify through raw slice
    raw[1] = 42;
    assert_eq!(*data[1].value(), 42u64);
}
