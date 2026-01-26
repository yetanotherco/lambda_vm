/// Configuration for trace segmentation.
///
/// Segmentation splits execution logs into fixed-size segments that can be
/// proven independently. Each segment must have a power-of-2 number of rows
/// to satisfy FRI requirements.
#[derive(Debug, Clone, Copy)]
pub struct SegmentConfig {
    /// Exact number of rows per segment (must be power of 2 >= 4).
    pub segment_size: usize,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self { segment_size: 64 }
    }
}

impl SegmentConfig {
    /// Creates a new segment configuration.
    ///
    /// # Panics
    ///
    /// Panics if `segment_size` is less than 4 or not a power of 2.
    pub fn new(segment_size: usize) -> Self {
        assert!(segment_size >= 4, "segment_size must be >= 4");
        assert!(
            segment_size.is_power_of_two(),
            "segment_size must be power of 2"
        );
        Self { segment_size }
    }
}
