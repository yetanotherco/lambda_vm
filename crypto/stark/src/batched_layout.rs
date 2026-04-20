/// Tracks per-table column offsets within a shared batched Merkle tree.
/// Each phase (main, aux, composition) has its own layout.
#[derive(Debug, Clone)]
pub struct BatchedLayout {
    /// (col_start, col_end) for each table in the concatenated row.
    pub table_ranges: Vec<(usize, usize)>,
    /// Total number of columns across all tables.
    pub total_columns: usize,
    /// Domain size (LDE size for main/aux, trace size for composition).
    pub domain_size: usize,
}

impl BatchedLayout {
    /// Build layout from per-table column counts.
    pub fn new(column_counts: &[usize], domain_size: usize) -> Self {
        let mut ranges = Vec::with_capacity(column_counts.len());
        let mut offset = 0;
        for &count in column_counts {
            ranges.push((offset, offset + count));
            offset += count;
        }
        BatchedLayout {
            table_ranges: ranges,
            total_columns: offset,
            domain_size,
        }
    }

    pub fn num_tables(&self) -> usize {
        self.table_ranges.len()
    }

    /// Extract one table's columns from a full row of opened values.
    pub fn extract_table<T: Clone>(&self, table_idx: usize, row: &[T]) -> Vec<T> {
        let (start, end) = self.table_ranges[table_idx];
        row[start..end].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layout_construction() {
        let layout = BatchedLayout::new(&[5, 3, 7], 1024);
        assert_eq!(layout.total_columns, 15);
        assert_eq!(layout.table_ranges, vec![(0, 5), (5, 8), (8, 15)]);
        assert_eq!(layout.num_tables(), 3);
        assert_eq!(layout.domain_size, 1024);
    }

    #[test]
    fn test_extract_table() {
        let layout = BatchedLayout::new(&[2, 3], 1024);
        let row = vec![10, 20, 30, 40, 50];
        assert_eq!(layout.extract_table(0, &row), vec![10, 20]);
        assert_eq!(layout.extract_table(1, &row), vec![30, 40, 50]);
    }

    #[test]
    fn test_empty_tables() {
        let layout = BatchedLayout::new(&[0, 3, 0], 512);
        assert_eq!(layout.total_columns, 3);
        assert_eq!(layout.table_ranges, vec![(0, 0), (0, 3), (3, 3)]);
        assert_eq!(layout.extract_table::<i32>(0, &[1, 2, 3]), vec![]);
        assert_eq!(layout.extract_table(1, &[1, 2, 3]), vec![1, 2, 3]);
    }

    #[test]
    fn test_single_table() {
        let layout = BatchedLayout::new(&[4], 2048);
        assert_eq!(layout.total_columns, 4);
        assert_eq!(layout.table_ranges, vec![(0, 4)]);
    }
}
