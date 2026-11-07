//! The packed tables written straight into the page-locked pool
//! (`LAMBDA_VM_PINNED_POOL_GIB`): every packer's buffer lies in a range the
//! driver has page-locked, and a group's upload of it puts the same words on
//! the card.

use multilinear::gpu::TableColumns;
use multilinear::narrow::NarrowColumns;
use stark::narrow::{NarrowMain, NarrowWriter};

use crate::tables::types::GoldilocksField;

/// `rows × cols` words, column `c` needing `widths[c]` bytes.
fn words(rows: usize, widths: &[u8]) -> Vec<Vec<u64>> {
    widths
        .iter()
        .enumerate()
        .map(|(c, &w)| {
            let top = if w == 8 {
                u64::MAX
            } else {
                (1u64 << (8 * w)) - 1
            };
            (0..rows as u64)
                .map(|r| {
                    ((r.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ c as u64) & top) | (top & !(top >> 1))
                })
                .collect()
        })
        .collect()
}

fn row_major(columns: &[Vec<u64>]) -> Vec<u64> {
    let rows = columns[0].len();
    (0..rows)
        .flat_map(|r| columns.iter().map(move |c| c[r]))
        .collect()
}

#[test]
fn every_packer_writes_into_page_locked_memory_and_the_upload_reads_the_same_words() {
    if math_cuda::device::backend().is_err() {
        eprintln!("PINNED POOL TEST: SKIPPED, no device");
        return;
    }
    assert_eq!(
        math_cuda::pinned::set_pool_gib_for_tests(2),
        2,
        "the knob was read already"
    );
    let pool = math_cuda::pinned::lock_pool().expect("the pool locked");
    assert_eq!(
        (pool.pieces, pool.refused),
        (2, 0),
        "the driver locked both pieces"
    );
    let rows = 1 << 16;
    let widths = [1u8, 2, 4, 8, 8, 4, 2, 1, 8, 8];
    let columns = words(rows, &widths);
    let flat = row_major(&columns);
    let locked = |what: &str, data: &[u8]| {
        assert!(
            data.len() >= math_cuda::pinned::MIN_BYTES,
            "{what}: {} bytes",
            data.len()
        );
        assert!(
            math_cuda::pinned::is_pinned(data.as_ptr(), data.len()),
            "{what}: not in a locked extent"
        );
        assert!(
            math_cuda::pinned::driver_has_locked(data.as_ptr()),
            "{what}: the driver has it pageable"
        );
    };

    let mut writer = NarrowWriter::new(rows, &widths);
    for (c, column) in columns.iter().enumerate() {
        for (r, &w) in column.iter().enumerate() {
            writer.set(r, c, w);
        }
    }
    let (_, _, written) = writer
        .finish()
        .expect("every word fits its width")
        .into_parts();
    locked("NarrowWriter", &written);
    let (_, _, packed_main) = NarrowMain::pack(&flat, widths.len()).into_parts();
    locked("NarrowMain::pack", &packed_main);
    assert_eq!(written, packed_main, "the two stark packers' bytes");

    let raw: Vec<&[u64]> = columns.iter().map(Vec::as_slice).collect();
    let packed = NarrowColumns::pack(&raw).expect("one height");
    locked("NarrowColumns::pack", packed.data());
    let from_rows = NarrowColumns::pack_row_major(&flat, widths.len()).expect("packs");
    locked("NarrowColumns::pack_row_major", from_rows.data());
    assert_eq!(
        packed.data(),
        from_rows.data(),
        "the two multilinear packers' bytes"
    );
    assert_eq!(
        packed.data(),
        &written[..],
        "stark's and multilinear's bytes"
    );

    let before = math_cuda::pinned::stats();
    let store =
        multilinear::gpu::upload_tables(&[TableColumns::<GoldilocksField>::Narrow(&from_rows)])
            .expect("the card takes the table");
    let after = math_cuda::pinned::stats();
    assert!(
        after.upload_pinned >= before.upload_pinned + from_rows.data().len() as u64,
        "the upload's bytes counted as page-locked"
    );
    assert_eq!(
        store.download().expect("the store back"),
        columns,
        "the words on the card"
    );
    eprintln!(
        "PINNED POOL TEST: on the card · {:.1} MiB of the pool handed out, locked in {:.3}s",
        after.carved_bytes as f64 / (1u64 << 20) as f64,
        pool.seconds
    );
}
