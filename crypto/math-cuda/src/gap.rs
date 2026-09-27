//! Temporary A/B knobs for the gap-fix campaign's device-layer fixes, one per
//! fix, default OFF.
//!
//! `LAMBDA_VM_GAP_<ID>=1` turns fix `<ID>` on. Unset, empty or `0` leaves the
//! pre-fix code path untouched, so one binary runs both arms of an A/B. Each
//! knob is read once per process and prints one line the first time it reads
//! ON, so a log shows the knob reached the path it names. A knob is deleted
//! when its fix is kept (the fix becomes the only path) or dropped.

use std::sync::OnceLock;

/// `None`, `""` and `"0"` read as off, `"1"` as on. Anything else panics with
/// the knob's name: a typo that silently read as off would turn a B arm into a
/// second A arm.
pub fn parse(name: &str, value: Option<&str>) -> bool {
    match value {
        None | Some("") | Some("0") => false,
        Some("1") => true,
        Some(other) => panic!("{name} must be `0` or `1`, got `{other}`"),
    }
}

fn read(name: &str, what: &str) -> bool {
    let on = parse(name, std::env::var(name).ok().as_deref());
    if on {
        eprintln!("GAP KNOB {name}=1 read: {what}");
    }
    on
}

/// I1's knob. See [`i1_retain_mempool`].
pub const I1_ENV: &str = "LAMBDA_VM_GAP_I1";

/// I1: keep the stream-ordered pool's retain-all default
/// ([`crate::device::DEFAULT_MEMPOOL_RELEASE_THRESHOLD_BYTES`]) even when
/// `LAMBDA_VM_MEMPOOL_RELEASE_MB` is set.
///
/// The record launchers export `LAMBDA_VM_MEMPOOL_RELEASE_MB=0` after an arm's
/// own environment, so an arm cannot unset it. This knob is how an arm
/// measures the production posture through the unchanged record chain.
pub fn i1_retain_mempool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        read(
            I1_ENV,
            "the mempool keeps its retain-all default; LAMBDA_VM_MEMPOOL_RELEASE_MB is ignored",
        )
    })
}

/// I5's knob. See [`i5_pinned_tree_download`].
pub const I5_ENV: &str = "LAMBDA_VM_GAP_I5";

/// I5: a precomputed subset tree that goes to the host cache is downloaded
/// through a pinned slab (a DMA at pinned speed, overlapped with the
/// multiplicity tree's build) instead of a synchronous copy into pageable
/// memory.
pub fn i5_pinned_tree_download() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        read(
            I5_ENV,
            "precomputed subset trees download through a pinned slab",
        )
    })
}

#[cfg(test)]
mod tests {
    use super::parse;

    #[test]
    fn off_unless_exactly_one() {
        assert!(!parse("K", None));
        assert!(!parse("K", Some("")));
        assert!(!parse("K", Some("0")));
        assert!(parse("K", Some("1")));
    }

    #[test]
    #[should_panic(expected = "K must be `0` or `1`, got `yes`")]
    fn anything_else_is_refused() {
        parse("K", Some("yes"));
    }
}
