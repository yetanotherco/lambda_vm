//! Helpers that abstract over `cfg(feature = "parallel")` for patterns
//! that recur across the prover.

/// Run `f(i)` for `i in 0..n` and return the unzipped pair of result vecs.
/// Parallel when `feature = "parallel"`, sequential otherwise.
pub(crate) fn map_unzip<A, B, F>(n: usize, f: F) -> (Vec<A>, Vec<B>)
where
    F: Fn(usize) -> (A, B) + Sync + Send,
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).unzip()
    }
    #[cfg(not(feature = "parallel"))]
    {
        (0..n).map(f).unzip()
    }
}

/// Run `a()` and `b()`, in parallel under `feature = "parallel"`.
pub(crate) fn join<A, B, FA, FB>(a: FA, b: FB) -> (A, B)
where
    FA: FnOnce() -> A + Send,
    FB: FnOnce() -> B + Send,
    A: Send,
    B: Send,
{
    #[cfg(feature = "parallel")]
    {
        rayon::join(a, b)
    }
    #[cfg(not(feature = "parallel"))]
    {
        (a(), b())
    }
}

/// Map `f(i)` over `range` and collect into a `Vec`, preserving index order.
/// Parallel when `feature = "parallel"`, sequential otherwise. Rayon's
/// `collect()` is index-ordered, so the result is identical either way.
pub(crate) fn par_map_collect<R: Send>(
    range: std::ops::Range<usize>,
    f: impl Fn(usize) -> R + Sync + Send,
) -> Vec<R> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        range.into_par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        range.map(f).collect()
    }
}

/// Run `f(&mut item)` for each element of `slice`. Parallel when
/// `feature = "parallel"`, sequential otherwise (ordering is irrelevant).
pub(crate) fn par_for_each_mut<T: Send>(slice: &mut [T], f: impl Fn(&mut T) + Sync + Send) {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        slice.par_iter_mut().for_each(f);
    }
    #[cfg(not(feature = "parallel"))]
    {
        slice.iter_mut().for_each(f);
    }
}

/// Run `f(&mut item)` for each element of `slice`, short-circuiting on the
/// first `Err`. Parallel when `feature = "parallel"`, sequential otherwise.
// Only called from `disk-spill`-gated paths; keep it available without warning
// when that feature is off.
#[cfg_attr(not(feature = "disk-spill"), allow(dead_code))]
pub(crate) fn par_try_for_each_mut<T: Send, E: Send>(
    slice: &mut [T],
    f: impl Fn(&mut T) -> Result<(), E> + Sync + Send,
) -> Result<(), E> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        slice.par_iter_mut().try_for_each(f)
    }
    #[cfg(not(feature = "parallel"))]
    {
        slice.iter_mut().try_for_each(f)
    }
}
