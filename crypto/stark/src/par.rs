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
