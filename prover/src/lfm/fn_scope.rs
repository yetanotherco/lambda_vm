//! Test only: regions the emitter marks as one call of a named function, as
//! instruction ranges of the builder, for the Field VM translator to outline.
//! Recording only: nothing is emitted, and nothing is kept outside [`record`].
//! Only outermost regions are recorded.

use std::cell::RefCell;

use super::builder::LfmBuilder;

/// One call: instructions `start..end` of the program, named `name`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    pub name: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Default)]
struct Recorder {
    open: Vec<(String, usize)>,
    done: Vec<Scope>,
}

std::thread_local! {
    static SCOPES: RefCell<Option<Recorder>> = const { RefCell::new(None) };
}

pub fn begin(b: &LfmBuilder, name: impl Into<String>) {
    SCOPES.with(|s| {
        if let Some(r) = s.borrow_mut().as_mut() {
            r.open.push((name.into(), b.instrs().len()));
        }
    });
}

pub fn end(b: &LfmBuilder) {
    SCOPES.with(|s| {
        if let Some(r) = s.borrow_mut().as_mut() {
            let (name, start) = r.open.pop().expect("an open scope");
            if r.open.is_empty() {
                r.done.push(Scope {
                    name,
                    start,
                    end: b.instrs().len(),
                });
            }
        }
    });
}

/// Runs `f` and returns the scopes its emission left, in order.
pub fn record<R>(f: impl FnOnce() -> R) -> (R, Vec<Scope>) {
    let previous = SCOPES.with(|s| s.replace(Some(Recorder::default())));
    let out = f();
    let rec = SCOPES.with(|s| s.replace(previous)).unwrap_or_default();
    assert!(rec.open.is_empty(), "every scope ends");
    (out, rec.done)
}
