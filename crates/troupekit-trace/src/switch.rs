use crate::category::{Trace, TraceSet};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, PoisonError};

const COUNT: usize = Trace::ALL.len();

static MASK: AtomicU32 = AtomicU32::new(0);
static COUNTS: Mutex<[u32; COUNT]> = Mutex::new([0; COUNT]);

/// Returns whether a point of category `t` records anything here.
/// One relaxed atomic load and a bit test when no live tracer has `t`;
/// otherwise also a check of the current scope on this thread.
#[inline]
pub fn enabled(t: Trace) -> bool {
    if MASK.load(Ordering::Relaxed) & t.bit() == 0 {
        return false;
    }
    scope_enabled(t)
}

#[cold]
#[inline(never)]
fn scope_enabled(t: Trace) -> bool {
    crate::tracer::scope_contains(t)
}

/// Returns the union of the categories of every live tracer.
/// Input: none. Output: the set; empty when no tracer is alive.
pub fn live_categories() -> TraceSet {
    TraceSet::from_bits(MASK.load(Ordering::Relaxed))
}

pub(crate) fn acquire(set: TraceSet) {
    let mut counts = COUNTS.lock().unwrap_or_else(PoisonError::into_inner);
    add_counts(&mut counts, set);
    MASK.store(mask_of(&counts), Ordering::Relaxed);
}

pub(crate) fn release(set: TraceSet) {
    let mut counts = COUNTS.lock().unwrap_or_else(PoisonError::into_inner);
    sub_counts(&mut counts, set);
    MASK.store(mask_of(&counts), Ordering::Relaxed);
}

fn add_counts(counts: &mut [u32; COUNT], set: TraceSet) {
    for t in set.iter() {
        let count = &mut counts[t.index()];
        *count = count.checked_add(1).expect("trace count overflow");
    }
}

fn sub_counts(counts: &mut [u32; COUNT], set: TraceSet) {
    for t in set.iter() {
        let count = &mut counts[t.index()];
        assert!(*count > 0, "trace count underflow for {t}");
        *count -= 1;
    }
}

fn mask_of(counts: &[u32; COUNT]) -> u32 {
    let mask = Trace::ALL
        .into_iter()
        .filter(|t| counts[t.index()] > 0)
        .fold(TraceSet::none(), TraceSet::with)
        .bits();
    assert_eq!(mask & !TraceSet::all().bits(), 0);
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic(expected = "trace count underflow")]
    fn releasing_an_unheld_category_panics() {
        let mut counts = [0; COUNT];
        sub_counts(&mut counts, TraceSet::none().with(Trace::Laws));
    }

    #[test]
    fn mask_follows_counts() {
        let mut counts = [0; COUNT];
        let set = TraceSet::none().with(Trace::Tools).with(Trace::Store);
        add_counts(&mut counts, set);
        add_counts(&mut counts, TraceSet::none().with(Trace::Tools));
        sub_counts(&mut counts, set);
        assert_eq!(mask_of(&counts), TraceSet::none().with(Trace::Tools).bits());
    }
}
