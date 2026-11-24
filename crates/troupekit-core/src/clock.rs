use std::sync::atomic::{AtomicI64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// Source of the current time in Unix milliseconds.
/// Code that needs the time takes a `Clock`, so tests can inject
/// a `FixedClock` instead of the wall clock.
pub trait Clock: Send + Sync {
    /// Returns the current time in milliseconds since the Unix epoch.
    /// Takes no input and never fails.
    /// The value is what timestamps on events and records use.
    fn now_ms(&self) -> i64;
}

/// The real wall clock.
/// Reads `SystemTime::now` on every call.
/// Use it in production; use `FixedClock` in tests.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> i64 {
        let d = SystemTime::now().duration_since(UNIX_EPOCH).expect("system clock before 1970");
        i64::try_from(d.as_millis()).expect("time fits in i64")
    }
}

/// A clock that only moves when told to, for deterministic tests.
/// Starts at the time given to `new` and moves forward with `advance`.
/// Safe to share across threads.
#[derive(Debug, Default)]
pub struct FixedClock(AtomicI64);

impl FixedClock {
    /// Creates a clock reading `ms` milliseconds since the Unix epoch.
    /// The clock stays at that value until `advance` is called.
    /// Returns the new clock.
    pub fn new(ms: i64) -> Self {
        Self(AtomicI64::new(ms))
    }

    /// Moves the clock forward by `ms` milliseconds.
    /// Panics when `ms` is negative, since time never runs backwards.
    /// Returns nothing; later `now_ms` calls see the new time.
    pub fn advance(&self, ms: i64) {
        assert!(ms >= 0, "clock cannot go backwards");
        self.0.fetch_add(ms, Ordering::SeqCst);
    }
}

impl Clock for FixedClock {
    fn now_ms(&self) -> i64 {
        self.0.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_clock_advances() {
        let c = FixedClock::new(1_000);
        c.advance(500);
        assert_eq!(c.now_ms(), 1_500);
        assert!(SystemClock.now_ms() > 1_700_000_000_000);
    }
}
