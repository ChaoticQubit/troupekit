use crate::error::TraceError;
use std::fmt;
use std::str::FromStr;

/// One trace category; every trace point belongs to exactly one.
/// `as_str` gives the snake-case name used in JSON lines and TOML, and
/// `FromStr` parses it back. Later releases may add categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Trace {
    Agents,
    Models,
    Tools,
    Memory,
    Knowledge,
    Graphs,
    Laws,
    Evals,
    Budget,
    Store,
    Routing,
}

const COUNT: usize = 11;
const ALL_BITS: u32 = (1u32 << COUNT) - 1;

const _: () = {
    assert!(COUNT <= 32);
    assert!(Trace::ALL.len() == COUNT);
    let mut i = 0;
    while i < COUNT {
        assert!(Trace::ALL[i] as usize == i);
        i += 1;
    }
};

impl Trace {
    /// Every category, in bit order.
    /// Input: none. Output: an array of all categories.
    /// `TraceSet::all()` holds exactly these.
    pub const ALL: [Trace; COUNT] = [
        Trace::Agents,
        Trace::Models,
        Trace::Tools,
        Trace::Memory,
        Trace::Knowledge,
        Trace::Graphs,
        Trace::Laws,
        Trace::Evals,
        Trace::Budget,
        Trace::Store,
        Trace::Routing,
    ];

    /// Returns the snake-case name of the category.
    /// Input: the category. Output: a static name such as `"tools"`.
    /// The same name appears in the `cat` key of every record.
    pub const fn as_str(self) -> &'static str {
        match self {
            Trace::Agents => "agents",
            Trace::Models => "models",
            Trace::Tools => "tools",
            Trace::Memory => "memory",
            Trace::Knowledge => "knowledge",
            Trace::Graphs => "graphs",
            Trace::Laws => "laws",
            Trace::Evals => "evals",
            Trace::Budget => "budget",
            Trace::Store => "store",
            Trace::Routing => "routing",
        }
    }

    pub(crate) const fn index(self) -> usize {
        self as usize
    }

    pub(crate) const fn bit(self) -> u32 {
        1u32 << self.index()
    }
}

impl fmt::Display for Trace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Trace {
    type Err = TraceError;

    fn from_str(text: &str) -> Result<Trace, TraceError> {
        Trace::ALL
            .into_iter()
            .find(|t| t.as_str() == text)
            .ok_or_else(|| TraceError::unknown(text))
    }
}

/// A set of trace categories stored as one bit per category.
/// Input: categories through `with` or `FromIterator`. Output: `contains`.
/// `bits` and `from_bits` round-trip it for storage without serde.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub struct TraceSet(u32);

impl TraceSet {
    /// Returns the set of every category.
    /// Input: none. Output: the full set.
    pub const fn all() -> TraceSet {
        TraceSet(ALL_BITS)
    }

    /// Returns the empty set.
    /// Input: none. Output: a set that contains nothing.
    pub const fn none() -> TraceSet {
        TraceSet(0)
    }

    /// Returns this set with `t` added.
    /// Input: one category. Output: the larger set.
    pub const fn with(self, t: Trace) -> TraceSet {
        TraceSet(self.0 | t.bit())
    }

    /// Returns whether the set holds `t`.
    /// Input: one category. Output: true when present.
    pub const fn contains(self, t: Trace) -> bool {
        self.0 & t.bit() != 0
    }

    /// Returns every category in either set.
    /// Input: another set. Output: the union.
    pub const fn union(self, other: TraceSet) -> TraceSet {
        TraceSet(self.0 | other.0)
    }

    /// Returns whether the set holds no category.
    /// Input: none. Output: true when empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns the raw bits, one per category in `Trace::ALL` order.
    /// Input: none. Output: the bits; `from_bits` reads them back.
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// Builds a set from raw bits, ignoring bits with no category.
    /// Input: bits from `bits`. Output: the set.
    pub const fn from_bits(bits: u32) -> TraceSet {
        TraceSet(bits & ALL_BITS)
    }

    /// Iterates the categories in the set, in `Trace::ALL` order.
    /// Input: none. Output: an iterator of categories.
    pub fn iter(self) -> impl Iterator<Item = Trace> {
        Trace::ALL.into_iter().filter(move |t| self.contains(*t))
    }
}

impl FromIterator<Trace> for TraceSet {
    fn from_iter<I: IntoIterator<Item = Trace>>(items: I) -> TraceSet {
        items.into_iter().fold(TraceSet::none(), TraceSet::with)
    }
}
