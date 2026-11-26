use crate::category::{Trace, TraceSet};
use crate::error::TraceError;

const NAMES_MAX: usize = 64;

/// Plain trace settings as written in a persona or workflow file.
/// Input: categories, content flag and an optional file path.
/// Output: getters; the default is every category, no content, no file.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TraceSpec {
    categories: TraceSet,
    content: bool,
    file: Option<String>,
}

impl Default for TraceSpec {
    fn default() -> TraceSpec {
        TraceSpec::new(TraceSet::all())
    }
}

impl TraceSpec {
    /// Returns a spec for `categories`, with content off and no file.
    /// Input: the categories. Output: the spec.
    pub fn new(categories: TraceSet) -> TraceSpec {
        TraceSpec {
            categories,
            content: false,
            file: None,
        }
    }

    /// Returns this spec with prompt and reply text kept or withheld.
    /// Input: true to keep content. Output: the spec.
    pub fn with_content(mut self, content: bool) -> TraceSpec {
        self.content = content;
        self
    }

    /// Returns this spec writing to `file`. The path is not checked here.
    /// Input: the file path. Output: the spec.
    pub fn with_file(mut self, file: impl Into<String>) -> TraceSpec {
        self.file = Some(file.into());
        self
    }

    /// Returns the categories. Input: none. Output: the set.
    pub fn categories(&self) -> TraceSet {
        self.categories
    }

    /// Returns whether content is kept. Input: none. Output: the flag.
    pub fn content(&self) -> bool {
        self.content
    }

    /// Returns the file path, if set. Input: none. Output: the path.
    pub fn file(&self) -> Option<&str> {
        self.file.as_deref()
    }

    /// Parses a TOML `trace` value given as one string or a list of strings.
    /// `"all"` or `"true"` is every category, `"false"` or `"none"` is none,
    /// any other item is a category name. Input: at most 64 items.
    /// Output: a spec with those categories, or the first bad item.
    pub fn parse_value<I, S>(items: I) -> Result<TraceSpec, TraceError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut categories = TraceSet::none();
        for (count, item) in items.into_iter().enumerate() {
            if count >= NAMES_MAX {
                return Err(TraceError::TooManyNames(count + 1));
            }
            categories = categories.union(parse_item(item.as_ref())?);
        }
        Ok(TraceSpec::new(categories))
    }
}

fn parse_item(item: &str) -> Result<TraceSet, TraceError> {
    match item {
        "all" | "true" => Ok(TraceSet::all()),
        "false" | "none" => Ok(TraceSet::none()),
        name => name.parse::<Trace>().map(|t| TraceSet::none().with(t)),
    }
}
