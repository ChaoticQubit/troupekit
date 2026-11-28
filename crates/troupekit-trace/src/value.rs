use std::borrow::Cow;

/// One field value of a trace record.
/// `Content` holds user text (prompts, replies, queries) and is written
/// only when the scope keeps content; otherwise only its byte count.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Value {
    Str(String),
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    Content(String),
}

/// Marks `text` as user content, withheld unless the scope keeps content.
/// Input: the text. Output: a `Value::Content`.
pub fn content(text: impl Into<String>) -> Value {
    Value::Content(text.into())
}

/// Returns the longest prefix of `text` of at most `bytes_max` bytes that
/// ends on a char boundary. Input: text and byte limit. Output: the prefix.
pub(crate) fn cut_to(text: &str, bytes_max: usize) -> &str {
    if text.len() <= bytes_max {
        return text;
    }
    let mut end = bytes_max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    assert!(end <= bytes_max);
    assert!(bytes_max - end < 4);
    &text[..end]
}

macro_rules! from_as {
    ($variant:ident, $target:ty, $($source:ty),*) => {
        $(impl From<$source> for Value {
            fn from(value: $source) -> Value {
                Value::$variant(<$target>::from(value))
            }
        })*
    };
}

from_as!(Int, i64, i8, i16, i32, i64);
from_as!(UInt, u64, u8, u16, u32, u64);
from_as!(Float, f64, f32, f64);
from_as!(Bool, bool, bool);
from_as!(Str, String, &str, String, &String, Box<str>, char);

impl From<isize> for Value {
    fn from(value: isize) -> Value {
        Value::Int(i64::try_from(value).unwrap_or(if value < 0 { i64::MIN } else { i64::MAX }))
    }
}

impl From<usize> for Value {
    fn from(value: usize) -> Value {
        Value::UInt(u64::try_from(value).unwrap_or(u64::MAX))
    }
}

impl From<Cow<'_, str>> for Value {
    fn from(value: Cow<'_, str>) -> Value {
        Value::Str(value.into_owned())
    }
}

impl From<std::time::Duration> for Value {
    fn from(value: std::time::Duration) -> Value {
        Value::UInt(u64::try_from(value.as_micros()).unwrap_or(u64::MAX))
    }
}
