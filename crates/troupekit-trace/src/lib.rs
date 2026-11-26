//! Trace spans for every Troupekit crate, with no dependencies.
//! A disabled trace point costs one atomic load and a branch.
//! Records go to sinks as JSON lines; the `view` module reads them back.

#![forbid(unsafe_code)]
