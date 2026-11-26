/// A destination for finished trace record lines.
/// Input: one JSON line per call, without a trailing newline.
/// A sink never returns an error and never panics through the caller.
pub trait TraceSink: Send + Sync {
    /// Takes one record line. Errors are counted by the sink, not returned.
    fn write(&self, line: &str);

    /// Pushes buffered lines to their destination. The default does nothing.
    fn flush(&self) {}
}
