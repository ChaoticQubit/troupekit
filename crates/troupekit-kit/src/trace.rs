use crate::agent::Agent;
use crate::error::KitError;
use crate::graph::Graph;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use troupekit_core::ids::RunId;
use troupekit_core::secret::Secret;
use troupekit_trace::{
    default_trace_path, trace_span, JsonlSink, Trace, TraceSet, TraceSink, Tracer,
};

#[derive(Clone, Default)]
pub(crate) struct TraceConfig {
    categories: Option<TraceSet>,
    content: bool,
    sink: Option<Arc<dyn TraceSink>>,
    file: Option<PathBuf>,
}

#[derive(Clone)]
pub(crate) struct Traced {
    pub tracer: Tracer,
    pub path: Option<PathBuf>,
}

impl TraceConfig {
    pub(crate) fn is_on(&self) -> bool {
        self.categories.is_some_and(|c| !c.is_empty())
    }

    fn set_categories(&mut self, categories: TraceSet) {
        self.categories = Some(categories);
    }

    pub(crate) fn open(
        &self,
        owner: &str,
        name: &str,
        secrets: &[Secret],
    ) -> Result<Option<Traced>, KitError> {
        let Some(categories) = self.categories.filter(|c| !c.is_empty()) else {
            return Ok(None);
        };
        let mut builder = Tracer::builder()
            .name(name)
            .categories(categories)
            .content(self.content);
        for secret in secrets {
            builder = builder.scrub(secret.expose());
        }
        let (sink, path): (Arc<dyn TraceSink>, Option<PathBuf>) = match &self.sink {
            Some(sink) => (sink.clone(), None),
            None => {
                let path = self
                    .file
                    .clone()
                    .unwrap_or_else(|| default_trace_path(name, builder.trace_id()));
                let sink = JsonlSink::create(&path).map_err(|e| {
                    KitError::config(owner, format!("trace file `{}`: {e}", path.display()))
                })?;
                (Arc::new(sink), Some(path))
            }
        };
        let tracer = builder
            .sink(sink)
            .build()
            .map_err(|e| KitError::config(owner, format!("tracer: {e}")))?;
        assert_eq!(
            tracer.categories(),
            categories,
            "the tracer keeps the categories"
        );
        Ok(Some(Traced { tracer, path }))
    }
}

/// Runs `work` as one root span `span_name` in `traced`, or untraced when
/// tracing is off. `run_id` is known for a resume; otherwise it is read
/// from the result. Output: the result of `work`, unchanged.
pub(crate) async fn root_span<T>(
    traced: Option<&Traced>,
    cat: Trace,
    span_name: &'static str,
    owner: &str,
    run_id: Option<RunId>,
    work: impl Future<Output = Result<T, KitError>>,
    id_of: impl Fn(&T) -> RunId,
) -> Result<T, KitError> {
    let Some(traced) = traced else {
        return work.await;
    };
    let scoped = async {
        let mut span = trace_span!(cat, span_name, name = owner);
        if let Some(id) = run_id {
            span.record("resumed", true);
            span.record("run_id", id.to_string());
        }
        let result = span.instrument(work).await;
        let found = match &result {
            Ok(value) => Some(id_of(value)),
            Err(err) => err.run_id(),
        };
        if let (None, Some(id)) = (run_id, found) {
            span.record("run_id", id.to_string());
        }
        if result.is_err() {
            span.fail("run did not finish");
        }
        result
    };
    traced.tracer.in_scope(scoped).await
}

pub(crate) fn path_of(traced: &Option<Traced>) -> Option<&Path> {
    traced.as_ref().and_then(|t| t.path.as_deref())
}

impl Agent {
    /// Turns tracing on with every category, or off. Default off.
    /// Covers `build` and every run; spans go to `trace_to`, `trace_file`
    /// or `./traces/<name>-<trace id>.jsonl` (`$TROUPEKIT_TRACE_DIR`).
    pub fn tracing(mut self, on: bool) -> Self {
        self.trace.categories = on.then(TraceSet::all);
        self
    }

    /// Turns tracing on for the listed categories only.
    /// An empty list turns tracing off.
    /// Returns the builder.
    pub fn trace(mut self, categories: impl IntoIterator<Item = Trace>) -> Self {
        self.trace.set_categories(categories.into_iter().collect());
        self
    }

    /// Keeps prompt, reply, tool argument, query and document text in
    /// spans. Default off: spans then hold only byte counts. Secrets are
    /// scrubbed either way.
    pub fn trace_content(mut self, on: bool) -> Self {
        self.trace.content = on;
        self
    }

    /// Sends trace records to `sink` instead of a file.
    /// Takes effect only with tracing on; replaces `trace_file`.
    /// Returns the builder.
    pub fn trace_to(mut self, sink: Arc<dyn TraceSink>) -> Self {
        self.trace.sink = Some(sink);
        self.trace.file = None;
        self
    }

    /// Writes trace records to the new file `path`; an existing file is
    /// refused at `build`. Takes effect only with tracing on; replaces
    /// `trace_to`.
    pub fn trace_file(mut self, path: impl AsRef<Path>) -> Self {
        self.trace.file = Some(path.as_ref().to_path_buf());
        self.trace.sink = None;
        self
    }
}

impl Graph {
    /// Turns tracing on with every category, or off. Default off.
    /// Covers `build`, every run and every agent node; spans go to
    /// `trace_to`, `trace_file` or `./traces/<name>-<trace id>.jsonl`.
    pub fn tracing(mut self, on: bool) -> Self {
        self.trace.categories = on.then(TraceSet::all);
        self
    }

    /// Turns tracing on for the listed categories only.
    /// An empty list turns tracing off.
    /// Returns the builder.
    pub fn trace(mut self, categories: impl IntoIterator<Item = Trace>) -> Self {
        self.trace.set_categories(categories.into_iter().collect());
        self
    }

    /// Keeps prompt, reply, tool argument, query and document text in
    /// spans. Default off: spans then hold only byte counts. Secrets are
    /// scrubbed either way.
    pub fn trace_content(mut self, on: bool) -> Self {
        self.trace.content = on;
        self
    }

    /// Sends trace records to `sink` instead of a file.
    /// Takes effect only with tracing on; replaces `trace_file`.
    /// Returns the builder.
    pub fn trace_to(mut self, sink: Arc<dyn TraceSink>) -> Self {
        self.trace.sink = Some(sink);
        self.trace.file = None;
        self
    }

    /// Writes trace records to the new file `path`; an existing file is
    /// refused at `build`. Takes effect only with tracing on; replaces
    /// `trace_to`.
    pub fn trace_file(mut self, path: impl AsRef<Path>) -> Self {
        self.trace.file = Some(path.as_ref().to_path_buf());
        self.trace.sink = None;
        self
    }
}
