use crate::tool::{Tool, ToolContext, ToolError, ToolOutput};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use troupekit_core::tool::ToolSpec;

const MAX_NAME_LEN: usize = 64;
const MAX_REPORTED_ERRORS: usize = 5;

/// A named group of tools built from one configuration, such as a search
/// setup or a file of declarative HTTP tools. Register it with
/// `ToolRegistry::register_toolkit`, which adds all tools or none.
#[derive(Clone)]
pub struct Toolkit {
    name: String,
    tools: Vec<Arc<dyn Tool>>,
}

impl Toolkit {
    /// Starts an empty toolkit called `name`.
    /// Returns the toolkit; add tools with `with`.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            tools: Vec::new(),
        }
    }

    /// Adds `tool` and returns the toolkit, so calls chain.
    /// Duplicate names are caught when the toolkit is registered.
    pub fn with(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Returns the toolkit's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the toolkit's tools in the order they were added.
    pub fn tools(&self) -> &[Arc<dyn Tool>] {
        &self.tools
    }
}

struct Entry {
    tool: Arc<dyn Tool>,
    validator: jsonschema::Validator,
}

/// Tools by name, each with its argument schema compiled once.
/// `call` rejects unknown names and schema-invalid arguments before the
/// tool runs. Names are unique and listed in registration order.
#[derive(Default)]
pub struct ToolRegistry {
    entries: Vec<Entry>,
    index: HashMap<String, usize>,
}

impl ToolRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one tool after checking its name and compiling its schema.
    /// Returns `DuplicateTool` for a taken name and `InvalidSpec` for a bad
    /// name or a schema that is not a compilable object schema.
    pub fn register(&mut self, tool: Arc<dyn Tool>) -> Result<(), ToolError> {
        self.register_toolkit(Toolkit::new("").with(tool))
    }

    /// Adds every tool of `toolkit`, or none of them when any check fails.
    /// Applies the same checks as `register`, including duplicates inside
    /// the toolkit itself.
    pub fn register_toolkit(&mut self, toolkit: Toolkit) -> Result<(), ToolError> {
        let mut checked = Vec::with_capacity(toolkit.tools.len());
        for tool in toolkit.tools {
            let name = tool.spec().name.clone();
            check_name(&name)?;
            if self.index.contains_key(&name)
                || checked.iter().any(|e: &Entry| e.tool.spec().name == name)
            {
                return Err(ToolError::DuplicateTool(name));
            }
            let validator = compile_schema(&tool.spec().schema)
                .map_err(|e| ToolError::InvalidSpec(format!("{name}: {e}")))?;
            checked.push(Entry { tool, validator });
        }
        for entry in checked {
            let name = entry.tool.spec().name.clone();
            self.index.insert(name, self.entries.len());
            self.entries.push(entry);
        }
        assert_eq!(self.index.len(), self.entries.len());
        Ok(())
    }

    /// Returns the specs of all tools in registration order.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.entries.iter().map(|e| e.tool.spec().clone()).collect()
    }

    /// Returns the tool called `name`, if registered.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.index.get(name).map(|&i| self.entries[i].tool.clone())
    }

    /// Calls the tool `name` with `args` after validating them.
    /// Returns `UnknownTool`, `InvalidArgs` listing up to five schema
    /// errors, or whatever the tool returns.
    pub async fn call(
        &self,
        name: &str,
        args: Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolError> {
        let entry = self
            .index
            .get(name)
            .map(|&i| &self.entries[i])
            .ok_or_else(|| ToolError::UnknownTool(name.to_string()))?;
        let errors = schema_errors(&entry.validator, &args);
        if !errors.is_empty() {
            return Err(ToolError::InvalidArgs(errors.join("; ")));
        }
        entry.tool.call(args, ctx).await
    }
}

/// Reports whether `name` is a valid tool name: 1 to 64 characters of
/// `[A-Za-z0-9_-]`. The registry refuses any other name, and persona
/// loaders check tool lists with it.
pub fn valid_tool_name(name: &str) -> bool {
    let ok_chars = name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-');
    !name.is_empty() && name.len() <= MAX_NAME_LEN && ok_chars
}

fn check_name(name: &str) -> Result<(), ToolError> {
    if !valid_tool_name(name) {
        return Err(ToolError::InvalidSpec(format!(
            "tool name `{name}` must be 1 to {MAX_NAME_LEN} of [A-Za-z0-9_-]"
        )));
    }
    Ok(())
}

/// Lists why `value` breaks the schema of `validator`, at most five
/// messages, each prefixed with the JSON pointer of the bad value.
/// Returns an empty list when the value is valid.
pub fn schema_errors(validator: &jsonschema::Validator, value: &Value) -> Vec<String> {
    validator
        .iter_errors(value)
        .take(MAX_REPORTED_ERRORS)
        .map(|e| {
            let path = e.instance_path().to_string();
            if path.is_empty() {
                e.to_string()
            } else {
                format!("{path}: {e}")
            }
        })
        .collect()
}

/// Compiles an object `schema` into a validator, with no remote or file
/// `$ref` resolution. Returns an error message when the top level is not
/// `"type": "object"` or the schema does not compile.
pub fn compile_schema(schema: &Value) -> Result<jsonschema::Validator, String> {
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err("schema must be an object schema with \"type\": \"object\"".into());
    }
    jsonschema::validator_for(schema).map_err(|e| e.to_string())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tool::{Tool, ToolContext, ToolError, ToolOutput};
    use async_trait::async_trait;
    use serde_json::{json, Value};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use troupekit_core::tool::ToolSpec;

    pub(crate) struct Echo {
        spec: ToolSpec,
        pub(crate) calls: AtomicUsize,
    }

    impl Echo {
        pub(crate) fn new(name: &str) -> Arc<Self> {
            Self::with_schema(
                name,
                json!({"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}),
            )
        }

        pub(crate) fn with_schema(name: &str, schema: Value) -> Arc<Self> {
            Arc::new(Self {
                spec: ToolSpec {
                    name: name.into(),
                    description: "echo q".into(),
                    schema,
                    hosted: None,
                },
                calls: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl Tool for Echo {
        fn spec(&self) -> &ToolSpec {
            &self.spec
        }

        async fn call(&self, args: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutput::text(args["q"].as_str().unwrap_or_default()))
        }
    }

    #[test]
    fn registers_and_lists_specs_in_order() {
        let mut r = ToolRegistry::new();
        r.register(Echo::new("a")).unwrap();
        r.register(Echo::new("b")).unwrap();
        let names: Vec<_> = r.specs().into_iter().map(|s| s.name).collect();
        assert_eq!(names, ["a", "b"]);
        assert!(r.get("a").is_some());
    }

    #[test]
    fn rejects_duplicate_names() {
        let mut r = ToolRegistry::new();
        r.register(Echo::new("a")).unwrap();
        assert_eq!(
            r.register(Echo::new("a")).unwrap_err(),
            ToolError::DuplicateTool("a".into())
        );
    }

    #[test]
    fn toolkit_registration_is_all_or_nothing() {
        let mut r = ToolRegistry::new();
        r.register(Echo::new("a")).unwrap();
        let kit = Toolkit::new("kit")
            .with(Echo::new("c"))
            .with(Echo::new("a"));
        assert_eq!(kit.name(), "kit");
        assert_eq!(kit.tools().len(), 2);
        assert!(matches!(
            r.register_toolkit(kit),
            Err(ToolError::DuplicateTool(_))
        ));
        assert!(r.get("c").is_none());
        let twice = Toolkit::new("twice")
            .with(Echo::new("d"))
            .with(Echo::new("d"));
        assert!(matches!(
            r.register_toolkit(twice),
            Err(ToolError::DuplicateTool(_))
        ));
        assert!(r.get("d").is_none());
        r.register_toolkit(Toolkit::new("ok").with(Echo::new("e")))
            .unwrap();
        assert!(r.get("e").is_some());
    }

    #[test]
    fn rejects_bad_names_and_schemas() {
        let mut r = ToolRegistry::new();
        for name in ["", "has space", &"x".repeat(65), "dot.name"] {
            assert!(
                matches!(r.register(Echo::new(name)), Err(ToolError::InvalidSpec(_))),
                "{name:?}"
            );
        }
        assert!(r.register(Echo::new(&"x".repeat(64))).is_ok());
        for schema in [
            json!({"type":"string"}),
            json!([1, 2]),
            json!({"type":"object","properties":{"x":{"type":"nope"}}}),
        ] {
            assert!(
                matches!(
                    r.register(Echo::with_schema("s", schema.clone())),
                    Err(ToolError::InvalidSpec(_))
                ),
                "{schema}"
            );
        }
    }

    #[test]
    fn remote_ref_is_rejected_without_io() {
        let mut r = ToolRegistry::new();
        let start = Instant::now();
        let schema = json!({"type":"object","properties":{"x":{"$ref":"http://192.0.2.1/s.json"}}});
        assert!(matches!(
            r.register(Echo::with_schema("remote", schema)),
            Err(ToolError::InvalidSpec(_))
        ));
        assert!(start.elapsed() < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn validates_args_before_call() {
        let mut r = ToolRegistry::new();
        let echo = Echo::new("echo");
        r.register(echo.clone()).unwrap();
        let ctx = ToolContext::default();
        for bad in [json!({"q": 1}), json!({}), json!("q")] {
            let err = r.call("echo", bad.clone(), &ctx).await.unwrap_err();
            assert!(matches!(err, ToolError::InvalidArgs(_)), "{bad}: {err}");
        }
        assert_eq!(echo.calls.load(Ordering::SeqCst), 0);
        assert_eq!(
            r.call("echo", json!({"q":"hi"}), &ctx).await.unwrap(),
            ToolOutput::text("hi")
        );
        assert_eq!(echo.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unknown_tool_is_an_error() {
        let r = ToolRegistry::new();
        let err = r
            .call("nope", json!({}), &ToolContext::default())
            .await
            .unwrap_err();
        assert_eq!(err, ToolError::UnknownTool("nope".into()));
    }

    #[test]
    fn from_result_folds_errors() {
        let out = ToolOutput::from_result(Err(ToolError::Blocked("x".into())));
        assert_eq!(out, ToolOutput::error("blocked: x"));
        assert_eq!(
            ToolOutput::from_result(Ok(ToolOutput::text("y"))),
            ToolOutput::text("y")
        );
    }
}
