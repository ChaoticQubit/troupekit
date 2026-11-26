use std::sync::Arc;
use troupekit_ffi_core::{AgentBuilder, ErrorCode, HostError, HostTool};

const OBJECT: &str = r#"{"type":"object"}"#;

struct Shaped {
    name: String,
    description: String,
    schema: String,
}

impl HostTool for Shaped {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn schema_json(&self) -> String {
        self.schema.clone()
    }

    fn call(&self, _args_json: String) -> Result<String, HostError> {
        Ok(String::new())
    }
}

fn shaped(name: &str, description: &str, schema: &str) -> Arc<dyn HostTool> {
    Arc::new(Shaped {
        name: name.into(),
        description: description.into(),
        schema: schema.into(),
    })
}

async fn build_with(tool: Arc<dyn HostTool>) -> Result<(), troupekit_ffi_core::FfiError> {
    AgentBuilder::new("calc".into())
        .instructions("Add numbers.".into())
        .model("openai/gpt-6-luna".into())
        .api_key("k".into())
        .tool(tool)
        .build()
        .await
        .map(|_| ())
}

#[tokio::test]
async fn host_tool_bad_schema_refused() {
    for schema in [
        "not json",
        "[1, 2]",
        "\"text\"",
        "{}",
        r#"{"type":"string"}"#,
    ] {
        let err = build_with(shaped("echo", "Echoes.", schema))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidArgument, "{schema}");
        assert!(err.message.contains("schema"), "{err}");
    }
}

#[tokio::test]
async fn host_tool_bad_name_refused() {
    let long = "a".repeat(65);
    for name in ["", "Echo", "echo-it", "échо", long.as_str()] {
        let err = build_with(shaped(name, "Echoes.", OBJECT))
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::InvalidArgument, "{name}");
        assert!(err.message.contains("name"), "{err}");
    }
    build_with(shaped(&"a".repeat(64), "Echoes.", OBJECT))
        .await
        .unwrap();
}

#[tokio::test]
async fn host_tool_long_description_refused() {
    let long = "d".repeat(16 * 1024 + 1);
    let err = build_with(shaped("echo", &long, OBJECT)).await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    assert!(err.message.contains("description"), "{err}");
}

#[tokio::test]
async fn two_host_tools_with_one_name_are_refused() {
    let err = AgentBuilder::new("calc".into())
        .instructions("Add numbers.".into())
        .model("openai/gpt-6-luna".into())
        .api_key("k".into())
        .tool(shaped("echo", "Echoes.", OBJECT))
        .tool(shaped("echo", "Echoes again.", OBJECT))
        .build()
        .await
        .unwrap_err();
    assert_ne!(err.code, ErrorCode::Unsupported);
}

struct Panicky;

impl HostTool for Panicky {
    fn name(&self) -> String {
        panic!("adapter bug")
    }

    fn description(&self) -> String {
        String::new()
    }

    fn schema_json(&self) -> String {
        OBJECT.into()
    }

    fn call(&self, _args_json: String) -> Result<String, HostError> {
        Ok(String::new())
    }
}

#[tokio::test]
async fn host_tool_panic_when_added_is_panic_code() {
    let err = build_with(Arc::new(Panicky)).await.unwrap_err();
    assert_eq!(err.code, ErrorCode::Panic);
    assert!(err.message.contains("host tool"), "{err}");
}
