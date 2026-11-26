use troupekit_agent::{AgentError, PersonaError};
use troupekit_core::error::CoreError;
use troupekit_core::event::PauseReason;
use troupekit_core::ids::RunId;
use troupekit_ffi_core::{AgentBuilder, ErrorCode, FfiError};
use troupekit_graph::{GraphError, WorkflowError};
use troupekit_kit::KitError;
use troupekit_providers::CatalogError;
use troupekit_tools::ToolError;

fn kit_errors(run_id: RunId) -> Vec<(KitError, ErrorCode)> {
    let all = vec![
        (
            KitError::Config {
                agent: "a".into(),
                message: "bad".into(),
            },
            ErrorCode::Config,
        ),
        (
            KitError::MissingKey {
                vendor: "openai".into(),
                env: "OPENAI_API_KEY".into(),
            },
            ErrorCode::MissingKey,
        ),
        (
            KitError::Catalog {
                agent: "a".into(),
                source: CatalogError::Parse("x".into()),
                hint: "fix it".into(),
            },
            ErrorCode::Catalog,
        ),
        (
            KitError::Persona(PersonaError::new("a.toml", "name", "blank")),
            ErrorCode::Persona,
        ),
        (
            KitError::Agent(AgentError::Checkpoint("torn".into())),
            ErrorCode::Agent,
        ),
        (
            KitError::Tool(ToolError::UnknownTool("t".into())),
            ErrorCode::Tool,
        ),
        (
            KitError::Workflow(WorkflowError::new("w.toml", "nodes", "empty")),
            ErrorCode::Workflow,
        ),
        (
            KitError::Graph(GraphError::Setup("no node".into())),
            ErrorCode::Graph,
        ),
        (
            KitError::Store(CoreError::InvalidMessage("bad".into())),
            ErrorCode::Store,
        ),
        (
            KitError::Failed {
                run_id,
                reason: "boom".into(),
            },
            ErrorCode::Failed,
        ),
        (
            KitError::Paused {
                run_id,
                reason: PauseReason::Budget {
                    limit: "run".into(),
                },
            },
            ErrorCode::Paused,
        ),
        (
            KitError::Retries {
                run_id,
                attempts: 3,
                last_error: "slow down".into(),
            },
            ErrorCode::Retries,
        ),
    ];
    #[cfg(feature = "knowledge")]
    let all = {
        let mut all = all;
        all.push((
            KitError::Context(troupekit_context::ContextError::Invalid("x".into())),
            ErrorCode::Context,
        ));
        all
    };
    all
}

#[test]
fn every_kit_error_has_its_code() {
    let run_id = RunId::new();
    for (kit, code) in kit_errors(run_id) {
        let message = kit.to_string();
        let carries_run = matches!(
            kit,
            KitError::Failed { .. } | KitError::Paused { .. } | KitError::Retries { .. }
        );
        let err = FfiError::from(kit);
        assert_eq!(err.code, code, "{message}");
        assert_eq!(err.message, message);
        let expected = carries_run.then(|| run_id.to_string());
        assert_eq!(err.run_id, expected, "{message}");
    }
}

#[test]
fn fallback_code_is_internal() {
    assert_eq!(ErrorCode::Internal as u32, 99);
}

#[test]
fn codes_are_stable_numbers() {
    let codes = [
        (ErrorCode::Config, 1),
        (ErrorCode::MissingKey, 2),
        (ErrorCode::Catalog, 3),
        (ErrorCode::Persona, 4),
        (ErrorCode::Agent, 5),
        (ErrorCode::Tool, 6),
        (ErrorCode::Workflow, 7),
        (ErrorCode::Graph, 8),
        (ErrorCode::Store, 9),
        (ErrorCode::Context, 10),
        (ErrorCode::Failed, 11),
        (ErrorCode::Paused, 12),
        (ErrorCode::Retries, 13),
        (ErrorCode::Cancelled, 14),
        (ErrorCode::HostCallback, 15),
        (ErrorCode::InvalidArgument, 16),
        (ErrorCode::Unsupported, 17),
        (ErrorCode::Panic, 18),
        (ErrorCode::Internal, 99),
    ];
    for (code, number) in codes {
        assert_eq!(code as u32, number, "{code:?}");
    }
}

#[tokio::test]
async fn error_message_never_holds_the_key() {
    let secret = "sk-test-very-secret-value";
    let err = AgentBuilder::new("calc".into())
        .instructions("Add numbers.".into())
        .model("nobody/x".into())
        .api_key(secret.into())
        .build()
        .await
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::Catalog, "{err}");
    assert!(!err.message.contains(secret), "{err}");
    assert!(!format!("{err:?}").contains(secret));
}
