use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use troupekit_core::error::ProviderErrorKind;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::tool::ToolSpec;
use troupekit_local::{
    ChatMessage, ChatRole, Flow, GenParams, GenStop, Generation, LocalChat, LocalChatSettings,
    LocalError, LocalInfo, LocalMetrics, LocalModel, ToolMode,
};
use troupekit_providers::request::{CompletionRequest, Delta, StopReason};
use troupekit_providers::Model;

/// A scripted model double: one token per byte of the prompt text, a
/// fixed reply streamed in 3-byte pieces, and a record of what it saw.
/// The real model needs the `llama` feature and a GGUF file (L8).
struct Scripted {
    info: LocalInfo,
    reply: String,
    logprob: Option<f64>,
    block_until_cancel: bool,
    saw_messages: Mutex<Vec<ChatMessage>>,
    saw_max_tokens: Mutex<Option<usize>>,
    generate_calls: Mutex<usize>,
    saw_cancel: AtomicBool,
}

impl Scripted {
    fn new(reply: &str) -> Self {
        Self {
            info: LocalInfo::new(4_096, 1_000, 0, true),
            reply: reply.to_string(),
            logprob: None,
            block_until_cancel: false,
            saw_messages: Mutex::new(Vec::new()),
            saw_max_tokens: Mutex::new(None),
            generate_calls: Mutex::new(0),
            saw_cancel: AtomicBool::new(false),
        }
    }
}

impl LocalModel for Scripted {
    fn info(&self) -> &LocalInfo {
        &self.info
    }

    fn tokenize(&self, text: &str, _special: bool) -> Result<Vec<u32>, LocalError> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, LocalError> {
        Ok(tokens.iter().map(|t| char::from(*t as u8)).collect())
    }

    fn render_chat(&self, msgs: &[ChatMessage], add: bool) -> Result<String, LocalError> {
        *self.saw_messages.lock().expect("lock") = msgs.to_vec();
        let mut out: String = msgs
            .iter()
            .map(|m| format!("<{}>{}", m.role().as_str(), m.content()))
            .collect();
        if add {
            out.push_str("<assistant>");
        }
        Ok(out)
    }

    fn generate(
        &self,
        _prompt: &[u32],
        params: &GenParams,
        on_token: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<Generation, LocalError> {
        *self.generate_calls.lock().expect("lock") += 1;
        *self.saw_max_tokens.lock().expect("lock") = Some(params.max_tokens());
        if self.block_until_cancel {
            let start = Instant::now();
            while start.elapsed() < Duration::from_secs(5) {
                if params.cancel().is_some_and(|c| c.load(Ordering::SeqCst)) {
                    self.saw_cancel.store(true, Ordering::SeqCst);
                    return Ok(Generation::new(
                        String::new(),
                        0,
                        GenStop::Cancelled,
                        None,
                        LocalMetrics::default(),
                    ));
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            return Err(LocalError::Decode("never cancelled".into()));
        }
        let pieces: Vec<String> = self
            .reply
            .as_bytes()
            .chunks(3)
            .map(|c| String::from_utf8_lossy(c).into_owned())
            .collect();
        for piece in &pieces {
            on_token(piece);
        }
        Ok(Generation::new(
            self.reply.clone(),
            pieces.len(),
            GenStop::EndOfText,
            self.logprob,
            LocalMetrics::default(),
        ))
    }

    fn score(&self, _tokens: &[u32]) -> Result<f64, LocalError> {
        Ok(0.0)
    }

    fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, LocalError> {
        Err(LocalError::Unsupported("embeddings"))
    }

    fn reset(&self) {}
}

fn weather() -> ToolSpec {
    ToolSpec {
        name: "get_weather".into(),
        description: "Weather for a city".into(),
        schema: json!({"type":"object","properties":{"city":{"type":"string"}}}),
        hosted: None,
    }
}

fn chat(model: Arc<Scripted>) -> LocalChat {
    LocalChat::new(model, "local/test")
}

async fn both(
    chat: &LocalChat,
    req: &CompletionRequest,
) -> (troupekit_providers::request::CompletionResponse, Vec<Delta>) {
    let unary = chat.complete(req).await.expect("complete");
    let mut deltas = Vec::new();
    let streamed = chat
        .stream(req, &mut |d| deltas.push(d.clone()))
        .await
        .expect("stream");
    assert_eq!(unary, streamed, "complete and stream agree");
    (unary, deltas)
}

#[test]
fn defaults_are_the_safe_values() {
    let s = LocalChatSettings::default();
    assert_eq!(s.tool_mode(), ToolMode::Auto);
    assert!((s.default_temperature() - 0.7).abs() < 1e-6);
    assert_eq!(s.seed(), 0x5452_4B54);
    assert!(!s.ban_eos());
    assert_eq!(s.min_confidence(), None);
}

#[tokio::test]
async fn renders_messages_through_template() {
    let model = Arc::new(Scripted::new("done"));
    let chat = chat(model.clone());
    let mut req = CompletionRequest::user("weather in Lisbon?", 64);
    req.system = Some("Be brief.".into());
    req.messages.push(
        Message::new(
            Role::Assistant,
            vec![ContentPart::ToolCall {
                id: "c1".into(),
                name: "get_weather".into(),
                arguments: json!({"city":"Lisbon"}),
            }],
        )
        .expect("assistant"),
    );
    req.messages.push(
        Message::new(
            Role::Tool,
            vec![ContentPart::ToolResult {
                call_id: "c1".into(),
                content: "sunny".into(),
                is_error: false,
            }],
        )
        .expect("tool"),
    );
    chat.complete(&req).await.expect("complete");
    let seen = model.saw_messages.lock().expect("lock").clone();
    let roles: Vec<ChatRole> = seen.iter().map(|m| m.role()).collect();
    assert_eq!(
        roles,
        [
            ChatRole::System,
            ChatRole::User,
            ChatRole::Assistant,
            ChatRole::User
        ]
    );
    assert!(seen[0].content().contains("Be brief."));
    assert!(
        seen[2].content().contains(r#""tool":"get_weather""#),
        "{}",
        seen[2].content()
    );
    assert!(
        seen[3].content().contains("sunny") && seen[3].content().contains("c1"),
        "{}",
        seen[3].content()
    );
}

#[tokio::test]
async fn tools_add_the_tool_prompt_and_parse_a_call() {
    let model = Arc::new(Scripted::new(
        r#"{"tool":"get_weather","arguments":{"city":"Lisbon"}}"#,
    ));
    let chat = chat(model.clone());
    let mut req = CompletionRequest::user("weather?", 64);
    req.tools = vec![weather()];
    let (resp, deltas) = both(&chat, &req).await;
    assert!(model.saw_messages.lock().expect("lock")[0]
        .content()
        .contains(r#""get_weather""#));
    assert_eq!(resp.stop, StopReason::ToolUse);
    assert!(
        matches!(&resp.message.parts[0], ContentPart::ToolCall { name, .. } if name == "get_weather")
    );
    assert!(
        !deltas.iter().any(|d| matches!(d, Delta::TextDelta { .. })),
        "no text before a tool call: {deltas:?}"
    );
    assert!(deltas
        .iter()
        .any(|d| matches!(d, Delta::ToolCallStarted { name, .. } if name == "get_weather")));
    assert!(deltas
        .iter()
        .any(|d| matches!(d, Delta::ToolCallArgs { .. })));
}

#[tokio::test]
async fn stream_and_complete_return_the_same_response() {
    let model = Arc::new(Scripted::new("Hello there, friend."));
    let chat = chat(model);
    let (resp, deltas) = both(&chat, &CompletionRequest::user("hi", 64)).await;
    assert_eq!(resp.stop, StopReason::EndTurn);
    assert_eq!(resp.message.text_content(), "Hello there, friend.");
    assert_eq!(resp.usage.cost_usd_micros, 0);
    assert!(resp.usage.input_tokens > 0 && resp.usage.output_tokens > 0);
    let text: String = deltas
        .iter()
        .filter_map(|d| match d {
            Delta::TextDelta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "Hello there, friend.");
    assert!(
        deltas
            .iter()
            .filter(|d| matches!(d, Delta::TextDelta { .. }))
            .count()
            > 1,
        "streams per token"
    );
}

#[tokio::test]
async fn dropping_the_future_cancels_generation() {
    let mut double = Scripted::new("");
    double.block_until_cancel = true;
    let model = Arc::new(double);
    let chat = chat(model.clone());
    let req = CompletionRequest::user("hi", 64);
    let timed = tokio::time::timeout(Duration::from_millis(50), chat.complete(&req)).await;
    assert!(timed.is_err(), "the call was still running");
    let start = Instant::now();
    while !model.saw_cancel.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(2) {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(
        model.saw_cancel.load(Ordering::SeqCst),
        "generation saw the cancel flag"
    );
}

#[tokio::test]
async fn low_confidence_returns_server_error() {
    let mut double = Scripted::new("maybe");
    double.logprob = Some(-3.0);
    let chat = LocalChat::new(Arc::new(double), "local/test")
        .with_settings(LocalChatSettings::default().with_min_confidence(Some(0.5)));
    let err = chat
        .complete(&CompletionRequest::user("hi", 16))
        .await
        .expect_err("low confidence");
    assert_eq!(err.kind, ProviderErrorKind::Server);
    assert!(err.message.contains("confidence"), "{}", err.message);

    let mut sure = Scripted::new("yes");
    sure.logprob = Some(-0.01);
    let chat = LocalChat::new(Arc::new(sure), "local/test")
        .with_settings(LocalChatSettings::default().with_min_confidence(Some(0.5)));
    assert!(chat
        .complete(&CompletionRequest::user("hi", 16))
        .await
        .is_ok());
}

#[tokio::test]
async fn context_overflow_is_an_error_before_decoding() {
    let mut double = Scripted::new("x");
    double.info = LocalInfo::new(16, 1_000, 0, true);
    let model = Arc::new(double);
    let chat = chat(model.clone());
    let err = chat
        .complete(&CompletionRequest::user(
            "this prompt is longer than sixteen bytes",
            8,
        ))
        .await
        .expect_err("overflow");
    assert_eq!(err.kind, ProviderErrorKind::Server);
    assert!(err.message.contains("context"), "{}", err.message);
    assert_eq!(*model.generate_calls.lock().expect("lock"), 0);
}

#[tokio::test]
async fn max_tokens_is_capped_to_the_context_left() {
    let mut double = Scripted::new("x");
    double.info = LocalInfo::new(64, 1_000, 0, true);
    let model = Arc::new(double);
    let chat = chat(model.clone());
    chat.complete(&CompletionRequest::user("hi", 10_000))
        .await
        .expect("complete");
    let cap = model
        .saw_max_tokens
        .lock()
        .expect("lock")
        .expect("generated");
    assert!(cap < 64 && cap > 0, "cap {cap}");
}

#[tokio::test]
async fn images_are_refused() {
    let chat = chat(Arc::new(Scripted::new("x")));
    let mut req = CompletionRequest::user("look", 16);
    req.messages[0].parts.push(ContentPart::Image {
        media_type: "image/png".into(),
        source: troupekit_core::message::MediaSource::Base64 {
            data: "AA==".into(),
        },
    });
    let err = chat.complete(&req).await.expect_err("image");
    assert_eq!(err.kind, ProviderErrorKind::BadRequest);
}
