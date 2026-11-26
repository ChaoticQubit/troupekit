use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use troupekit_local::{
    ChatMessage, ChatRole, Flow, GenParams, GenStop, Generation, Grammar, LocalError, LocalInfo,
    LocalMetrics, LocalModel,
};

/// A model double that splits text on spaces, one token per word.
/// Input: none. Output: a `LocalModel` that counts `tokenize` calls.
/// A real model needs the `llama` feature and a GGUF file (L4, L8).
struct WordModel {
    info: LocalInfo,
    tokenize_calls: AtomicUsize,
}

impl WordModel {
    fn new() -> Self {
        Self {
            info: LocalInfo::new(64, 1_000, 0, true),
            tokenize_calls: AtomicUsize::new(0),
        }
    }
}

impl LocalModel for WordModel {
    fn info(&self) -> &LocalInfo {
        &self.info
    }

    fn tokenize(&self, text: &str, _special: bool) -> Result<Vec<u32>, LocalError> {
        self.tokenize_calls.fetch_add(1, Ordering::SeqCst);
        Ok(text.split_whitespace().map(|w| w.len() as u32).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, LocalError> {
        Ok(tokens
            .iter()
            .map(|t| "x".repeat(*t as usize))
            .collect::<Vec<_>>()
            .join(" "))
    }

    fn render_chat(&self, msgs: &[ChatMessage], add_assistant: bool) -> Result<String, LocalError> {
        let mut out: String = msgs
            .iter()
            .map(|m| format!("{}: {}\n", m.role().as_str(), m.content()))
            .collect();
        if add_assistant {
            out.push_str("assistant: ");
        }
        Ok(out)
    }

    fn generate(
        &self,
        prompt: &[u32],
        params: &GenParams,
        on_token: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<Generation, LocalError> {
        let mut text = String::new();
        let mut count = 0;
        let mut stop = GenStop::MaxTokens;
        while count < params.max_tokens() {
            count += 1;
            text.push('a');
            if on_token("a") == Flow::Stop {
                stop = GenStop::Stop;
                break;
            }
        }
        let metrics = LocalMetrics::default().with_reused_prefix_tokens(prompt.len());
        Ok(Generation::new(text, count, stop, None, metrics))
    }

    fn score(&self, tokens: &[u32]) -> Result<f64, LocalError> {
        Ok(-(tokens.len() as f64))
    }

    fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, LocalError> {
        Err(LocalError::Unsupported("embeddings"))
    }

    fn reset(&self) {}
}

#[test]
fn trait_is_object_safe() {
    let boxed: Box<dyn LocalModel> = Box::new(WordModel::new());
    let shared: Arc<dyn LocalModel> = Arc::new(WordModel::new());
    assert_eq!(boxed.info().context_length(), 64);
    assert_eq!(shared.info().vocab_size(), 1_000);
    let mut seen = 0;
    let out = shared
        .generate(
            &[1, 2],
            &GenParams::default().with_max_tokens(5),
            &mut |_| {
                seen += 1;
                if seen == 3 {
                    Flow::Stop
                } else {
                    Flow::Continue
                }
            },
        )
        .expect("double generates");
    assert_eq!(out.tokens(), 3);
    assert_eq!(out.stop(), GenStop::Stop);
    assert_eq!(out.metrics().reused_prefix_tokens(), 2);
}

#[test]
fn defaults_are_the_safe_values() {
    let p = GenParams::default();
    assert_eq!(p.max_tokens(), 512);
    assert!((p.temperature() - 0.7).abs() < 1e-6);
    assert!((p.top_p() - 0.95).abs() < 1e-6);
    assert_eq!(p.seed(), 0x5452_4B54);
    assert!(p.stop().is_empty());
    assert!(p.grammar().is_none());
    assert!(!p.confidence());
    assert!(!p.ban_eos());
    assert!(p.cancel().is_none());

    let m = LocalMetrics::default();
    assert_eq!(m.prefill_tokens_per_second(), 0.0);
    assert_eq!(m.decode_tokens_per_second(), 0.0);
    assert_eq!(m.time_to_first_token_ms(), 0.0);
    assert_eq!(m.peak_ram_bytes(), None);
    assert_eq!(m.reused_prefix_tokens(), 0);
}

#[test]
fn count_tokens_uses_tokenize() {
    let model = WordModel::new();
    assert_eq!(model.count_tokens("one two three").expect("counts"), 3);
    assert_eq!(model.tokenize_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn setters_change_only_their_field() {
    let p = GenParams::default()
        .with_temperature(0.0)
        .with_top_p(0.5)
        .with_seed(7)
        .with_stop(vec!["END".to_string()])
        .with_confidence(true)
        .with_ban_eos(true);
    assert_eq!(p.temperature(), 0.0);
    assert_eq!(p.top_p(), 0.5);
    assert_eq!(p.seed(), 7);
    assert_eq!(p.stop(), ["END".to_string()]);
    assert!(p.confidence());
    assert!(p.ban_eos());
    assert_eq!(p.max_tokens(), 512);
}

#[test]
fn grammar_rejects_empty_oversized_and_bad_root() {
    assert!(Grammar::new("root ::= \"yes\" | \"no\"", "root").is_ok());
    assert!(matches!(
        Grammar::new("", "root"),
        Err(LocalError::Grammar(_))
    ));
    assert!(matches!(
        Grammar::new("root ::= \"a\"", ""),
        Err(LocalError::Grammar(_))
    ));
    assert!(matches!(
        Grammar::new("root ::= \"a\"", "ro ot"),
        Err(LocalError::Grammar(_))
    ));
    assert!(matches!(
        Grammar::new("root ::= \"a\"\0", "root"),
        Err(LocalError::Grammar(_))
    ));
    let big = format!("root ::= \"{}\"", "a".repeat(Grammar::MAX_BYTES));
    assert!(matches!(
        Grammar::new(&big, "root"),
        Err(LocalError::Grammar(_))
    ));
}

#[test]
fn chat_roles_have_template_names() {
    let names: Vec<&str> = [
        ChatRole::System,
        ChatRole::User,
        ChatRole::Assistant,
        ChatRole::Tool,
    ]
    .iter()
    .map(|r| r.as_str())
    .collect();
    assert_eq!(names, ["system", "user", "assistant", "tool"]);
    let msg = ChatMessage::new(ChatRole::User, "hi");
    assert_eq!(msg.role(), ChatRole::User);
    assert_eq!(msg.content(), "hi");
}

#[test]
fn context_full_error_names_both_sizes() {
    let e = LocalError::ContextFull {
        needed: 5_000,
        limit: 4_096,
    };
    let text = e.to_string();
    assert!(text.contains("5000") && text.contains("4096"), "{text}");
}
