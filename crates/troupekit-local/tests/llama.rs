use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use troupekit_local::llama::{LlamaModel, LlamaSettings};
use troupekit_local::{
    ChatMessage, ChatRole, Flow, GenParams, GenStop, Grammar, LocalError, LocalModel,
};

const FETCH: &str =
    "sh scripts/fetch-test-model.sh, then set TROUPEKIT_TEST_GGUF to the printed path";

fn test_gguf() -> PathBuf {
    let path = std::env::var_os("TROUPEKIT_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("TROUPEKIT_TEST_GGUF is not set: run {FETCH}"));
    assert!(path.is_file(), "{} is missing: run {FETCH}", path.display());
    path
}

fn load() -> LlamaModel {
    LlamaModel::load(&test_gguf(), LlamaSettings::default().with_context(1_024))
        .expect("test GGUF loads")
}

fn greedy(max_tokens: usize) -> GenParams {
    GenParams::default()
        .with_temperature(0.0)
        .with_max_tokens(max_tokens)
}

fn prompt(model: &LlamaModel, text: &str) -> Vec<u32> {
    let chat = model
        .render_chat(&[ChatMessage::new(ChatRole::User, text)], true)
        .expect("template renders");
    model.tokenize(&chat, true).expect("tokenizes")
}

#[test]
fn llama_backend_initializes() {
    let info = troupekit_local::llama::system_info();
    assert!(!info.is_empty(), "llama.cpp reported no system info");
    assert_eq!(troupekit_local::llama::system_info(), info);
}

#[test]
fn tokenize_round_trips_ascii() {
    let model = load();
    let text = "The quick brown fox jumps over the lazy dog.";
    let tokens = model.tokenize(text, false).expect("tokenizes");
    assert!(!tokens.is_empty());
    assert!(tokens
        .iter()
        .all(|t| (*t as usize) < model.info().vocab_size()));
    let back = model.detokenize(&tokens).expect("detokenizes");
    assert_eq!(back.trim_start(), text);
    assert_eq!(model.count_tokens(text).expect("counts"), tokens.len());
}

#[test]
fn generate_stops_at_max_tokens() {
    let model = load();
    let p = prompt(&model, "Count from one to one hundred.");
    let out = model
        .generate(&p, &greedy(4), &mut |_| Flow::Continue)
        .expect("generates");
    assert!(out.tokens() <= 4);
    if out.tokens() == 4 {
        assert_eq!(out.stop(), GenStop::MaxTokens);
    }
}

#[test]
fn ban_eos_reaches_max_tokens() {
    let model = load();
    let p = prompt(&model, "Say hi.");
    let params = greedy(24).with_ban_eos(true);
    let out = model
        .generate(&p, &params, &mut |_| Flow::Continue)
        .expect("generates");
    assert_eq!(out.tokens(), 24);
    assert_eq!(out.stop(), GenStop::MaxTokens);
    assert!(out.metrics().decode_tokens_per_second() > 0.0);
}

#[test]
fn cancel_stops_within_one_token() {
    let model = load();
    let p = prompt(&model, "Write a long story.");
    let cancel = Arc::new(AtomicBool::new(false));
    let params = greedy(64)
        .with_ban_eos(true)
        .with_cancel(Some(cancel.clone()));
    let mut seen = 0;
    let out = model
        .generate(&p, &params, &mut |_| {
            seen += 1;
            if seen == 3 {
                cancel.store(true, Ordering::SeqCst);
            }
            Flow::Continue
        })
        .expect("generates");
    assert_eq!(out.stop(), GenStop::Cancelled);
    assert!(out.tokens() <= 4, "stopped after {} tokens", out.tokens());
}

#[test]
fn prefix_reuse_skips_shared_tokens() {
    let model = load();
    let first = prompt(&model, "Name three colours.");
    model
        .generate(&first, &greedy(2), &mut |_| Flow::Continue)
        .expect("first call");
    let mut second = first.clone();
    second.extend(
        model
            .tokenize(" And three animals.", false)
            .expect("tokenizes"),
    );
    let out = model
        .generate(&second, &greedy(2), &mut |_| Flow::Continue)
        .expect("second call");
    assert!(
        out.metrics().reused_prefix_tokens() >= first.len() - 1,
        "reused {} of {}",
        out.metrics().reused_prefix_tokens(),
        first.len()
    );
}

#[test]
fn reset_clears_the_cache() {
    let model = load();
    let p = prompt(&model, "Hello.");
    model
        .generate(&p, &greedy(2), &mut |_| Flow::Continue)
        .expect("first call");
    model.reset();
    let out = model
        .generate(&p, &greedy(2), &mut |_| Flow::Continue)
        .expect("after reset");
    assert_eq!(out.metrics().reused_prefix_tokens(), 0);
}

#[test]
fn grammar_constrains_output() {
    let model = load();
    let p = prompt(&model, "Is the sky blue? Answer yes or no.");
    let grammar = Grammar::new("root ::= \"yes\" | \"no\"", "root").expect("grammar");
    for seed in 0..5 {
        let params = GenParams::default()
            .with_seed(seed)
            .with_max_tokens(8)
            .with_grammar(Some(grammar.clone()));
        let out = model
            .generate(&p, &params, &mut |_| Flow::Continue)
            .expect("generates");
        assert!(
            out.text() == "yes" || out.text() == "no",
            "seed {seed}: {:?}",
            out.text()
        );
    }
}

#[test]
fn nul_in_path_or_prompt_is_refused() {
    let err = LlamaModel::load(
        std::path::Path::new("model\0.gguf"),
        LlamaSettings::default(),
    )
    .expect_err("NUL path");
    assert!(matches!(err, LocalError::Load(_)), "{err:?}");
    let model = load();
    let err = model.tokenize("a\0b", false).expect_err("NUL text");
    assert!(matches!(err, LocalError::Tokenize(_)), "{err:?}");
    let err = model
        .render_chat(&[ChatMessage::new(ChatRole::User, "a\0b")], true)
        .expect_err("NUL message");
    assert!(matches!(err, LocalError::Tokenize(_)), "{err:?}");
}

#[test]
fn non_gguf_file_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("fake.gguf");
    let mut f = std::fs::File::create(&path).expect("create");
    f.write_all(b"NOTGGUF and more bytes").expect("write");
    drop(f);
    let err = LlamaModel::load(&path, LlamaSettings::default()).expect_err("not GGUF");
    assert!(matches!(err, LocalError::Load(_)), "{err:?}");
    let err = LlamaModel::load(dir.path(), LlamaSettings::default()).expect_err("a folder");
    assert!(matches!(err, LocalError::Load(_)), "{err:?}");
}

#[test]
fn llama_defaults_are_the_safe_values() {
    let s = LlamaSettings::default();
    assert_eq!(s.context(), 4_096);
    assert!(s.mmap());
    assert!(!s.embeddings());
    assert!((1..=8).contains(&s.threads()));
}
