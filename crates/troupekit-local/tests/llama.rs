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
fn system_info_is_stable_across_threads() {
    let first = troupekit_local::llama::system_info();
    let threads: Vec<_> = (0..8)
        .map(|_| {
            std::thread::spawn(|| {
                (0..200)
                    .map(|_| troupekit_local::llama::system_info())
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for t in threads {
        for info in t.join().expect("thread") {
            assert_eq!(info, first);
        }
    }
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

#[test]
fn generate_emits_a_model_local_span() {
    use troupekit_trace::{MemorySink, Trace, TraceSet, Tracer};
    let model = load();
    let p = prompt(&model, "Hi.");
    let sink = std::sync::Arc::new(MemorySink::new(1_000));
    let tracer = Tracer::builder()
        .categories(TraceSet::none().with(Trace::Models))
        .sink(sink.clone())
        .name("local")
        .build()
        .expect("tracer");
    tracer.enter(|| {
        model
            .generate(&p, &greedy(2), &mut |_| Flow::Continue)
            .expect("generates")
    });
    drop(tracer);
    let lines = sink.lines();
    assert!(
        lines.iter().any(|l| l.contains("\"model.local\"")),
        "{lines:?}"
    );
    assert!(lines.iter().any(|l| l.contains("decode_tps")), "{lines:?}");
}

#[test]
fn reset_from_the_token_callback_does_not_deadlock() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let model = load();
        let p = prompt(&model, "Count to ten.");
        model
            .generate(&p, &greedy(8), &mut |_| {
                model.reset();
                Flow::Stop
            })
            .expect("generates");
        let again = model
            .generate(&p, &greedy(1), &mut |_| Flow::Continue)
            .expect("generates again");
        tx.send(again.metrics().reused_prefix_tokens())
            .expect("send");
    });
    let reused = rx
        .recv_timeout(std::time::Duration::from_secs(120))
        .expect("generation finished; reset in the callback must not deadlock");
    assert_eq!(reused, 0, "the reset took effect before the next call");
}

#[test]
fn end_of_text_is_left_out_of_the_confidence_mean() {
    let model = load();
    let p = prompt(&model, "Say hi.");
    let ended = model
        .generate(&p, &greedy(64).with_confidence(true), &mut |_| Flow::Continue)
        .expect("generates");
    assert_eq!(ended.stop(), GenStop::EndOfText, "pick a prompt that ends");
    assert!(ended.tokens() > 0);
    model.reset();
    let capped = model
        .generate(
            &p,
            &greedy(ended.tokens()).with_confidence(true),
            &mut |_| Flow::Continue,
        )
        .expect("generates");
    assert_eq!(capped.stop(), GenStop::MaxTokens);
    assert_eq!(capped.text(), ended.text());
    let (a, b) = (ended.mean_logprob().unwrap(), capped.mean_logprob().unwrap());
    assert!((a - b).abs() < 1e-4, "EOG counted in the mean: {a} vs {b}");
}

#[test]
fn fixed_length_overflow_is_refused_before_any_token() {
    let model = LlamaModel::load(&test_gguf(), LlamaSettings::default().with_context(64))
        .expect("test GGUF loads");
    let limit = model.info().context_length();
    let text = "one two three four five six seven eight nine ten ".repeat(20);
    let tokens = model.tokenize(&text, false).expect("tokenizes");
    let p = &tokens[..limit - 1];
    let mut calls = 0usize;
    let err = model
        .generate(p, &greedy(2).with_ban_eos(true), &mut |_| {
            calls += 1;
            Flow::Continue
        })
        .expect_err("prompt plus two tokens is past the context");
    assert!(
        matches!(err, LocalError::ContextFull { needed, limit: l } if needed == limit + 1 && l == limit),
        "{err:?}"
    );
    assert_eq!(calls, 0, "no token before the error");
    let one = model
        .generate(p, &greedy(1).with_ban_eos(true), &mut |_| Flow::Continue)
        .expect("one token still fits");
    assert_eq!(one.tokens(), 1);
}

#[test]
fn model_call_from_its_own_token_callback_is_refused_not_deadlocked() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let model = load();
        let p = prompt(&model, "Count to ten.");
        let mut nested = None;
        model
            .generate(&p, &greedy(4), &mut |_| {
                nested = Some(model.score(&p[..2]));
                Flow::Stop
            })
            .expect("generates");
        let after = model.score(&p[..2]);
        tx.send((nested, after.is_ok())).expect("send");
    });
    let (nested, after) = rx
        .recv_timeout(std::time::Duration::from_secs(120))
        .expect("a nested call must not deadlock");
    let err = nested.expect("callback ran").expect_err("nested call refused");
    assert!(matches!(err, LocalError::Unsupported(_)), "{err:?}");
    assert!(err.to_string().contains("token callback"), "{err}");
    assert!(after, "the model works again once generate returns");
}
