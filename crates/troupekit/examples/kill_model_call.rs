use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use troupekit::agent::EventSink;
use troupekit::core::event::Event;
use troupekit::core::ids::RunId;
use troupekit::local::llama::{LlamaModel, LlamaSettings};
use troupekit::local::{LocalChat, LocalChatSettings};
use troupekit::trace::{CallbackSink, Trace};
use troupekit::SqliteStore;
use troupekit_kit::{Agent, BuiltAgent, Storage};

const CONTEXT: usize = 1_024;
const MAX_TOKENS: u32 = 960;
const USAGE: &str = "usage: kill_model_call start <db> <gguf> | resume <db> <gguf> <run id>";

/// Prints `RUN <id>` once, on the first event of the run, so the parent
/// process knows which run to resume. Input: run events. Output: one
/// flushed stdout line.
struct PrintRun(AtomicBool);

impl EventSink for PrintRun {
    fn emit(&self, event: Event) {
        if !self.0.swap(true, Ordering::SeqCst) {
            println!("RUN {}", event.run_id);
            std::io::stdout().flush().expect("stdout flushes");
        }
    }
}

/// Builds the agent: the test GGUF behind `LocalChat` with end of text
/// banned, so generation runs until `MAX_TOKENS`, checkpoints in SQLite
/// file `db`, and `GENERATING` printed when a local model call starts.
async fn agent(db: &Path, gguf: &Path) -> Result<BuiltAgent, Box<dyn Error>> {
    let model = LlamaModel::load(gguf, LlamaSettings::default().with_context(CONTEXT))?;
    let settings = LocalChatSettings::default().with_ban_eos(true);
    let chat = LocalChat::new(Arc::new(model), "local/test").with_settings(settings);
    let store = SqliteStore::open(db).await?;
    let said = AtomicBool::new(false);
    let sink = CallbackSink::new(move |line| {
        let local_start =
            line.contains("\"t\":\"start\"") && line.contains("\"name\":\"model.local\"");
        if local_start && !said.swap(true, Ordering::SeqCst) {
            println!("GENERATING");
            std::io::stdout().flush().expect("stdout flushes");
        }
    });
    let built = Agent::new("storyteller")
        .instructions("Write a very long story.")
        .model("local/test")
        .model_client(Arc::new(chat))
        .max_tokens(MAX_TOKENS)
        .checkpoints(Storage::custom(Arc::new(store)))
        .events(Arc::new(PrintRun(AtomicBool::new(false))))
        .trace([Trace::Models])
        .trace_to(Arc::new(sink))
        .build()
        .await?;
    Ok(built)
}

/// Runs one half of the kill test. `start` runs a fresh story (the parent
/// kills it mid call); `resume` continues the run id from its checkpoint
/// and prints `RESUMED <bytes>`. Exits non-zero on any error.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mode, db, gguf) = match args.as_slice() {
        [mode, db, gguf, ..] => (mode.as_str(), PathBuf::from(db), PathBuf::from(gguf)),
        _ => return Err(USAGE.into()),
    };
    let built = agent(&db, &gguf).await?;
    match (mode, args.get(3)) {
        ("start", None) => {
            let text = built.run("Tell the story.").await?;
            println!("DONE {}", text.len());
        }
        ("resume", Some(id)) => {
            let run_id: RunId = serde_json::from_str(&format!("\"{id}\""))?;
            let text = built.resume(run_id).await?;
            println!("RESUMED {}", text.len());
        }
        _ => return Err(USAGE.into()),
    }
    Ok(())
}
