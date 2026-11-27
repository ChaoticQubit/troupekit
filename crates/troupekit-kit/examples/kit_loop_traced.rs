use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};
use troupekit_graph::testing::{text_reply, FnModel};
use troupekit_kit::Agent;
use troupekit_providers::Model;
use troupekit_trace::{JsonlSink, MemorySink, TraceSink};

const RUNS: usize = 10_000;
const WARMUP: usize = 200;

fn model() -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", |req| {
        let asked = req
            .messages
            .last()
            .map(|m| m.text_content())
            .unwrap_or_default();
        text_reply(&format!("Hello! You said: {asked}"), 10)
    }))
}

fn report(label: &str, mut samples: Vec<Duration>) {
    assert_eq!(samples.len(), RUNS);
    samples.sort_unstable();
    let median = samples[RUNS / 2];
    let p99 = samples[RUNS * 99 / 100];
    assert!(median <= p99);
    println!(
        "{label}: runs {RUNS}, median_us {}, p99_us {}",
        median.as_micros(),
        p99.as_micros()
    );
}

async fn measure(label: &str, sink: Arc<dyn TraceSink>) -> Result<(), Box<dyn Error>> {
    let built = Agent::new("greeter")
        .instructions("Greet the user by name.")
        .model("fake/m")
        .model_client(model())
        .tracing(true)
        .trace_to(sink)
        .build()
        .await?;
    let mut samples = Vec::with_capacity(RUNS);
    for i in 0..WARMUP + RUNS {
        let started = Instant::now();
        built.run("I am Ada").await?;
        if i >= WARMUP {
            samples.push(started.elapsed());
        }
    }
    report(label, samples);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let dir = std::env::temp_dir().join(format!("kit-loop-traced-{}", std::process::id()));
    let path = dir.join("loop.jsonl");
    let jsonl = Arc::new(JsonlSink::create(&path)?);
    measure("kit loop, tracing on, JsonlSink", jsonl.clone()).await?;
    println!(
        "jsonl bytes {}, errors {}",
        jsonl.bytes_written(),
        jsonl.errors()
    );
    drop(jsonl);
    std::fs::remove_dir_all(&dir)?;
    let memory = Arc::new(MemorySink::default());
    measure("kit loop, tracing on, MemorySink", memory.clone()).await?;
    println!("memory lines dropped {}", memory.dropped());
    Ok(())
}
