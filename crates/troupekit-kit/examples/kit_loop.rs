use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};
use troupekit_graph::testing::{text_reply, FnModel};
use troupekit_kit::Agent;
use troupekit_providers::Model;

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let built = Agent::new("greeter")
        .instructions("Greet the user by name.")
        .model("fake/m")
        .model_client(model())
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
    report("kit loop, tracing off", samples);
    Ok(())
}
