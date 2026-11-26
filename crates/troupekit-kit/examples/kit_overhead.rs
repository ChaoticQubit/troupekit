use std::error::Error;
use std::sync::Arc;
use std::time::{Duration, Instant};
use troupekit_agent::{drive, AgentRun, Limits, Persona, SimpleIo};
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::message::{Message, Role};
use troupekit_graph::testing::{text_reply, FnModel};
use troupekit_kit::Agent;
use troupekit_providers::{Model, Price};
use troupekit_tools::ToolRegistry;

const RUNS: usize = 10_000;
const WARMUP: usize = 200;
const PERSONA: &str = r#"
name = "greeter"
display_name = "greeter"
role = "An assistant."
goal = "Answer the user's request."
instructions = "Greet the user by name."
model = "fake/m"
"#;

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

fn summary(label: &str, mut samples: Vec<Duration>) -> Duration {
    assert_eq!(samples.len(), RUNS);
    samples.sort_unstable();
    let median = samples[RUNS / 2];
    let p99 = samples[RUNS * 99 / 100];
    println!("{label}: median {median:?}, p99 {p99:?}");
    median
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let persona = Persona::parse("greeter.toml", PERSONA)?;
    let io = SimpleIo::new(ToolRegistry::new()).with_model("fake/m", model(), Price::FREE);
    let started = Instant::now();
    let built = Agent::new("greeter")
        .instructions("Greet the user by name.")
        .model("fake/m")
        .model_client(model())
        .build()
        .await?;
    println!("kit build: {:?}", started.elapsed());
    let mut direct = Vec::with_capacity(RUNS);
    let mut kit = Vec::with_capacity(RUNS);
    for i in 0..WARMUP + RUNS {
        let input = vec![Message::text(Role::User, "I am Ada")];
        let started = Instant::now();
        let mut run = AgentRun::new(&persona, input, Limits::default(), &io).await?;
        drive(&mut run, &io, &InMemCheckpoints::default()).await?;
        let direct_time = started.elapsed();
        let started = Instant::now();
        built.run("I am Ada").await?;
        let kit_time = started.elapsed();
        if i >= WARMUP {
            direct.push(direct_time);
            kit.push(kit_time);
        }
    }
    let direct_median = summary("direct AgentRun::new + drive", direct);
    let kit_median = summary("kit BuiltAgent::run", kit);
    let overhead = kit_median.saturating_sub(direct_median);
    println!("median overhead per run: {overhead:?}");
    Ok(())
}
