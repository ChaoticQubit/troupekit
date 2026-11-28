use std::path::PathBuf;
use std::process::ExitCode;
use troupekit_local::llama::{system_info, LlamaModel, LlamaSettings};
use troupekit_local::{Flow, GenParams, LocalMetrics, LocalModel};

const USAGE: &str =
    "usage: local_bench <model.gguf> [--runs N] [--prompt-tokens N] [--gen-tokens N] [--context N]";

struct Args {
    model: PathBuf,
    runs: usize,
    prompt_tokens: usize,
    gen_tokens: usize,
    context: usize,
}

struct Row {
    prefill: f64,
    decode: f64,
    ttft: f64,
    reused: usize,
}

/// Measures local generation speed for one GGUF model: a cold call and
/// a prefix-reuse call per run, fixed length with end of text banned.
/// Input: the model path and options. Output: a table of medians.
fn main() -> ExitCode {
    let args = match parse(std::env::args().skip(1).collect()) {
        Ok(a) => a,
        Err(message) => {
            eprintln!("{message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("local_bench: {message}");
            ExitCode::FAILURE
        }
    }
}

fn parse(argv: Vec<String>) -> Result<Args, String> {
    let mut it = argv.into_iter();
    let model = it.next().ok_or("missing model path")?;
    let mut args = Args {
        model: PathBuf::from(model),
        runs: 3,
        prompt_tokens: 512,
        gen_tokens: 128,
        context: 4_096,
    };
    while let Some(flag) = it.next() {
        let value = it.next().ok_or_else(|| format!("{flag} needs a value"))?;
        let n: usize = value
            .parse()
            .map_err(|_| format!("{flag} needs a number, got {value}"))?;
        match flag.as_str() {
            "--runs" if (1..=50).contains(&n) => args.runs = n,
            "--prompt-tokens" if (1..=32_768).contains(&n) => args.prompt_tokens = n,
            "--gen-tokens" if (1..=4_096).contains(&n) => args.gen_tokens = n,
            "--context" if (64..=131_072).contains(&n) => args.context = n,
            _ => {
                return Err(format!(
                    "unknown flag or value out of range: {flag} {value}"
                ))
            }
        }
    }
    Ok(args)
}

fn run(args: &Args) -> Result<(), String> {
    let settings = LlamaSettings::default().with_context(args.context);
    let threads = settings.threads();
    let model = LlamaModel::load(&args.model, settings).map_err(|e| e.to_string())?;
    let limit = model.info().context_length();
    if args.prompt_tokens + args.gen_tokens + 8 >= limit {
        return Err(format!(
            "prompt plus generation must stay below the context of {limit} tokens"
        ));
    }
    let prompt = make_prompt(&model, args.prompt_tokens)?;
    let mut extended = prompt.clone();
    extended.extend(make_prompt(&model, 8)?);
    let params = GenParams::default()
        .with_temperature(0.0)
        .with_ban_eos(true)
        .with_max_tokens(args.gen_tokens);
    let mut cold = Vec::new();
    let mut warm = Vec::new();
    let mut peak = None;
    for _ in 0..args.runs {
        model.reset();
        let first = model
            .generate(&prompt, &params, &mut |_| Flow::Continue)
            .map_err(|e| e.to_string())?;
        let second = model
            .generate(&extended, &params, &mut |_| Flow::Continue)
            .map_err(|e| e.to_string())?;
        cold.push(row(first.metrics()));
        warm.push(row(second.metrics()));
        peak = second.metrics().peak_ram_bytes().or(peak);
    }
    println!("model: {}", args.model.display());
    println!("backend: {}", system_info().trim());
    println!(
        "threads: {threads}  context: {limit}  prompt tokens: {}  generated tokens: {}  runs: {}",
        prompt.len(),
        args.gen_tokens,
        args.runs
    );
    println!(
        "peak RAM bytes: {}",
        peak.map_or("unknown".to_string(), |b| b.to_string())
    );
    println!("| call | prefill tok/s | decode tok/s | first token ms | reused tokens |");
    println!("|---|---|---|---|---|");
    for (name, rows) in [("cold", &cold), ("prefix reuse", &warm)] {
        println!(
            "| {name} | {:.1} | {:.1} | {:.1} | {} |",
            median(rows.iter().map(|r| r.prefill)),
            median(rows.iter().map(|r| r.decode)),
            median(rows.iter().map(|r| r.ttft)),
            rows.iter().map(|r| r.reused).min().unwrap_or(0)
        );
    }
    Ok(())
}

fn make_prompt(model: &LlamaModel, n: usize) -> Result<Vec<u32>, String> {
    let text = "the quick brown fox jumps over the lazy dog ".repeat(n / 4 + 4);
    let mut tokens = model.tokenize(&text, false).map_err(|e| e.to_string())?;
    if tokens.len() < n {
        return Err(format!("could not build a {n}-token prompt"));
    }
    tokens.truncate(n);
    Ok(tokens)
}

fn row(m: &LocalMetrics) -> Row {
    Row {
        prefill: m.prefill_tokens_per_second(),
        decode: m.decode_tokens_per_second(),
        ttft: m.time_to_first_token_ms(),
        reused: m.reused_prefix_tokens(),
    }
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut v: Vec<f64> = values.filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
