//! The `troupekit` binary. 0.2 ships one command, `ingest`.
//! Exit codes: 0 success, 1 some files failed, 2 bad usage.

use std::process::ExitCode;

const USAGE: &str = "usage: troupekit ingest <dir> [--db FILE] [--index NAME] [--scope NAME] --embedder SELECTOR [--catalog FILE] [--key-env VAR] [--chunk-tokens N] [--overlap-tokens N] [--kind brute|hnsw] [--quant f32|f16|i8]";

fn main() -> ExitCode {
    eprintln!("{USAGE}");
    ExitCode::from(2)
}
