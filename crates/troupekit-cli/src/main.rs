//! The `troupekit` binary. 0.2 ships one command, `ingest`.
//! Exit codes: 0 success, 1 some files failed, 2 bad usage.

mod args;

use std::process::ExitCode;

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match args::parse(&argv) {
        Err(message) => {
            eprintln!("troupekit: {message}");
            eprintln!("{}", args::USAGE);
            ExitCode::from(2)
        }
        Ok(call) => {
            eprintln!("troupekit: ingest of {} is not available in this build yet", call.dir.display());
            ExitCode::from(1)
        }
    }
}
