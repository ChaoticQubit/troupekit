use std::fs::{File, OpenOptions};
use std::io::{BufReader, Write};
use std::path::Path;

use troupekit_trace::view::{
    read_jsonl, render_html, render_terminal, SpanTree, TermOptions, ViewLimits,
};

use crate::args::TraceArgs;

/// Runs one `troupekit trace` call: reads the file, then prints the tree
/// and timeline or writes the HTML page. `tty` says whether stdout is a
/// terminal. Output: the text to print, or a message for exit code 1.
pub fn run(call: &TraceArgs, tty: bool) -> Result<String, String> {
    let tree = read(&call.file)?;
    match &call.html {
        Some(out) => {
            write_page(out, &render_html(&tree))?;
            Ok(format!("wrote {}", out.display()))
        }
        None => {
            let options = TermOptions::default()
                .with_color(call.color && tty)
                .with_only(call.only);
            Ok(render_terminal(&tree, &options))
        }
    }
}

fn read(path: &Path) -> Result<SpanTree, String> {
    let file = File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    read_jsonl(BufReader::new(file), ViewLimits::default())
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// Writes `page` to a new file at `out`, never over an existing one.
/// The file is private (mode 0600 on Unix): traces describe user runs.
fn write_page(out: &Path, page: &str) -> Result<(), String> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(out)
        .map_err(|e| format!("cannot create {}: {e}", out.display()))?;
    file.write_all(page.as_bytes())
        .map_err(|e| format!("cannot write {}: {e}", out.display()))
}
