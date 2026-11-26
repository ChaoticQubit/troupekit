use std::path::Path;
use troupekit_ffi::{ErrorCode, TroupekitError};
use troupekit_ffi_core as neutral;

const RENAMES: &[(&str, &str)] = &[
    ("FfiError", "TroupekitError"),
    ("FfiEvent", "Event"),
    ("FfiEventKind", "EventKind"),
    ("FfiUsage", "Usage"),
    ("FfiPause", "Pause"),
];
const FREEZE_NAMES_MIN: usize = 50;

/// Every UniFFI error code has the number of the neutral code of the same name.
/// Input: the 19 code pairs of plan section 6.1. Output: passes when each pair
/// has equal numbers and an `FfiError` keeps code, message and run id.
#[test]
fn error_code_survives_conversion() {
    let pairs = [
        (ErrorCode::Config, neutral::ErrorCode::Config),
        (ErrorCode::MissingKey, neutral::ErrorCode::MissingKey),
        (ErrorCode::Catalog, neutral::ErrorCode::Catalog),
        (ErrorCode::Persona, neutral::ErrorCode::Persona),
        (ErrorCode::Agent, neutral::ErrorCode::Agent),
        (ErrorCode::Tool, neutral::ErrorCode::Tool),
        (ErrorCode::Workflow, neutral::ErrorCode::Workflow),
        (ErrorCode::Graph, neutral::ErrorCode::Graph),
        (ErrorCode::Store, neutral::ErrorCode::Store),
        (ErrorCode::Context, neutral::ErrorCode::Context),
        (ErrorCode::Failed, neutral::ErrorCode::Failed),
        (ErrorCode::Paused, neutral::ErrorCode::Paused),
        (ErrorCode::Retries, neutral::ErrorCode::Retries),
        (ErrorCode::Cancelled, neutral::ErrorCode::Cancelled),
        (ErrorCode::HostCallback, neutral::ErrorCode::HostCallback),
        (
            ErrorCode::InvalidArgument,
            neutral::ErrorCode::InvalidArgument,
        ),
        (ErrorCode::Unsupported, neutral::ErrorCode::Unsupported),
        (ErrorCode::Panic, neutral::ErrorCode::Panic),
        (ErrorCode::Internal, neutral::ErrorCode::Internal),
    ];
    assert_eq!(pairs.len(), 19);
    for (mirror, code) in pairs {
        assert_eq!(mirror as u32, code as u32);
        let mut error = neutral::FfiError::new(code, "boom");
        error.run_id = Some("run-1".to_owned());
        let TroupekitError::Failure {
            code: number,
            message,
            run_id,
        } = TroupekitError::from(error);
        assert_eq!(number, code as u32);
        assert_eq!(message, "boom");
        assert_eq!(run_id.as_deref(), Some("run-1"));
    }
}

/// Every name in the freeze list is exported by the UniFFI adapter.
/// Input: `troupekit-ffi-core/contract/freeze.txt` and the adapter sources.
/// Output: passes when each `Type` has a declaration and each `Type::method`
/// a `fn method(` inside an `impl Type` block, after the Q3 renames.
#[test]
fn freeze_names_match_adapters() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let freeze = std::fs::read_to_string(root.join("../troupekit-ffi-core/contract/freeze.txt"))
        .expect("freeze.txt is readable");
    let source = adapter_source(&root.join("src"));
    let names: Vec<&str> = freeze
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    assert!(names.len() >= FREEZE_NAMES_MIN);
    let missing: Vec<&str> = names
        .iter()
        .copied()
        .filter(|name| !exported(&source, name))
        .collect();
    assert!(
        missing.is_empty(),
        "not exported by troupekit-ffi: {missing:?}"
    );
}

/// Reads every `.rs` file directly under `dir` into one string.
/// Input: the adapter `src` directory. Output: the joined sources, each file
/// starting on a new line; fails the test when a file cannot be read.
fn adapter_source(dir: &Path) -> String {
    let mut out = String::new();
    for entry in std::fs::read_dir(dir).expect("src is readable") {
        let path = entry.expect("entry is readable").path();
        if path.extension().is_some_and(|e| e == "rs") {
            out.push('\n');
            out.push_str(&std::fs::read_to_string(&path).expect("source is readable"));
        }
    }
    assert!(!out.is_empty());
    out
}

/// Tells whether the adapter source declares one freeze name.
/// Input: the joined source and a neutral name, `Type` or `Type::method`.
/// Output: true when the declaration or the method is found.
fn exported(source: &str, name: &str) -> bool {
    match name.split_once("::") {
        Some((ty, method)) => impl_blocks(source, renamed(ty))
            .iter()
            .any(|block| block.contains(&format!("fn {method}("))),
        None => {
            let ty = renamed(name);
            ["struct", "enum", "trait"]
                .iter()
                .any(|kind| declares(source, &format!("{kind} {ty}")))
                || source.contains(&format!("fn {ty}("))
        }
    }
}

/// Tells whether `head` appears followed by a delimiter that ends the name.
/// Input: the source and `kind Name`. Output: true on a whole-name match.
fn declares(source: &str, head: &str) -> bool {
    [" ", "(", ":", "{"]
        .iter()
        .any(|end| source.contains(&format!("{head}{end}")))
}

/// Returns the text of each `impl Type {` block, up to the next `impl`.
/// Input: the source and the adapter type name. Output: the blocks found.
fn impl_blocks<'a>(source: &'a str, ty: &str) -> Vec<&'a str> {
    let head = format!("impl {ty} {{");
    let mut blocks = Vec::new();
    let mut rest = source;
    while let Some(start) = rest.find(&head) {
        let body = &rest[start + head.len()..];
        let end = body.find("\nimpl ").unwrap_or(body.len());
        blocks.push(&body[..end]);
        rest = &body[end..];
    }
    blocks
}

/// Maps a neutral type name to its binding name (freeze open question Q3).
/// Input: a neutral name. Output: the adapter name, unchanged when not renamed.
fn renamed(name: &str) -> &str {
    RENAMES
        .iter()
        .find(|(from, _)| *from == name)
        .map_or(name, |(_, to)| to)
}
