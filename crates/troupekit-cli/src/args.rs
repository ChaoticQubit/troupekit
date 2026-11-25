use std::path::PathBuf;

/// Usage line printed for any bad call, with exit code 2.
/// Lists every flag; the key is read from the variable named by
/// `--key-env`, never from a flag.
pub const USAGE: &str = "usage: troupekit ingest <dir> [--db FILE] [--index NAME] [--scope NAME] --embedder SELECTOR [--catalog FILE] [--key-env VAR] [--chunk-tokens N] [--overlap-tokens N] [--kind brute|hnsw] [--quant f32|f16|i8]";

const MAX_DIMS: usize = 65_536;
const MAX_TOKENS: usize = 1_000_000;

/// Which embedder the ingest uses.
/// `Hashing` is the offline embedder with its dimension; `Catalog`
/// names a `vendor/model` embedding row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EmbedderChoice {
    Hashing(usize),
    Catalog(String),
}

/// Index kind chosen with `--kind`. Default `Brute` (S41).
/// `Hnsw` uses the default HNSW params.
/// Mapped to the index crate's `IndexKind` by the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Brute,
    Hnsw,
}

/// In-memory vector format chosen with `--quant`. Default `F32` (S41).
/// The file always keeps f32; this only changes memory.
/// Mapped to the index crate's `Quantization` by the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quant {
    F32,
    F16,
    I8,
}

/// A parsed `troupekit ingest` call with every default applied.
/// Built only by `parse`; the runner reads the fields.
/// `chunk_tokens` and `overlap_tokens` are `None` for the library default.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IngestArgs {
    pub dir: PathBuf,
    pub db: PathBuf,
    pub index: String,
    pub scope: String,
    pub embedder: EmbedderChoice,
    pub catalog: Option<PathBuf>,
    pub key_env: String,
    pub chunk_tokens: Option<usize>,
    pub overlap_tokens: Option<usize>,
    pub kind: Kind,
    pub quant: Quant,
}

fn number(flag: &str, value: &str, max: usize) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if n <= max => Ok(n),
        _ => Err(format!("{flag} needs a whole number up to {max}, got {value:?}")),
    }
}

fn embedder(value: &str) -> Result<EmbedderChoice, String> {
    if let Some(dims) = value.strip_prefix("hashing:") {
        let d = number("--embedder hashing:<dims>", dims, MAX_DIMS)?;
        if d == 0 {
            return Err("--embedder hashing:<dims> needs dims above 0".into());
        }
        return Ok(EmbedderChoice::Hashing(d));
    }
    match value.split_once('/') {
        Some((v, m)) if !v.is_empty() && !m.is_empty() => Ok(EmbedderChoice::Catalog(value.to_owned())),
        _ => Err(format!("--embedder must be vendor/model or hashing:<dims>, got {value:?}")),
    }
}

/// Parses the arguments after the program name.
/// Input: e.g. `["ingest", "docs", "--embedder", "hashing:256"]`.
/// Output: the call with defaults, or a message for exit code 2.
pub fn parse(args: &[String]) -> Result<IngestArgs, String> {
    let mut it = args.iter();
    match it.next().map(String::as_str) {
        Some("ingest") => {}
        Some(other) => return Err(format!("unknown command {other:?}")),
        None => return Err("missing command".into()),
    }
    let mut dir = None;
    let mut db = None;
    let mut index = None;
    let mut scope = None;
    let mut chosen = None;
    let mut catalog = None;
    let mut key_env = None;
    let mut chunk_tokens = None;
    let mut overlap_tokens = None;
    let mut kind = None;
    let mut quant = None;
    while let Some(arg) = it.next() {
        if !arg.starts_with("--") {
            if dir.replace(PathBuf::from(arg)).is_some() {
                return Err(format!("unexpected argument {arg:?}"));
            }
            continue;
        }
        let value = it.next().ok_or_else(|| format!("{arg} needs a value"))?;
        let repeated = match arg.as_str() {
            "--db" => db.replace(PathBuf::from(value)).is_some(),
            "--index" => index.replace(value.clone()).is_some(),
            "--scope" => scope.replace(value.clone()).is_some(),
            "--embedder" => chosen.replace(embedder(value)?).is_some(),
            "--catalog" => catalog.replace(PathBuf::from(value)).is_some(),
            "--key-env" => key_env.replace(value.clone()).is_some(),
            "--chunk-tokens" => {
                let n = number(arg, value, MAX_TOKENS)?;
                if n == 0 {
                    return Err("--chunk-tokens needs a number above 0".into());
                }
                chunk_tokens.replace(n).is_some()
            }
            "--overlap-tokens" => overlap_tokens.replace(number(arg, value, MAX_TOKENS)?).is_some(),
            "--kind" => kind
                .replace(match value.as_str() {
                    "brute" => Kind::Brute,
                    "hnsw" => Kind::Hnsw,
                    _ => return Err(format!("--kind must be brute or hnsw, got {value:?}")),
                })
                .is_some(),
            "--quant" => quant
                .replace(match value.as_str() {
                    "f32" => Quant::F32,
                    "f16" => Quant::F16,
                    "i8" => Quant::I8,
                    _ => return Err(format!("--quant must be f32, f16 or i8, got {value:?}")),
                })
                .is_some(),
            _ => return Err(format!("unknown flag {arg}")),
        };
        if repeated {
            return Err(format!("{arg} given twice"));
        }
    }
    let key_env = key_env.unwrap_or_else(|| "TROUPEKIT_EMBED_KEY".into());
    if key_env.is_empty() {
        return Err("--key-env needs a variable name".into());
    }
    Ok(IngestArgs {
        dir: dir.ok_or("missing <dir>")?,
        db: db.unwrap_or_else(|| PathBuf::from("./troupekit.db")),
        index: index.unwrap_or_else(|| "docs".into()),
        scope: scope.unwrap_or_else(|| "docs".into()),
        embedder: chosen.ok_or("missing --embedder")?,
        catalog,
        key_env,
        chunk_tokens,
        overlap_tokens,
        kind: kind.unwrap_or(Kind::Brute),
        quant: quant.unwrap_or(Quant::F32),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn parses_all_flags_and_defaults() {
        let a = parse(&args(&["ingest", "docs", "--embedder", "hashing:256"])).unwrap();
        assert_eq!(
            a,
            IngestArgs {
                dir: "docs".into(),
                db: "./troupekit.db".into(),
                index: "docs".into(),
                scope: "docs".into(),
                embedder: EmbedderChoice::Hashing(256),
                catalog: None,
                key_env: "TROUPEKIT_EMBED_KEY".into(),
                chunk_tokens: None,
                overlap_tokens: None,
                kind: Kind::Brute,
                quant: Quant::F32,
            }
        );
        let a = parse(&args(&[
            "ingest", "--db", "x.db", "--index", "kb", "--scope", "s", "--embedder", "openai/text-embedding-3-small", "--catalog", "c.toml", "--key-env", "MY_KEY", "--chunk-tokens", "300", "--overlap-tokens", "0", "--kind", "hnsw", "--quant", "i8", "folder",
        ]))
        .unwrap();
        assert_eq!(a.dir, PathBuf::from("folder"));
        assert_eq!(a.db, PathBuf::from("x.db"));
        assert_eq!((a.index.as_str(), a.scope.as_str(), a.key_env.as_str()), ("kb", "s", "MY_KEY"));
        assert_eq!(a.embedder, EmbedderChoice::Catalog("openai/text-embedding-3-small".into()));
        assert_eq!(a.catalog, Some(PathBuf::from("c.toml")));
        assert_eq!((a.chunk_tokens, a.overlap_tokens), (Some(300), Some(0)));
        assert_eq!((a.kind, a.quant), (Kind::Hnsw, Quant::I8));
    }

    #[test]
    fn rejects_unknown_flag_and_bad_numbers() {
        for bad in [
            vec![],
            vec!["serve"],
            vec!["ingest", "--embedder", "hashing:8"],
            vec!["ingest", "d"],
            vec!["ingest", "d", "e", "--embedder", "hashing:8"],
            vec!["ingest", "d", "--embedder", "hashing:8", "--colour", "red"],
            vec!["ingest", "d", "--embedder"],
            vec!["ingest", "d", "--embedder", "hashing:0"],
            vec!["ingest", "d", "--embedder", "hashing:65537"],
            vec!["ingest", "d", "--embedder", "hashing:x"],
            vec!["ingest", "d", "--embedder", "nomodel"],
            vec!["ingest", "d", "--embedder", "hashing:8", "--chunk-tokens", "0"],
            vec!["ingest", "d", "--embedder", "hashing:8", "--chunk-tokens", "-5"],
            vec!["ingest", "d", "--embedder", "hashing:8", "--overlap-tokens", "1.5"],
            vec!["ingest", "d", "--embedder", "hashing:8", "--kind", "flat"],
            vec!["ingest", "d", "--embedder", "hashing:8", "--quant", "f64"],
            vec!["ingest", "d", "--embedder", "hashing:8", "--db", "a", "--db", "b"],
            vec!["ingest", "d", "--embedder", "hashing:8", "--key-env", ""],
        ] {
            assert!(parse(&args(&bad)).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn key_flag_does_not_exist() {
        for flag in ["--key", "--api-key", "--token"] {
            let r = parse(&args(&["ingest", "d", "--embedder", "hashing:8", flag, "sk-abc"]));
            let err = r.unwrap_err();
            assert!(err.contains("unknown flag"), "{err}");
            assert!(!err.contains("sk-abc"), "{err}");
        }
        assert!(!USAGE.contains("--key ") && !USAGE.contains("--api-key"));
    }
}
