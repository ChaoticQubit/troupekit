use crate::lit::quote_str;
use crate::tokens::{Delim, Spacing, SpanId, Tok};

/// Lexes Rust-like source `src` into `Tok`s for parser tests.
/// Handles idents (raw too), punctuation, string, raw string and number
/// literals, nested groups, lifetimes, and `///` as `#[doc = "..."]`.
/// Panics on input it cannot lex, which is a test bug.
pub fn lex(src: &str) -> Vec<Tok> {
    let chars: Vec<char> = src.chars().collect();
    let mut pos = 0;
    let mut next_id = 0u32;
    let toks = group(&chars, &mut pos, &mut next_id, None);
    assert_eq!(pos, chars.len(), "unbalanced delimiters in test source");
    toks
}

fn id(next: &mut u32) -> SpanId {
    *next += 1;
    SpanId(*next - 1)
}

fn group(c: &[char], pos: &mut usize, next: &mut u32, close: Option<char>) -> Vec<Tok> {
    let mut out = Vec::new();
    while *pos < c.len() {
        let ch = c[*pos];
        if Some(ch) == close {
            *pos += 1;
            return out;
        }
        if ch.is_whitespace() {
            *pos += 1;
        } else if c[*pos..].starts_with(&['/', '/', '/']) {
            let start = *pos + 3;
            let end = c[start..]
                .iter()
                .position(|&x| x == '\n')
                .map_or(c.len(), |n| start + n);
            let text: String = c[start..end].iter().collect();
            *pos = end;
            let hash = id(next);
            let g = id(next);
            out.push(Tok::Punct('#', Spacing::Alone, hash));
            out.push(Tok::Group(
                Delim::Bracket,
                vec![
                    Tok::Ident("doc".into(), id(next)),
                    Tok::Punct('=', Spacing::Alone, id(next)),
                    Tok::Lit(quote_str(&text), id(next)),
                ],
                g,
            ));
        } else if let Some((delim, closing)) = match ch {
            '(' => Some((Delim::Paren, ')')),
            '[' => Some((Delim::Bracket, ']')),
            '{' => Some((Delim::Brace, '}')),
            _ => None,
        } {
            *pos += 1;
            let g = id(next);
            let inner = group(c, pos, next, Some(closing));
            out.push(Tok::Group(delim, inner, g));
        } else if ch == '"'
            || (ch == 'r' && matches!(c.get(*pos + 1), Some('"') | Some('#')) && raw_start(c, *pos))
        {
            let start = *pos;
            *pos = string_end(c, *pos);
            out.push(Tok::Lit(c[start..*pos].iter().collect(), id(next)));
        } else if ch.is_ascii_digit() {
            let start = *pos;
            while *pos < c.len()
                && (c[*pos].is_ascii_alphanumeric() || c[*pos] == '_' || c[*pos] == '.')
            {
                *pos += 1;
            }
            out.push(Tok::Lit(c[start..*pos].iter().collect(), id(next)));
        } else if ch.is_alphabetic() || ch == '_' {
            let start = *pos;
            if ch == 'r' && c.get(*pos + 1) == Some(&'#') {
                *pos += 2;
            }
            while *pos < c.len() && (c[*pos].is_alphanumeric() || c[*pos] == '_') {
                *pos += 1;
            }
            out.push(Tok::Ident(c[start..*pos].iter().collect(), id(next)));
        } else {
            *pos += 1;
            let joint = c
                .get(*pos)
                .is_some_and(|n| n.is_ascii_punctuation() && !"()[]{}\"".contains(*n));
            let joint = joint || (ch == '\'' && c.get(*pos).is_some_and(|n| n.is_alphabetic()));
            let spacing = if joint {
                Spacing::Joint
            } else {
                Spacing::Alone
            };
            out.push(Tok::Punct(ch, spacing, id(next)));
        }
    }
    assert!(close.is_none(), "unclosed group in test source");
    out
}

fn raw_start(c: &[char], pos: usize) -> bool {
    let hashes = c[pos + 1..].iter().take_while(|&&x| x == '#').count();
    c.get(pos + 1 + hashes) == Some(&'"')
}

fn string_end(c: &[char], start: usize) -> usize {
    if c[start] == 'r' {
        let hashes = c[start + 1..].iter().take_while(|&&x| x == '#').count();
        let mut pos = start + 2 + hashes;
        loop {
            assert!(pos < c.len(), "unclosed raw string in test source");
            if c[pos] == '"'
                && c[pos + 1..]
                    .iter()
                    .take(hashes)
                    .filter(|&&x| x == '#')
                    .count()
                    == hashes
            {
                return pos + 1 + hashes;
            }
            pos += 1;
        }
    }
    let mut pos = start + 1;
    loop {
        assert!(pos < c.len(), "unclosed string in test source");
        match c[pos] {
            '\\' => pos += 2,
            '"' => return pos + 1,
            _ => pos += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(toks: &[Tok]) -> String {
        toks.iter()
            .map(|t| match t {
                Tok::Ident(s, _) | Tok::Lit(s, _) => s.clone(),
                Tok::Punct(c, _, _) => c.to_string(),
                Tok::Group(d, inner, _) => {
                    let (o, c) = match d {
                        Delim::Paren => ("(", ")"),
                        Delim::Bracket => ("[", "]"),
                        Delim::Brace => ("{", "}"),
                        Delim::None => ("<", ">"),
                    };
                    format!("{o}{}{c}", shape(inner))
                }
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn lexer_groups_nest() {
        let toks =
            lex("/// Doc.\npub fn r#f(a: Vec<(i64, [u8; 2])>) -> String { \"x\\\"y\" r#\"q\"# }");
        assert_eq!(
            shape(&toks),
            "# [doc = \" Doc.\"] pub fn r#f (a : Vec < (i64 , [u8 ; 2]) >) - > String {\"x\\\"y\" r#\"q\"#}"
        );
        let Tok::Group(Delim::Paren, params, _) = &toks[5] else {
            panic!("params group");
        };
        assert!(params[0].is_ident("a"));
        assert!(matches!(toks[6], Tok::Punct('-', Spacing::Joint, _)));
        let mut ids: Vec<u32> = Vec::new();
        fn walk(t: &[Tok], ids: &mut Vec<u32>) {
            for x in t {
                ids.push(x.span().0);
                if let Tok::Group(_, inner, _) = x {
                    walk(inner, ids);
                }
            }
        }
        walk(&toks, &mut ids);
        let count = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), count, "span ids are unique");
    }
}
