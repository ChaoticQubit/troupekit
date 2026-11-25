use proc_macro::{Delimiter, Group, Ident, Literal, Punct, TokenStream, TokenTree};

/// Index of one original token tree in a `Source`.
/// Parsed tokens carry it so errors and re-emitted user code keep the
/// span the user wrote.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpanId(pub u32);

/// Whether a punctuation character joins the next one, as in `::`.
/// Mirrors `proc_macro::Spacing` so the parser runs without a compiler.
/// `Joint` means the next token is punctuation written right after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spacing {
    Alone,
    Joint,
}

/// The delimiter of a token group: `()`, `[]`, `{}` or invisible.
/// Mirrors `proc_macro::Delimiter`.
/// `None` groups come from macro expansion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Delim {
    Paren,
    Bracket,
    Brace,
    None,
}

/// One token tree in the macro's own model, testable without a compiler.
/// Idents keep a `r#` prefix; literals keep their source text, quotes and
/// escapes included. Every token carries the `SpanId` of its original.
#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String, SpanId),
    Punct(char, Spacing, SpanId),
    Lit(String, SpanId),
    Group(Delim, Vec<Tok>, SpanId),
}

impl Tok {
    /// Returns the `SpanId` of the original token tree.
    /// For a group, the span covers the whole group.
    /// Used to aim errors and re-emit user code.
    pub fn span(&self) -> SpanId {
        match self {
            Tok::Ident(_, s) | Tok::Punct(_, _, s) | Tok::Lit(_, s) | Tok::Group(_, _, s) => *s,
        }
    }

    /// Reports whether this token is the ident `name`.
    /// Raw idents match only with their `r#` prefix.
    /// Returns false for every other token kind.
    pub fn is_ident(&self, name: &str) -> bool {
        matches!(self, Tok::Ident(n, _) if n == name)
    }

    /// Reports whether this token is the punctuation character `c`.
    /// Spacing is ignored.
    /// Returns false for every other token kind.
    pub fn is_punct(&self, c: char) -> bool {
        matches!(self, Tok::Punct(p, _, _) if *p == c)
    }
}

/// A macro failure: the message and the token it points at.
/// `emit_error` turns it into a `compile_error!` at that span.
/// The message is the exact text the user sees.
#[derive(Clone, Debug, PartialEq)]
pub struct Error {
    pub span: SpanId,
    pub message: String,
}

impl Error {
    /// Builds the error `message` aimed at the token with `span`.
    /// Returns the error.
    /// The message should name the fix.
    pub fn new(span: SpanId, message: impl Into<String>) -> Self {
        Error {
            span,
            message: message.into(),
        }
    }
}

/// One unit of generated code: Rust source text, an original token tree
/// re-emitted with its own span, or a delimited group of pieces.
/// Original trees keep rustc errors pointing at the user's code.
#[derive(Clone, Debug)]
pub enum Piece {
    Code(String),
    Orig(SpanId),
    Group(Delim, Vec<Piece>),
}

/// The original token trees of one macro call, indexed by `SpanId`.
/// Built by `from_proc`; read by `emit` and `emit_error`.
/// Only valid inside the macro call that built it.
pub struct Source {
    trees: Vec<TokenTree>,
}

impl Source {
    /// Starts an empty source for one macro call.
    /// Add the attribute and the item to the same source so their
    /// `SpanId`s never collide.
    pub fn new() -> Self {
        Source { trees: Vec::new() }
    }

    /// Converts the compiler tokens `stream` into `Tok`s, recording every
    /// tree, nested ones included, under its own `SpanId`.
    /// Returns the top-level tokens in order.
    pub fn add(&mut self, stream: TokenStream) -> Vec<Tok> {
        convert(stream, self)
    }
}

fn convert(stream: TokenStream, source: &mut Source) -> Vec<Tok> {
    let mut out = Vec::new();
    for tree in stream {
        let id = SpanId(u32::try_from(source.trees.len()).expect("token count fits in u32"));
        source.trees.push(tree.clone());
        let tok = match tree {
            TokenTree::Ident(i) => Tok::Ident(i.to_string(), id),
            TokenTree::Punct(p) => {
                let spacing = match p.spacing() {
                    proc_macro::Spacing::Joint => Spacing::Joint,
                    proc_macro::Spacing::Alone => Spacing::Alone,
                };
                Tok::Punct(p.as_char(), spacing, id)
            }
            TokenTree::Literal(l) => Tok::Lit(l.to_string(), id),
            TokenTree::Group(g) => {
                let delim = match g.delimiter() {
                    Delimiter::Parenthesis => Delim::Paren,
                    Delimiter::Bracket => Delim::Bracket,
                    Delimiter::Brace => Delim::Brace,
                    Delimiter::None => Delim::None,
                };
                Tok::Group(delim, convert(g.stream(), source), id)
            }
        };
        out.push(tok);
    }
    out
}

/// Assembles `pieces` into compiler tokens, re-emitting originals from
/// `source` with their spans. Panics when a `Code` piece is not valid
/// token text, which is a bug in the macro, never in user input.
pub fn emit(pieces: &[Piece], source: &Source) -> TokenStream {
    let mut out = TokenStream::new();
    for piece in pieces {
        match piece {
            Piece::Code(text) => match text.parse::<TokenStream>() {
                Ok(tokens) => out.extend(tokens),
                Err(e) => panic!("generated code is valid tokens: {text}: {e}"),
            },
            Piece::Orig(id) => out.extend([source.tree(*id).clone()]),
            Piece::Group(delim, inner) => {
                let delimiter = match delim {
                    Delim::Paren => Delimiter::Parenthesis,
                    Delim::Bracket => Delimiter::Bracket,
                    Delim::Brace => Delimiter::Brace,
                    Delim::None => Delimiter::None,
                };
                out.extend([TokenTree::Group(Group::new(delimiter, emit(inner, source)))]);
            }
        }
    }
    out
}

/// Builds `::core::compile_error! { "message" }` with every token at the
/// span of the token `error` points at, so rustc underlines the user's
/// code. Returns the tokens to output instead of the item.
pub fn emit_error(error: &Error, source: &Source) -> TokenStream {
    let span = source.tree(error.span).span();
    let punct = |c: char, joint: bool| {
        let spacing = if joint {
            proc_macro::Spacing::Joint
        } else {
            proc_macro::Spacing::Alone
        };
        let mut p = Punct::new(c, spacing);
        p.set_span(span);
        TokenTree::Punct(p)
    };
    let mut message = Literal::string(&error.message);
    message.set_span(span);
    let mut body = Group::new(
        Delimiter::Brace,
        TokenStream::from(TokenTree::Literal(message)),
    );
    body.set_span(span);
    TokenStream::from_iter([
        punct(':', true),
        punct(':', false),
        TokenTree::Ident(Ident::new("core", span)),
        punct(':', true),
        punct(':', false),
        TokenTree::Ident(Ident::new("compile_error", span)),
        punct('!', false),
        TokenTree::Group(body),
    ])
}

impl Source {
    fn tree(&self, id: SpanId) -> &TokenTree {
        let index = usize::try_from(id.0).expect("u32 fits in usize");
        assert!(index < self.trees.len(), "span ids come from this source");
        &self.trees[index]
    }
}
