/// A parsed JSON value. Integers that fit stay exact in `Int` or `UInt`.
/// Objects keep their keys in file order, duplicates included.
#[derive(Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

/// Why a line did not parse: bad syntax, or nesting past the depth limit.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ParseError {
    Syntax,
    Depth,
}

/// Parses one JSON text whose arrays and objects nest at most `depth_max`
/// levels deep. Input: the text and limit. Output: the value or the error.
/// Never panics; recursion depth is bounded by `depth_max`.
pub(crate) fn parse(text: &str, depth_max: usize) -> Result<Json, ParseError> {
    assert!(depth_max > 0);
    let mut parser = Parser {
        text,
        bytes: text.as_bytes(),
        pos: 0,
        depth_max,
    };
    let value = parser.value(1)?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(ParseError::Syntax);
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
    depth_max: usize,
}

impl Parser<'_> {
    fn skip_ws(&mut self) {
        while matches!(self.bytes.get(self.pos), Some(b' ' | b'\t' | b'\r' | b'\n')) {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Result<u8, ParseError> {
        self.bytes.get(self.pos).copied().ok_or(ParseError::Syntax)
    }

    fn expect(&mut self, byte: u8) -> Result<(), ParseError> {
        if self.peek()? != byte {
            return Err(ParseError::Syntax);
        }
        self.pos += 1;
        Ok(())
    }

    fn value(&mut self, depth: usize) -> Result<Json, ParseError> {
        self.skip_ws();
        match self.peek()? {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            b'"' => self.string().map(Json::Str),
            b't' => self.literal("true", Json::Bool(true)),
            b'f' => self.literal("false", Json::Bool(false)),
            b'n' => self.literal("null", Json::Null),
            b'-' | b'0'..=b'9' => self.number(),
            _ => Err(ParseError::Syntax),
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, ParseError> {
        if !self.bytes[self.pos..].starts_with(word.as_bytes()) {
            return Err(ParseError::Syntax);
        }
        self.pos += word.len();
        Ok(value)
    }

    fn object(&mut self, depth: usize) -> Result<Json, ParseError> {
        if depth > self.depth_max {
            return Err(ParseError::Depth);
        }
        self.expect(b'{')?;
        let mut fields = Vec::new();
        self.skip_ws();
        if self.peek()? == b'}' {
            self.pos += 1;
            return Ok(Json::Obj(fields));
        }
        loop {
            self.skip_ws();
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            fields.push((key, self.value(depth + 1)?));
            self.skip_ws();
            match self.peek()? {
                b',' => self.pos += 1,
                b'}' => break,
                _ => return Err(ParseError::Syntax),
            }
        }
        self.pos += 1;
        Ok(Json::Obj(fields))
    }

    fn array(&mut self, depth: usize) -> Result<Json, ParseError> {
        if depth > self.depth_max {
            return Err(ParseError::Depth);
        }
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek()? == b']' {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek()? {
                b',' => self.pos += 1,
                b']' => break,
                _ => return Err(ParseError::Syntax),
            }
        }
        self.pos += 1;
        Ok(Json::Arr(items))
    }

    fn string(&mut self) -> Result<String, ParseError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            let run_start = self.pos;
            while let Some(&b) = self.bytes.get(self.pos) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            out.push_str(&self.text[run_start..self.pos]);
            match self.peek()? {
                b'"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                b'\\' => {
                    self.pos += 1;
                    out.push(self.escape()?);
                }
                _ => return Err(ParseError::Syntax),
            }
        }
    }

    fn escape(&mut self) -> Result<char, ParseError> {
        let byte = self.peek()?;
        self.pos += 1;
        Ok(match byte {
            b'"' => '"',
            b'\\' => '\\',
            b'/' => '/',
            b'b' => '\u{8}',
            b'f' => '\u{c}',
            b'n' => '\n',
            b'r' => '\r',
            b't' => '\t',
            b'u' => return self.unicode(),
            _ => return Err(ParseError::Syntax),
        })
    }

    fn unicode(&mut self) -> Result<char, ParseError> {
        let high = self.hex4()?;
        if !(0xD800..0xDC00).contains(&high) {
            return char::from_u32(high).ok_or(ParseError::Syntax);
        }
        if !self.bytes[self.pos..].starts_with(b"\\u") {
            return Err(ParseError::Syntax);
        }
        self.pos += 2;
        let low = self.hex4()?;
        if !(0xDC00..0xE000).contains(&low) {
            return Err(ParseError::Syntax);
        }
        let code = 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00);
        char::from_u32(code).ok_or(ParseError::Syntax)
    }

    fn hex4(&mut self) -> Result<u32, ParseError> {
        let digits = self
            .text
            .get(self.pos..self.pos + 4)
            .ok_or(ParseError::Syntax)?;
        if !digits.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(ParseError::Syntax);
        }
        self.pos += 4;
        u32::from_str_radix(digits, 16).map_err(|_| ParseError::Syntax)
    }

    fn number(&mut self) -> Result<Json, ParseError> {
        let start = self.pos;
        while matches!(
            self.bytes.get(self.pos),
            Some(b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9')
        ) {
            self.pos += 1;
        }
        let text = &self.text[start..self.pos];
        assert!(!text.is_empty());
        if !text.contains(['.', 'e', 'E']) {
            if let Ok(n) = text.parse::<u64>() {
                return Ok(Json::UInt(n));
            }
            if let Ok(n) = text.parse::<i64>() {
                return Ok(Json::Int(n));
            }
        }
        match text.parse::<f64>() {
            Ok(n) if n.is_finite() => Ok(Json::Float(n)),
            _ => Err(ParseError::Syntax),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars_strings_and_nesting() {
        let value = parse(
            r#"{"a":[1,-2,1.5,true,null],"s":"x\"\u00e9\ud83d\ude00"}"#,
            8,
        )
        .unwrap();
        let Json::Obj(fields) = value else {
            panic!("not an object")
        };
        assert_eq!(
            fields[1],
            ("s".to_owned(), Json::Str("x\"\u{e9}\u{1f600}".to_owned()))
        );
        assert_eq!(
            fields[0].1,
            Json::Arr(vec![
                Json::UInt(1),
                Json::Int(-2),
                Json::Float(1.5),
                Json::Bool(true),
                Json::Null
            ])
        );
    }

    #[test]
    fn refuses_bad_input() {
        for bad in [
            "",
            "{",
            "{\"a\":}",
            "[1,]",
            "\"\\ud800\"",
            "\"a\nb\"",
            "1 2",
            "nan",
            "-",
        ] {
            assert_eq!(parse(bad, 8), Err(ParseError::Syntax), "{bad:?}");
        }
        assert_eq!(parse("[[[1]]]", 2), Err(ParseError::Depth));
        assert!(parse("[[1]]", 2).is_ok());
    }
}
