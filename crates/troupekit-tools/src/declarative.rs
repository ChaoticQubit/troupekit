use crate::http::{receive, HttpRequest, Method, Transport};
use crate::registry::Toolkit;
use crate::secrets::SecretSource;
use crate::text::{fence_capped, response_text, MAX_TEXT_CHARS};
use crate::tool::{Tool, ToolContext, ToolError, ToolOutput};
use async_trait::async_trait;
use percent_encoding::{utf8_percent_encode, AsciiSet, NON_ALPHANUMERIC};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::sync::Arc;
use troupekit_core::secret::Secret;
use troupekit_core::tool::ToolSpec;
use url::Url;

const SEGMENT: &AsciiSet = &NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');
const SECRET_OPEN: &str = "${secret:";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolsFile {
    #[serde(default)]
    tool: Vec<ToolDef>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ToolDef {
    name: String,
    description: String,
    method: Method,
    url: String,
    #[serde(default)]
    headers: BTreeMap<String, String>,
    schema: Value,
}

enum Part {
    Lit(String),
    Var(String),
}

enum HeaderValue {
    Plain(String),
    Secret(Secret),
}

impl HeaderValue {
    fn expose(&self) -> &str {
        match self {
            HeaderValue::Plain(v) => v,
            HeaderValue::Secret(s) => s.expose(),
        }
    }
}

/// One HTTP endpoint described in TOML and called through a transport.
/// Path placeholders are filled from arguments, the rest go to the query
/// or a JSON body; header secrets were resolved at load time.
struct HttpTool {
    spec: ToolSpec,
    method: Method,
    template: Vec<Part>,
    headers: Vec<(String, HeaderValue)>,
    http: Arc<dyn Transport>,
}

impl std::fmt::Debug for HttpTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let names: Vec<&str> = self.headers.iter().map(|(n, _)| n.as_str()).collect();
        f.debug_struct("HttpTool")
            .field("name", &self.spec.name)
            .field("method", &self.method)
            .field("headers", &names)
            .finish()
    }
}

/// Loads declarative HTTP tools from TOML into a toolkit called `toolkit`:
/// each `[[tool]]` has `name`, `description`, `method`, `url` with `{arg}`
/// path placeholders, optional `headers` (values may hold `${secret:NAME}`,
/// read from `secrets` now) and a JSON `schema`. Returns `Config` for any
/// malformed or unsafe definition. Pass a `GuardedHttp`, never an unguarded one.
pub fn http_toolkit_from_toml(
    toolkit: &str,
    text: &str,
    secrets: &dyn SecretSource,
    http: Arc<dyn Transport>,
) -> Result<Toolkit, ToolError> {
    let tools = parse_tools(text, secrets, http)?;
    Ok(tools
        .into_iter()
        .fold(Toolkit::new(toolkit), |kit, t| kit.with(Arc::new(t))))
}

fn parse_tools(
    text: &str,
    secrets: &dyn SecretSource,
    http: Arc<dyn Transport>,
) -> Result<Vec<HttpTool>, ToolError> {
    let file: ToolsFile =
        toml::from_str(text).map_err(|e| ToolError::Config(format!("http tools: {e}")))?;
    file.tool
        .into_iter()
        .map(|def| build_tool(def, secrets, http.clone()))
        .collect()
}

fn build_tool(
    def: ToolDef,
    secrets: &dyn SecretSource,
    http: Arc<dyn Transport>,
) -> Result<HttpTool, ToolError> {
    let name = def.name.clone();
    let err = |m: String| ToolError::Config(format!("{name}: {m}"));
    let template = parse_template(&def.url).map_err(err)?;
    check_template(&template, &def.schema).map_err(err)?;
    let mut headers = Vec::with_capacity(def.headers.len());
    for (header, raw) in def.headers {
        headers.push((header, resolve_header(&raw, secrets).map_err(err)?));
    }
    Ok(HttpTool {
        spec: ToolSpec {
            name: def.name,
            description: def.description,
            schema: def.schema,
            hosted: None,
        },
        method: def.method,
        template,
        headers,
        http,
    })
}

fn parse_template(url: &str) -> Result<Vec<Part>, String> {
    let mut parts = Vec::new();
    let mut rest = url;
    while let Some(open) = rest.find('{') {
        if open > 0 {
            parts.push(Part::Lit(rest[..open].to_string()));
        }
        let close = rest[open..]
            .find('}')
            .ok_or_else(|| format!("unclosed placeholder in `{url}`"))?
            + open;
        let var = &rest[open + 1..close];
        if var.is_empty() || !var.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!("bad placeholder `{{{var}}}` in `{url}`"));
        }
        parts.push(Part::Var(var.to_string()));
        rest = &rest[close + 1..];
    }
    if rest.contains('}') {
        return Err(format!("stray `}}` in `{url}`"));
    }
    if !rest.is_empty() {
        parts.push(Part::Lit(rest.to_string()));
    }
    Ok(parts)
}

fn check_template(template: &[Part], schema: &Value) -> Result<(), String> {
    let prefix: String = template
        .iter()
        .map_while(|p| {
            if let Part::Lit(l) = p {
                Some(l.as_str())
            } else {
                None
            }
        })
        .collect();
    let has_var = template.iter().any(|p| matches!(p, Part::Var(_)));
    let path_start = prefix.find("://").map(|i| prefix[i + 3..].contains('/'));
    if has_var && path_start != Some(true) {
        return Err("placeholders may only appear after the host's first `/`".into());
    }
    let sample: String = template
        .iter()
        .map(|p| {
            if let Part::Lit(l) = p {
                l.as_str()
            } else {
                "x"
            }
        })
        .collect();
    let parsed = Url::parse(&sample).map_err(|e| format!("bad url `{sample}`: {e}"))?;
    if !matches!(parsed.scheme(), "http" | "https") || parsed.host().is_none() {
        return Err(format!("url `{sample}` must be http or https with a host"));
    }
    let required: Vec<&str> = schema
        .get("required")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    for part in template {
        if let Part::Var(v) = part {
            let declared = schema.pointer(&format!("/properties/{v}")).is_some();
            if !declared || !required.contains(&v.as_str()) {
                return Err(format!(
                    "placeholder `{v}` must be a required schema property"
                ));
            }
        }
    }
    Ok(())
}

fn resolve_header(raw: &str, secrets: &dyn SecretSource) -> Result<HeaderValue, String> {
    if !raw.contains(SECRET_OPEN) {
        return Ok(HeaderValue::Plain(raw.to_string()));
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(open) = rest.find(SECRET_OPEN) {
        out.push_str(&rest[..open]);
        let after = &rest[open + SECRET_OPEN.len()..];
        let close = after
            .find('}')
            .ok_or_else(|| "unclosed `${secret:` in header".to_string())?;
        let name = &after[..close];
        let secret = secrets
            .get(name)
            .ok_or_else(|| format!("secret `{name}` is not set"))?;
        out.push_str(secret.expose());
        rest = &after[close + 1..];
    }
    out.push_str(rest);
    Secret::new(out)
        .map(HeaderValue::Secret)
        .map_err(|e| e.to_string())
}

fn scalar(name: &str, value: &Value) -> Result<String, ToolError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        _ => Err(ToolError::InvalidArgs(format!(
            "`{name}` must be a string, number or boolean"
        ))),
    }
}

impl HttpTool {
    fn request(&self, args: Value) -> Result<HttpRequest, ToolError> {
        let Value::Object(mut rest) = args else {
            return Err(ToolError::InvalidArgs(
                "arguments must be a JSON object".into(),
            ));
        };
        let mut url = String::new();
        for part in &self.template {
            match part {
                Part::Lit(l) => url.push_str(l),
                Part::Var(v) => {
                    let value = rest
                        .remove(v)
                        .ok_or_else(|| ToolError::InvalidArgs(format!("missing `{v}`")))?;
                    let value = scalar(v, &value)?;
                    if value.is_empty() || value == "." || value == ".." {
                        return Err(ToolError::InvalidArgs(format!(
                            "`{v}` must not be empty, `.` or `..`"
                        )));
                    }
                    url.extend(utf8_percent_encode(&value, SEGMENT));
                }
            }
        }
        let mut req = HttpRequest {
            method: self.method,
            url,
            headers: Vec::with_capacity(self.headers.len() + 1),
            body: None,
            accept: Vec::new(),
        };
        req.headers.extend(
            self.headers
                .iter()
                .map(|(n, v)| (n.clone(), v.expose().to_string())),
        );
        match self.method {
            Method::Get | Method::Delete => req.url = with_query(&req.url, &rest)?,
            Method::Post | Method::Put | Method::Patch => {
                req.body =
                    Some(serde_json::to_vec(&Value::Object(rest)).expect("a JSON map serializes"));
                req.headers
                    .push(("Content-Type".into(), "application/json".into()));
            }
        }
        Ok(req)
    }
}

fn with_query(url: &str, args: &Map<String, Value>) -> Result<String, ToolError> {
    let pairs: Vec<(&String, String)> = args
        .iter()
        .filter(|(_, v)| !v.is_null())
        .map(|(k, v)| scalar(k, v).map(|s| (k, s)))
        .collect::<Result<_, _>>()?;
    if pairs.is_empty() {
        return Ok(url.to_string());
    }
    let mut parsed =
        Url::parse(url).map_err(|e| ToolError::InvalidArgs(format!("bad url: {e}")))?;
    parsed
        .query_pairs_mut()
        .extend_pairs(pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())));
    Ok(parsed.into())
}

#[async_trait]
impl Tool for HttpTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let res = receive(self.http.as_ref(), self.request(args)?).await?;
        let fenced = fence_capped(
            &res.url,
            &response_text(&res),
            MAX_TEXT_CHARS,
            res.truncated,
        );
        if (200..300).contains(&res.status) {
            Ok(ToolOutput::text(fenced))
        } else {
            Ok(ToolOutput::error(format!("HTTP {}\n{fenced}", res.status)))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::{FakeHttp, HttpError, HttpErrorKind, Method};
    use crate::registry::ToolRegistry;
    use crate::tool::{ToolContext, ToolError, ToolOutput};
    use serde_json::json;
    use std::collections::HashMap;
    use troupekit_core::secret::Secret;

    const WEATHER: &str = r#"
[[tool]]
name = "forecast"
description = "Weather forecast for a city"
method = "GET"
url = "https://api.example.com/v1/{city}/forecast"
headers = { Authorization = "Bearer ${secret:WEATHER_KEY}", Accept = "application/json" }
schema = { type = "object", properties = { city = { type = "string" }, days = { type = "integer" } }, required = ["city"] }

[[tool]]
name = "add_note"
description = "Add a note"
method = "POST"
url = "https://api.example.com/v1/notes/{id}"
schema = { type = "object", properties = { id = { type = "integer" }, note = { type = "string" } }, required = ["id", "note"] }
"#;

    fn secrets() -> HashMap<String, Secret> {
        let mut m = HashMap::new();
        m.insert(
            "WEATHER_KEY".to_string(),
            Secret::new("k-123".into()).unwrap(),
        );
        m
    }

    fn registry(fake: Arc<FakeHttp>) -> ToolRegistry {
        let mut r = ToolRegistry::new();
        r.register_toolkit(http_toolkit_from_toml("example", WEATHER, &secrets(), fake).unwrap())
            .unwrap();
        r
    }

    #[tokio::test]
    async fn get_fills_template_and_query() {
        let fake = FakeHttp::new(vec![FakeHttp::reply(
            200,
            "application/json",
            r#"{"t":21}"#,
        )]);
        let out = registry(fake.clone())
            .call(
                "forecast",
                json!({"city":"São Paulo/x","days":2}),
                &ToolContext::default(),
            )
            .await
            .unwrap();
        let req = &fake.requests()[0];
        assert_eq!(req.method, Method::Get);
        assert_eq!(
            req.url,
            "https://api.example.com/v1/S%C3%A3o%20Paulo%2Fx/forecast?days=2"
        );
        assert!(req.body.is_none());
        assert!(!out.is_error);
        assert!(
            out.content
                .starts_with("<untrusted source=\"https://api.example.com/v1/")
                && out.content.contains("{\"t\":21}"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn post_sends_leftover_args_as_json() {
        let fake = FakeHttp::new(vec![FakeHttp::reply(201, "application/json", "{}")]);
        registry(fake.clone())
            .call(
                "add_note",
                json!({"id":7,"note":"hi"}),
                &ToolContext::default(),
            )
            .await
            .unwrap();
        let req = &fake.requests()[0];
        assert_eq!(
            (req.method, req.url.as_str()),
            (Method::Post, "https://api.example.com/v1/notes/7")
        );
        assert_eq!(req.body.as_deref(), Some(br#"{"note":"hi"}"#.as_slice()));
        assert!(req
            .headers
            .contains(&("Content-Type".to_string(), "application/json".to_string())));
    }

    #[tokio::test]
    async fn secret_header_resolved_and_hidden() {
        let fake = FakeHttp::new(vec![FakeHttp::reply(200, "text/plain", "ok")]);
        registry(fake.clone())
            .call("forecast", json!({"city":"Oslo"}), &ToolContext::default())
            .await
            .unwrap();
        let req = &fake.requests()[0];
        assert!(req
            .headers
            .contains(&("Authorization".to_string(), "Bearer k-123".to_string())));
        assert_eq!(req.url, "https://api.example.com/v1/Oslo/forecast");
        let tools = parse_tools(WEATHER, &secrets(), fake).unwrap();
        let dbg = format!("{:?}", tools[0]);
        assert!(
            dbg.contains("Authorization") && !dbg.contains("k-123"),
            "{dbg}"
        );
        assert!(!serde_json::to_string(tools[0].spec())
            .unwrap()
            .contains("k-123"));
    }

    #[tokio::test]
    async fn rejects_dot_segments_and_non_scalars() {
        let fake = FakeHttp::new(vec![]);
        let tools = parse_tools(WEATHER, &secrets(), fake.clone()).unwrap();
        let ctx = ToolContext::default();
        for bad in [
            json!({"city":".."}),
            json!({"city":"."}),
            json!({"city":["a"]}),
            json!({"city":"x","days":[1]}),
            json!("x"),
        ] {
            assert!(
                matches!(
                    tools[0].call(bad.clone(), &ctx).await,
                    Err(ToolError::InvalidArgs(_))
                ),
                "{bad}"
            );
        }
        assert!(fake.requests().is_empty());
    }

    #[test]
    fn load_errors() {
        let tool = |method: &str, url: &str, required: &str, header: &str| {
            format!(
                "[[tool]]\nname = \"t\"\ndescription = \"d\"\nmethod = \"{method}\"\nurl = \"{url}\"\n{header}schema = {{ type = \"object\", properties = {{ a = {{ type = \"string\" }} }}, required = [{required}] }}\n"
            )
        };
        let cases = [
            tool("GET", "https://{a}.example.com/x", "\"a\"", ""),
            tool("GET", "https://api.example.com{a}", "\"a\"", ""),
            tool("GET", "https://api.example.com/{a}", "", ""),
            tool("GET", "https://api.example.com/{b}", "\"a\"", ""),
            tool("GET", "https://api.example.com/{a", "\"a\"", ""),
            tool("FETCH", "https://api.example.com/{a}", "\"a\"", ""),
            tool("GET", "ftp://api.example.com/{a}", "\"a\"", ""),
            tool(
                "GET",
                "https://api.example.com/{a}",
                "\"a\"",
                "headers = { X = \"${secret:NOPE}\" }\n",
            ),
            tool(
                "GET",
                "https://api.example.com/{a}",
                "\"a\"",
                "headers = { X = \"${secret:OPEN\" }\n",
            ),
            tool(
                "GET",
                "https://api.example.com/{a}",
                "\"a\"",
                "colour = \"red\"\n",
            ),
        ];
        for case in &cases {
            let res = http_toolkit_from_toml("k", case, &secrets(), FakeHttp::new(vec![]));
            assert!(matches!(res, Err(ToolError::Config(_))), "{case}");
        }
        assert!(http_toolkit_from_toml(
            "k",
            &tool("GET", "https://api.example.com/{a}", "\"a\"", ""),
            &secrets(),
            FakeHttp::new(vec![])
        )
        .is_ok());
    }

    #[tokio::test]
    async fn non_2xx_is_error_output() {
        let fake = FakeHttp::new(vec![FakeHttp::reply(
            404,
            "application/json",
            r#"{"error":"no city"}"#,
        )]);
        let out = registry(fake)
            .call(
                "forecast",
                json!({"city":"Atlantis"}),
                &ToolContext::default(),
            )
            .await
            .unwrap();
        assert!(out.is_error);
        assert!(
            out.content.starts_with("HTTP 404\n<untrusted source=")
                && out.content.contains("no city"),
            "{}",
            out.content
        );
    }

    #[tokio::test]
    async fn transport_failure_is_err() {
        let fake = FakeHttp::new(vec![Err(HttpError::new(
            HttpErrorKind::Blocked,
            "10.0.0.1 is not public",
        ))]);
        let out = ToolOutput::from_result(
            registry(fake)
                .call("forecast", json!({"city":"x"}), &ToolContext::default())
                .await,
        );
        assert!(
            out.is_error && out.content.starts_with("blocked: "),
            "{}",
            out.content
        );
    }
}
