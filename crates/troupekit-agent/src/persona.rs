use crate::dsl::{parse_call, Call};
use crate::error::PersonaError;
use crate::schema::compile_output;
use serde::Deserialize;
use serde_json::Value;
use troupekit_providers::model::valid_model_id;
use troupekit_tools::valid_tool_name;

/// Model calls a run may make when the persona sets no `max_turns`.
/// Counts every agent model call, the repair call included;
/// judge calls do not count.
pub const DEFAULT_MAX_TURNS: u32 = 8;

/// Output tokens one agent model call may use when the persona sets no
/// `max_tokens`. Also the output side of the worst-case budget estimate,
/// so a tight `call_usd` needs a smaller `max_tokens`.
pub const DEFAULT_MAX_TOKENS: u32 = 4096;

const MAX_TURNS_RANGE: std::ops::RangeInclusive<i64> = 2..=50;
const MAX_TOKENS_RANGE: std::ops::RangeInclusive<i64> = 1..=200_000;
const MAX_BUDGET_USD: f64 = 1_000_000.0;

/// Spend limits a persona sets for itself, in millionths of a US dollar.
/// `call` caps one model call and `run` the whole run; `None` is no limit.
/// Read from `budget = { call_usd = 0.05, run_usd = 0.5 }`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Budget {
    pub call_usd_micros: Option<u64>,
    pub run_usd_micros: Option<u64>,
}

/// A validated persona: who the agent is, which model and tools it uses,
/// its guardrails, evals, typed output schema, turn cap and budget.
/// `source` names the file it came from, for error messages.
#[derive(Debug, Clone, PartialEq)]
pub struct Persona {
    pub name: String,
    pub display_name: String,
    pub role: String,
    pub goal: String,
    pub instructions: String,
    pub instructions_file: Option<String>,
    pub context: String,
    pub model: String,
    pub tools: Vec<String>,
    pub guardrails: Vec<String>,
    pub evals: Vec<Call>,
    pub output: Option<Value>,
    pub max_turns: u32,
    pub max_tokens: u32,
    pub budget: Budget,
    pub source: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PersonaFile {
    name: String,
    display_name: String,
    role: String,
    goal: String,
    instructions: Option<String>,
    instructions_file: Option<String>,
    #[serde(default)]
    context: String,
    model: String,
    #[serde(default)]
    tools: Vec<String>,
    #[serde(default)]
    guardrails: Vec<String>,
    #[serde(default)]
    evals: Vec<String>,
    output: Option<toml::Table>,
    max_turns: Option<i64>,
    max_tokens: Option<i64>,
    budget: Option<BudgetFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BudgetFile {
    call_usd: Option<f64>,
    run_usd: Option<f64>,
}

impl Persona {
    /// Parses and validates persona TOML `text` read from `file`.
    /// With `instructions_file` set, `instructions` stays empty until
    /// `with_instructions` fills it. Returns the first problem found.
    pub fn parse(file: &str, text: &str) -> Result<Persona, PersonaError> {
        let raw: PersonaFile = toml::from_str(text).map_err(|e| toml_error(file, text, &e))?;
        let err = |field: &str, msg: String| PersonaError::new(file, field, msg);
        if !valid_name(&raw.name) {
            return Err(err(
                "name",
                "must be 1 to 64 of [a-z0-9-], not starting with `-`".into(),
            ));
        }
        for (field, value) in [
            ("display_name", &raw.display_name),
            ("role", &raw.role),
            ("goal", &raw.goal),
        ] {
            if value.trim().is_empty() {
                return Err(err(field, "must not be blank".into()));
            }
        }
        let instructions = match (raw.instructions, &raw.instructions_file) {
            (Some(_), Some(_)) => {
                return Err(err(
                    "instructions",
                    "set instructions or instructions_file, not both".into(),
                ))
            }
            (None, None) => {
                return Err(err(
                    "instructions",
                    "set instructions or instructions_file".into(),
                ))
            }
            (Some(text), None) if text.trim().is_empty() => {
                return Err(err("instructions", "must not be blank".into()))
            }
            (Some(text), None) => text,
            (None, Some(path)) if !valid_relative_path(path) => {
                return Err(err(
                    "instructions_file",
                    format!("`{path}` must be a relative path of [A-Za-z0-9_./-] without `..`"),
                ))
            }
            (None, Some(_)) => String::new(),
        };
        if !valid_model(&raw.model) {
            return Err(err(
                "model",
                format!(
                    "`{}` must be vendor/model or a role alias of [a-z0-9_-]",
                    raw.model
                ),
            ));
        }
        for (i, tool) in raw.tools.iter().enumerate() {
            if !valid_tool_name(tool) {
                return Err(err(
                    &format!("tools[{i}]"),
                    format!("`{tool}` must be 1 to 64 of [A-Za-z0-9_-]"),
                ));
            }
            if raw.tools[..i].contains(tool) {
                return Err(err(&format!("tools[{i}]"), format!("`{tool}` repeats")));
            }
        }
        if let Some(i) = raw.guardrails.iter().position(|g| g.trim().is_empty()) {
            return Err(err(&format!("guardrails[{i}]"), "must not be blank".into()));
        }
        let mut evals = Vec::with_capacity(raw.evals.len());
        for (i, src) in raw.evals.iter().enumerate() {
            let call = parse_call(src).and_then(|c| check_judge(&c).map(|_| c));
            evals.push(call.map_err(|e| err(&format!("evals[{i}]"), e))?);
        }
        let output = raw
            .output
            .as_ref()
            .map(compile_output)
            .transpose()
            .map_err(|e| err("output", e))?;
        let max_turns = match raw.max_turns {
            None => DEFAULT_MAX_TURNS,
            Some(n) if MAX_TURNS_RANGE.contains(&n) => u32::try_from(n).expect("range fits in u32"),
            Some(n) => return Err(err("max_turns", format!("{n} is outside 2 to 50"))),
        };
        let max_tokens = match raw.max_tokens {
            None => DEFAULT_MAX_TOKENS,
            Some(n) if MAX_TOKENS_RANGE.contains(&n) => {
                u32::try_from(n).expect("range fits in u32")
            }
            Some(n) => return Err(err("max_tokens", format!("{n} is outside 1 to 200000"))),
        };
        let budget = match raw.budget {
            None => Budget::default(),
            Some(b) => Budget {
                call_usd_micros: usd_micros(b.call_usd).map_err(|e| err("budget.call_usd", e))?,
                run_usd_micros: usd_micros(b.run_usd).map_err(|e| err("budget.run_usd", e))?,
            },
        };
        Ok(Persona {
            name: raw.name,
            display_name: raw.display_name,
            role: raw.role,
            goal: raw.goal,
            instructions,
            instructions_file: raw.instructions_file,
            context: raw.context,
            model: raw.model,
            tools: raw.tools,
            guardrails: raw.guardrails,
            evals,
            output,
            max_turns,
            max_tokens,
            budget,
            source: file.to_string(),
        })
    }

    /// Fills `instructions` with `text` read from `instructions_file`.
    /// Returns the persona, or an error naming `instructions_file` when the
    /// text is blank.
    pub fn with_instructions(mut self, text: String) -> Result<Persona, PersonaError> {
        if text.trim().is_empty() {
            return Err(PersonaError::new(
                &self.source,
                "instructions_file",
                "file is blank",
            ));
        }
        self.instructions = text;
        Ok(self)
    }

    /// Builds the system prompt in a fixed order: identity and role, goal,
    /// instructions, context, guardrail rules, then the output contract
    /// with the JSON Schema when the persona has typed output.
    pub fn system_prompt(&self) -> String {
        assert!(
            !self.instructions.trim().is_empty(),
            "persona instructions are loaded before use"
        );
        let mut parts = vec![
            format!("You are {}. {}", self.display_name, self.role.trim()),
            format!("Goal: {}", self.goal.trim()),
            self.instructions.trim().to_string(),
        ];
        if !self.context.trim().is_empty() {
            parts.push(format!("Context:\n{}", self.context.trim()));
        }
        if !self.guardrails.is_empty() {
            let rules: Vec<String> = self
                .guardrails
                .iter()
                .map(|g| format!("- {}", g.trim()))
                .collect();
            parts.push(format!("Rules:\n{}", rules.join("\n")));
        }
        if let Some(schema) = &self.output {
            parts.push(format!(
                "Output: reply with only one JSON object that matches this JSON Schema:\n{schema}"
            ));
        }
        parts.join("\n\n")
    }
}

/// Reports whether `name` is a safe persona name: 1 to 64 of
/// `[a-z0-9-]`, not starting with `-`.
/// Loaders check it before joining any path.
pub fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && !name.starts_with('-')
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
}

/// Reports whether `path` is a safe relative file path: 1 to 200 of
/// `[A-Za-z0-9_./-]`, not starting with `/` and with no `..` segment.
/// Persona sources check it before reading.
pub fn valid_relative_path(path: &str) -> bool {
    !path.is_empty()
        && path.len() <= 200
        && !path.starts_with('/')
        && path.split('/').all(|seg| !seg.is_empty() && seg != "..")
        && path
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"_./-".contains(&b))
}

fn valid_model(model: &str) -> bool {
    match model.split_once('/') {
        Some((vendor, rest)) => !vendor.is_empty() && !rest.is_empty() && valid_model_id(model),
        None => {
            let mut chars = model.chars();
            chars.next().is_some_and(|c| c.is_ascii_lowercase())
                && model.len() <= 64
                && chars
                    .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
        }
    }
}

fn check_judge(call: &Call) -> Result<(), String> {
    if call.name != "judge" {
        return Ok(());
    }
    let criteria_ok = call
        .args
        .first()
        .and_then(Value::as_str)
        .is_some_and(|c| !c.trim().is_empty());
    let score_ok = call
        .args
        .get(1)
        .and_then(Value::as_f64)
        .is_some_and(|s| (0.0..=10.0).contains(&s));
    let model_ok = call
        .args
        .get(2)
        .is_none_or(|m| m.as_str().is_some_and(valid_model));
    if !criteria_ok || !score_ok || !model_ok || call.args.len() > 3 {
        return Err("judge takes ('criteria', min_score 0 to 10[, 'model'])".into());
    }
    Ok(())
}

fn usd_micros(usd: Option<f64>) -> Result<Option<u64>, String> {
    let Some(usd) = usd else { return Ok(None) };
    if !usd.is_finite() || !(0.0..=MAX_BUDGET_USD).contains(&usd) {
        return Err(format!(
            "{usd} must be between 0 and {MAX_BUDGET_USD} US dollars"
        ));
    }
    Ok(Some((usd * 1_000_000.0).round() as u64))
}

/// Turns a TOML decode error `e` on `text` read from `file` into a
/// `PersonaError` naming the field: the missing or unknown key, else the
/// key on the failing line, else `toml`. Workflow files reuse it.
pub fn toml_error(file: &str, text: &str, e: &toml::de::Error) -> PersonaError {
    let message = e.message().to_string();
    let named = ["missing field `", "unknown field `"]
        .iter()
        .find_map(|p| message.strip_prefix(p))
        .and_then(|rest| rest.split('`').next());
    let at_span = e.span().and_then(|span| {
        let start = text.get(..span.start)?.rfind('\n').map_or(0, |i| i + 1);
        let line = text[start..].lines().next()?;
        let key = line.split_once('=')?.0.trim();
        let key = key.trim_matches('"');
        (!key.is_empty()
            && key
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'.'))
        .then_some(key)
    });
    let field = named.or(at_span).unwrap_or("toml").to_string();
    PersonaError::new(file, &field, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    pub(crate) const FULL: &str = r#"
name = "coach"
display_name = "Coach"
role = "Plans training weeks."
goal = "A safe weekly plan."
instructions = "Plan the week."
context = "The owner trains at home."
model = "anthropic/claude-opus-5-5"
tools = ["web_search"]
guardrails = ["No medical advice."]
evals = ["min_items(actions, 1)", "judge('Plan is safe', 7)"]
max_turns = 6
budget = { call_usd = 0.05, run_usd = 0.5 }

[output]
actions = "list<{name, minutes: int}>"
"#;

    #[test]
    fn parses_full_persona() {
        let p = Persona::parse("lib/coach.toml", FULL).unwrap();
        assert_eq!(
            (p.name.as_str(), p.max_turns, p.tools.len()),
            ("coach", 6, 1)
        );
        assert_eq!(
            p.budget,
            Budget {
                call_usd_micros: Some(50_000),
                run_usd_micros: Some(500_000)
            }
        );
        assert_eq!(
            p.evals[1],
            Call {
                name: "judge".into(),
                args: vec![json!("Plan is safe"), json!(7)]
            }
        );
        assert_eq!(p.output.unwrap()["required"], json!(["actions"]));
        let min = Persona::parse(
            "m.toml",
            "name='m'\ndisplay_name='M'\nrole='r'\ngoal='g'\ninstructions='i'\nmodel='fast'",
        )
        .unwrap();
        assert_eq!(
            (min.max_turns, min.max_tokens, min.budget, min.output),
            (
                DEFAULT_MAX_TURNS,
                DEFAULT_MAX_TOKENS,
                Budget::default(),
                None
            )
        );
    }

    #[test]
    fn system_prompt_order() {
        let prompt = Persona::parse("c.toml", FULL).unwrap().system_prompt();
        let at = |s: &str| {
            prompt
                .find(s)
                .unwrap_or_else(|| panic!("{s} missing in {prompt}"))
        };
        assert!(at("You are Coach. Plans training weeks.") < at("Goal: A safe weekly plan."));
        assert!(
            at("Goal:") < at("Plan the week.") && at("Plan the week.") < at("Context:\nThe owner")
        );
        assert!(
            at("Context:") < at("Rules:\n- No medical advice.")
                && at("Rules:") < at("Output: reply with only one JSON object")
        );
        assert!(prompt.contains("\"additionalProperties\":false"));
    }

    #[test]
    fn errors_name_file_and_field() {
        let cases = [
            (FULL.replace("goal = ", "colour = 'red'\ngoal = "), "colour"),
            (
                FULL.replace("role = \"Plans training weeks.\"\n", ""),
                "role",
            ),
            (
                FULL.replace("anthropic/claude-opus-5-5", "Bad Model"),
                "model",
            ),
            (FULL.replace("max_turns = 6", "max_turns = 1"), "max_turns"),
            (
                FULL.replace("max_turns = 6", "max_turns = \"8\""),
                "max_turns",
            ),
            (
                FULL.replace("max_turns = 6", "max_turns = 6\nmax_tokens = 0"),
                "max_tokens",
            ),
            (
                FULL.replace("tools = [\"web_search\"]", "tools = \"x\""),
                "tools",
            ),
            (
                FULL.replace("max_turns = 6", "max_turns = 6\nnonsense line"),
                "toml",
            ),
            (
                FULL.replace("call_usd = 0.05", "call_usd = -1.0"),
                "budget.call_usd",
            ),
            (
                FULL.replace("judge('Plan is safe', 7)", "judge('Plan is safe', 70)"),
                "evals[1]",
            ),
            (
                FULL.replace("min_items(actions, 1)", "not a call"),
                "evals[0]",
            ),
            (FULL.replace("minutes: int", "minutes: lst"), "output"),
            (
                FULL.replace("[\"web_search\"]", "[\"a\", \"a\"]"),
                "tools[1]",
            ),
            (
                FULL.replace("name = \"coach\"", "name = \"Coach!\""),
                "name",
            ),
            (
                FULL.replace(
                    "instructions = \"Plan the week.\"",
                    "instructions_file = \"../secret.md\"",
                ),
                "instructions_file",
            ),
            (
                FULL.replace(
                    "instructions = \"Plan the week.\"",
                    "instructions = \"a\"\ninstructions_file = \"a.md\"",
                ),
                "instructions",
            ),
        ];
        for (text, field) in cases {
            let e = Persona::parse("lib/coach.toml", &text).unwrap_err();
            assert_eq!(
                (e.file.as_str(), e.field.as_str()),
                ("lib/coach.toml", field),
                "{e}"
            );
            assert!(
                e.to_string()
                    .starts_with(&format!("lib/coach.toml: {field}: ")),
                "{e}"
            );
        }
    }

    #[test]
    fn instructions_file_is_relative_only() {
        for good in ["prompts/coach.md", "coach.md", "a-b_c.txt"] {
            assert!(valid_relative_path(good), "{good}");
        }
        for bad in [
            "/etc/passwd",
            "../x.md",
            "a/../../x",
            "a//b",
            "",
            "a\\b",
            "c:x",
        ] {
            assert!(!valid_relative_path(bad), "{bad}");
        }
        let text = FULL.replace(
            "instructions = \"Plan the week.\"",
            "instructions_file = \"prompts/coach.md\"",
        );
        let p = Persona::parse("c.toml", &text).unwrap();
        assert_eq!(
            (p.instructions.as_str(), p.instructions_file.as_deref()),
            ("", Some("prompts/coach.md"))
        );
        assert!(p.clone().with_instructions("  ".into()).is_err());
        assert_eq!(
            p.with_instructions("Plan it.".into()).unwrap().instructions,
            "Plan it."
        );
    }
}
