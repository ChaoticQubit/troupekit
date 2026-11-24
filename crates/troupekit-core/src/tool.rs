use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Name of the hosted web search tool, on the wire and in tool registries.
/// Cost meters count its hosted calls.
/// `HostedTool::WebSearch` returns it from `name`.
pub const WEB_SEARCH: &str = "web_search";

/// A server-side tool a vendor can run itself instead of the local tool
/// of the same name. `max_uses` caps calls per agent run; fetch also caps
/// the page tokens it reads. Serializes with a `kind` tag in snake case.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum HostedTool {
    WebSearch { max_uses: u32 },
    WebFetch { max_uses: u32, max_content_tokens: u32 },
}

impl HostedTool {
    /// The tool name vendors use for this server-side tool and the local
    /// tool that stands in for it must carry: `web_search` or `web_fetch`.
    /// Returns the name.
    pub fn name(&self) -> &'static str {
        match self {
            HostedTool::WebSearch { .. } => WEB_SEARCH,
            HostedTool::WebFetch { .. } => "web_fetch",
        }
    }

    /// Returns the calls per run this tool allows.
    /// The agent sends what is left and drops the tool at zero.
    /// Same for every kind.
    pub fn max_uses(&self) -> u32 {
        match self {
            HostedTool::WebSearch { max_uses } | HostedTool::WebFetch { max_uses, .. } => *max_uses,
        }
    }

    /// Returns a copy allowing `uses` calls, for the calls left in a run.
    /// Every other setting stays.
    /// Pair it with `max_uses`.
    pub fn with_max_uses(self, uses: u32) -> Self {
        match self {
            HostedTool::WebSearch { .. } => HostedTool::WebSearch { max_uses: uses },
            HostedTool::WebFetch { max_content_tokens, .. } => HostedTool::WebFetch { max_uses: uses, max_content_tokens },
        }
    }
}

/// A tool as a model sees it: its name, what it does and a JSON Schema
/// for its arguments. `hosted` marks a tool a vendor may run server-side;
/// wires without that send it as a plain tool and the local one runs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hosted: Option<HostedTool>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn round_trips_as_plain_fields() {
        let t = ToolSpec { name: "search".into(), description: "Web search".into(), schema: json!({"type":"object"}), hosted: None };
        let v = serde_json::to_value(&t).unwrap();
        assert_eq!(v, json!({"name":"search","description":"Web search","schema":{"type":"object"}}));
        assert_eq!(serde_json::from_value::<ToolSpec>(v).unwrap(), t);
    }

    #[test]
    fn hosted_tools_know_their_names_and_uses() {
        let s = HostedTool::WebSearch { max_uses: 5 };
        assert_eq!((s.name(), s.max_uses()), ("web_search", 5));
        let f = HostedTool::WebFetch { max_uses: 5, max_content_tokens: 8000 };
        assert_eq!(f.with_max_uses(2), HostedTool::WebFetch { max_uses: 2, max_content_tokens: 8000 });
        assert_eq!(f.name(), "web_fetch");
    }
}
