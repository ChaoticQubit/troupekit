use crate::agent::Agent;
use crate::built::{describe, member_settings, BuiltAgent};
use crate::error::KitError;
use async_trait::async_trait;
use serde_json::Value;
use troupekit_agent::delegate::{AgentTool, DelegateSettings, DelegationTools};
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput, ToolSpec};

const MEMBERS_MAX: usize = 64;
const NEST_MAX: u32 = 16;

/// The member agents of a leader: agents called as tools and coworkers
/// for the delegation tools, built with the leader. After the build they
/// are gone and only the names of the tools they became remain.
#[derive(Clone, Default)]
pub(crate) struct Members {
    agent_tools: Vec<Agent>,
    delegates: Vec<Agent>,
    tool_names: Vec<String>,
    nest: u32,
}

impl Members {
    /// Returns true when no member waits to be built.
    /// An agent without the 0.5 methods always returns true.
    pub(crate) fn is_empty(&self) -> bool {
        self.agent_tools.is_empty() && self.delegates.is_empty()
    }

    /// Returns the names of the tools the members became at build.
    /// Empty before the build and for an agent without members.
    pub(crate) fn tool_names(&self) -> &[String] {
        &self.tool_names
    }
}

impl Agent {
    /// Adds `member` as a tool named after it. The model sends `{task,
    /// context?}`; the member runs to its end inside the call, with its own
    /// model, tools, limits and ledger, and its answer comes back fenced as
    /// untrusted text. Transient member failures come back as a tool error.
    pub fn agent_tool(mut self, member: Agent) -> Self {
        self.members.agent_tools.push(member);
        self
    }

    /// Adds `coworkers` behind two tools, `delegate_work` and
    /// `ask_question`, each naming one coworker. Coworkers run like
    /// `agent_tool` members. `build` refuses an empty set, duplicate names
    /// and more than 64 members.
    pub fn delegates(mut self, coworkers: impl IntoIterator<Item = Agent>) -> Self {
        self.members.delegates.extend(coworkers);
        self
    }

    /// Builds every member and registers the tools they become. Returns
    /// the leader with no members left and the tool names recorded.
    /// Fails with the first member build error or a members limit.
    pub(crate) async fn attach_members(mut self) -> Result<Agent, KitError> {
        let members = std::mem::take(&mut self.members);
        let leader = self.name().to_string();
        assert!(
            !members.is_empty(),
            "attach runs only for an agent with members"
        );
        if members.nest >= NEST_MAX {
            let m = format!("members nest deeper than {NEST_MAX} agents");
            return Err(KitError::config(&leader, m));
        }
        if members.agent_tools.len() + members.delegates.len() > MEMBERS_MAX {
            let m = format!("more than {MEMBERS_MAX} member agents");
            return Err(KitError::config(&leader, m));
        }
        let mut names = Vec::new();
        for member in members.agent_tools {
            let built = build_member(member, members.nest, &leader).await?;
            names.push(built.persona().name.clone());
            self = self.tool_arc(built.agent_tool().clone());
        }
        if !members.delegates.is_empty() {
            let mut coworkers = Vec::with_capacity(members.delegates.len());
            for member in members.delegates {
                let built = build_member(member, members.nest, &leader).await?;
                let persona = built.persona();
                let settings = member_settings(built.limits(), built.store());
                let tool = AgentTool::new(
                    &persona.name,
                    &describe(persona),
                    persona.clone(),
                    built.io(),
                    settings,
                )?;
                coworkers.push((persona.name.clone(), tool));
            }
            let set = DelegationTools::new(coworkers, DelegateSettings::default())?;
            for tool in set.tools() {
                names.push(tool.spec().name.clone());
                self = self.tool_arc(tool);
            }
        }
        assert!(!names.is_empty(), "every member became a tool");
        self.members = Members {
            tool_names: names,
            nest: members.nest,
            ..Members::default()
        };
        assert!(self.members.is_empty());
        Ok(self)
    }
}

async fn build_member(mut member: Agent, nest: u32, leader: &str) -> Result<BuiltAgent, KitError> {
    member.members.nest = nest + 1;
    let built = Box::pin(member.build()).await?;
    if built.has_preamble() {
        let m = format!(
            "member `{}` has memory or knowledge, which member runs do not read yet",
            built.persona().name
        );
        return Err(KitError::config(leader, m));
    }
    Ok(built)
}

#[async_trait]
impl Tool for BuiltAgent {
    fn spec(&self) -> &ToolSpec {
        self.agent_tool().spec()
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        if self.has_preamble() {
            return Err(ToolError::Failed(format!(
                "agent `{}` has memory or knowledge, which a tool call does not read yet",
                self.persona().name
            )));
        }
        self.agent_tool().call(args, ctx).await
    }
}
