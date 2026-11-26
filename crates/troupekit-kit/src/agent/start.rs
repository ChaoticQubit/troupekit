use super::Agent;
use crate::built::{BuiltAgent, Input};
use crate::error::KitError;
use troupekit_core::ids::RunId;

impl Agent {
    /// Builds the agent and starts a run on `input` with no model call.
    /// Returns the built agent and the run id to store for `resume` or
    /// `resume_for`. Fails like `build`, then like `BuiltAgent::start`.
    pub async fn start(self, input: impl Into<Input>) -> Result<(BuiltAgent, RunId), KitError> {
        let built = self.build().await?;
        let run_id = built.start(input).await?;
        Ok((built, run_id))
    }
}
