use super::Agent;
use crate::built::{Input, RunStream};
use crate::error::KitError;

impl Agent {
    /// Builds the agent and starts a streamed run on `input`.
    /// Output: the stream once the run id is known, or the first build
    /// or setup error. Keep a `BuiltAgent` and call its `stream` instead
    /// to run many times.
    pub async fn stream(self, input: impl Into<Input>) -> Result<RunStream, KitError> {
        let built = self.build().await?;
        built.stream(input).await
    }
}
